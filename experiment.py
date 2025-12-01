import gc
import json
import logging
import os.path
import re
import string
import time
import warnings
from pathlib import Path
import pandas as pd

# Supress ollama http logs
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
# Supress codecarbon warnings
logger = logging.getLogger("codecarbon")
logger.setLevel(logging.ERROR)
logger.propagate = False

import torch
from datasets import Dataset
from tqdm import tqdm

from config import CONFIG
from inference import inference
from prompts import build_prompt
from retrieval import retrieve_from_wiki_dump
from scorers import exact_match, f1_score
from utils import (
    convert_seconds,
    ensure_config_dirs,
    setup_logging,
)

# Remove existing handlers
for handler in logger.handlers:
    logger.removeHandler(handler)

logger = setup_logging()
ensure_config_dirs()

CONCURRENCY = 8
PROMPT_BUFFER = []
RESULT_BUFFER = []


def run(file_suffix: None | str = "") -> None:
    """Main experiment runner."""
    data_dir = "data"
    start_time = time.time()

    logger.info(f"Starting experiment with config:\n{CONFIG}")
    logger.info(f"Using device: {CONFIG.device}")

    # Load dataset
    try:
        dataset = load_config_dataset(data_dir)
        logger.info(f"Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Dataset loading failed: {str(e)}")
        return

    # Preload Wikipedia if needed
    wiki_data = None

    for model_name, provider in CONFIG.model_types.items():
        model_start_time = time.time()
        logger.info(f"\n{'=' * 60}\nRunning model: {model_name}\n{'=' * 60}")

        tokenizer, model = None, None
        if not CONFIG.retrieval_only:
            tokenizer = True
            model = model_name

        model_modes = CONFIG.modes.get(model_name, {})
        logger.info(f"\n{'+' * 30}\nRunning modes: {model_modes}\n{'+' * 30}")
        for mode_tag, include_passage in model_modes.items():
            run_model_mode(
                model_name,
                mode_tag,
                include_passage,
                dataset,
                wiki_data,
                file_suffix,
            )

        cleanup_resources(model, tokenizer)

        model_time = time.time() - model_start_time
        logger.info(f"Completed {model_name} in {format_time(model_time)}")

    total_time = time.time() - start_time
    logger.info(
        f"\n{'=' * 60}\nExperiment completed in {format_time(total_time)}\n{'=' * 60}"
    )


def load_config_dataset(data_dir: Path) -> Dataset:
    """Load dataset based on configuration."""
    dataset_path = Path(os.path.join(data_dir, CONFIG.dataset_file))

    if dataset_path.exists():
        logger.info(f"Loading dataset from {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        return Dataset.from_list(data[: CONFIG.dataset_size])

    raise ValueError(f"Unsupported dataset: {CONFIG.dataset_name}")


def cleanup_resources(model, tokenizer) -> None:
    """Release model resources and clean memory."""
    if model:
        del model
    if tokenizer:
        del tokenizer
    if CONFIG.device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def format_time(seconds: float) -> str:
    """Convert seconds to human-readable format."""
    h, m, s = convert_seconds(seconds)
    return f"{h}h {m}m {s}s"


def run_model_mode(
        model_name: str,
        mode_tag: str,
        include_passage: bool,
        dataset,
        wiki_data: tuple,
        file_suffix: None | str = "",
) -> None:
    # Run evaluation for a specific model and mode, with concurrent inference and batched writes
    csv_path = (
            CONFIG.result_dir
            / f"nq_{model_name.split('/')[-1].replace(':', '-')}_{mode_tag}"
              f"{'_1000' if 'mini' in CONFIG.dataset_file else ''}"
              f"{'_think' if CONFIG.think else ''}"
              f"{'_long' if CONFIG.gold == True and 'r' in mode_tag else ''}"
              f"{'_first' if CONFIG.gold == False and 'r' in mode_tag else ''}"
              f"{'_' + file_suffix if file_suffix != '' else ''}.csv"
    )
    logger.info(f"Saving results to: {csv_path}")
    start_idx = get_resume_index(csv_path)
    overall_t0 = time.time()

    # Pre-build prompts + metadata
    prompt_t0 = time.time()
    jobs = []
    total_ret = {"duration": 0.0, "energy_consumed": 0.0, "emissions": 0.0}
    for idx in range(start_idx, len(dataset)):
        sample = dataset[idx]
        ret_metrics = None
        if mode_tag == "q+r":
            ctx, ret_metrics = retrieve_context(sample, wiki_data)
            sample = {**sample, "retrieved_context": ctx}

            if ret_metrics:
                for k in total_ret:
                    total_ret[k] += ret_metrics.get(k, 0.0)
        prompt = build_prompt(sample, include_passage)
        jobs.append((idx, sample, prompt, ret_metrics))

    logger.info(
        f"Built {len(jobs)} prompts in {format_time(time.time() - prompt_t0)} "
        f"(+ retrieval: {format_time(total_ret['duration'])}, "
        f"{total_ret['energy_consumed']:.4f} kWh, {total_ret['emissions']:.4f} kg)"
    )

    result_buffer = []

    # Spin up LM
    _ = inference("Ready?", model_name, mode_tag, 'ollama')

    # Loop through all prompts for current model
    for idx, sample, prompt, ret_metrics in tqdm(
            jobs, desc=f"{model_name} ({mode_tag})", total=len(jobs)
    ):
        full_output, inf_metrics = inference(prompt, model_name, mode_tag, 'ollama')

        llm_prediction = extract_prediction(full_output)

        em = -1
        f1 = -1
        for short_answer in set(
                [answer[0] for answer in sample["short_answers"] if len(answer) > 0]
        ):
            instance_em = exact_match(llm_prediction, short_answer)
            if instance_em > em:
                em = instance_em
                sample["answer"] = short_answer
            instance_f1 = f1_score(llm_prediction, short_answer)
            if instance_f1 > f1:
                f1 = instance_f1
                sample["answer"] = short_answer

        llm_prediction = llm_prediction.split("think>")[-1].lower().strip()
        # remove punctuation
        llm_prediction = llm_prediction.translate(
            str.maketrans("", "", string.punctuation)
        )
        # remove articles
        llm_prediction = re.sub(r"\b(a|an|the)\b", " ", llm_prediction)
        # collapse whitespace
        llm_prediction = re.sub(r"\s+", " ", llm_prediction)

        row = {
            "qid": idx,
            "prompt": prompt,
            "original_pred": full_output.replace(",", " ")
            .replace("  ", " ")
            .replace("\n", " "),
            "pred": llm_prediction,
            "gold": sample["answer"],
            "em": em,
            "f1": f1,
            "inference_duration (s)": inf_metrics["duration"],
            "inference_energy_consumed (kWh)": inf_metrics["energy_consumed"],
            "inference_emissions (kg)": inf_metrics["emissions"],
            "retrieval_duration (s)": (
                ret_metrics.get("duration") if ret_metrics else 0.0
            ),
            "retrieval_energy_consumed (kWh)": (
                ret_metrics.get("energy_consumed") if ret_metrics else 0.0
            ),
            "retrieval_emissions (kg)": (
                ret_metrics.get("emissions") if ret_metrics else 0.0
            ),
        }

        result_buffer.append(row)

        if len(result_buffer) >= CONFIG.batch_size:
            save_results(result_buffer, csv_path)
            result_buffer.clear()

    if result_buffer:
        save_results(result_buffer, csv_path)

    logger.info(
        f"Completed {mode_tag} for {model_name} in {format_time(time.time() - overall_t0)}"
    )


def get_resume_index(csv_path: Path) -> int:
    """Get resume index from existing results file."""
    if not csv_path.exists():
        return 0

    try:
        existing_df = pd.read_csv(csv_path)
        return 0 if existing_df.empty else int(existing_df["qid"].max()) + 1
    except Exception as e:
        logger.warning(f"Couldn't read existing CSV: {str(e)}")
        return 0


def retrieve_context(sample: dict, wiki_data: tuple | None) -> tuple[str, dict]:
    """Retrieve context passage based on dataset type."""
    retrieval_metrics = {
        "duration": 0.0,
        "energy_consumed": 0.0,
        "emissions": 0.0,
    }

    if wiki_data is not None:
        docs, titles, vectorizer, tfidf_matrix, inv_index = wiki_data
        _, ret_metrics = retrieve_from_wiki_dump(
            sample["question"], vectorizer, tfidf_matrix, titles, inv_index
        )
        retrieval_metrics.update(ret_metrics)

    # Get gold/1st paragraph for context
    if CONFIG.gold:
        context = sample["long_answer"]
    else:
        context = sample["first_paragraph"]
    return context, retrieval_metrics


def extract_prediction(full_output: str) -> str:
    # If model uses <think>...True, extract the last line or the final word
    if "<think>" in full_output:
        full_output = full_output.split("</think>")[-1]
    # If model doesn't think and prepends reply with "Answer:" split on that instead
    elif "Answer: " in full_output:
        full_output = full_output.split("Answer:")[-1]
    for prefix in ["The answer is:", "Answer:", "answer:", "A:", "a:"]:
        if full_output.lower().startswith(prefix.lower()):
            full_output = full_output[len(prefix):]

    # Fallback is taking the original output
    return full_output.strip()


def save_results(results: list[dict], csv_path: Path) -> None:
    """Save results to CSV file."""
    if not results:
        return

    df = pd.DataFrame(results)
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user")
    except Exception as error:
        logger.exception(f"Critical error: {str(error)}")
        raise
