import logging
import os

import torch
from codecarbon import EmissionsTracker
from ollama import chat
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from config import CONFIG

logger = logging.getLogger("codecarbon")
logger.setLevel(logging.ERROR)
logger.propagate = False

# Remove existing handlers
for handler in logger.handlers:
    logger.removeHandler(handler)


def inference_ollama(prompt, model_name):
    resp = chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
        options={
            "temperature": 0.6,
            "top_p": 0.95,
            "num_thread": os.cpu_count(),
        },
        think=CONFIG.think,
    )
    if CONFIG.think:
        if resp.done_reason == "length":
            return f"{resp.message.thinking}"
        else:
            return f"{resp.message.thinking} {resp.message.content}"
    else:
        return resp.message.content


def inference(
    prompt: str,
    model_name: str,
    run_tag: str,
    provider: str,
):
    """
    Run inference with emissions tracking and return generated text with energy metrics.

    Args:
        prompt (str): Input prompt for the model.
        model (PreTrainedModel): Loaded model for inference.
        tokenizer (PreTrainedTokenizer): Tokenizer associated with the model.
        model_name (str): Name of the model (for logging purposes).
        run_tag (str): Tag identifying the run (used in emissions log naming).
        provider (str): The service providing the language model.

    Returns:
        tuple[str, dict[str, float]]: Generated text and energy/emissions data.
    """
    if provider == "ollama":
        try:
            with EmissionsTracker(
                save_to_file=False,
                project_name=f"{CONFIG.dataset_name.split('/')[-1]}_{model_name}_{run_tag}",
                log_level="error",
            ) as tracker:
                text = inference_ollama(prompt, model_name)

            return text, {
                "duration": float(tracker.final_emissions_data.duration),
                "energy_consumed": float(tracker.final_emissions_data.energy_consumed),
                "emissions": float(tracker.final_emissions_data.emissions),
            }
        except Exception as e:
            print(f"Error during {provider} inference: {str(e)}")
            raise
    else:
        print(
            f"Error during inference: Provider in CONFIG.model_types must be 'ollama'."
        )
        raise EnvironmentError
