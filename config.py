from dataclasses import dataclass, field
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ExperimentConfig:
    model_types: dict[str, str]
    dataset_name: str
    dataset_file: str
    config: str
    split: str
    batch_size: int
    device: str
    modes: dict[str, dict]
    think: bool
    gold: bool
    wiki_dir: Path
    corpus_cache: Path
    tfidf_cache: Path
    index_cache: Path
    token_pattern: str
    energy_dir: Path
    result_dir: Path
    data_dir: Path
    viz_dir: Path
    retrieval_only: bool
    seed: int
    dataset_size: int
    log_level: str = "INFO"
    prompt_templates: dict[str, str] = field(
        default_factory=lambda: {
            "natural_questions_parsed": {
                "with_context": (
                    "Using the provided context, answer the question. Provide concise, direct answers without excessive reasoning or repetition. "
                    "Context: {context}\n"
                    "Question: {question}\n"
                    "Answer: "
                ),
                "without_context": (
                    "Answer the question. Provide concise, direct answers without excessive reasoning or repetition. "
                    "Question: {question}\n"
                    "Answer: "
                ),
            },
        }
    )


CONFIG = ExperimentConfig(
    model_types={
        "qwen3:0.6b": "ollama",
        "qwen3:1.7b": "ollama",
        "qwen3:4b": "ollama",
        "qwen3:8b": "ollama",
        "qwen3:14b": "ollama",
        "qwen3:32b": "ollama",
    },
    modes={
        "qwen3:0.6b": {"q": False, "q+r": True},
        "qwen3:1.7b": {"q": False, "q+r": True},
        "qwen3:4b": {"q": False, "q+r": True},
        "qwen3:8b": {"q": False, "q+r": True},
        "qwen3:14b": {"q": False, "q+r": True},
        "qwen3:32b": {"q": False},
    },
    dataset_name="hugosousa/natural_questions_parsed",
    dataset_file="nq_mini_1000.jsonl",
    config="fullwiki",
    split="validation",
    batch_size=4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    think=True,
    gold=True,
    wiki_dir=Path("enwiki-20200101"),
    corpus_cache=Path(f"cache/wiki_2wiki.pkl"),
    tfidf_cache=Path("cache/tfidf_2wiki.pkl"),
    index_cache=Path("cache/index_2wiki.pkl"),
    token_pattern=r"(?u)\b\w+\b",
    energy_dir=Path("results/energy"),
    result_dir=Path("results"),
    data_dir=Path("data"),
    viz_dir=Path("plots"),
    retrieval_only=False,
    seed=42,
    dataset_size=1000,
)
