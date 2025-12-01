import logging
import os
import re
import string
from pathlib import Path
from typing import Dict, Tuple

from bs4 import BeautifulSoup

from config import CONFIG


def normalize(text: str) -> str:
    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def remove_punc(s):
        return "".join(ch for ch in s if ch not in string.punctuation)

    def white_space_fix(s):
        return " ".join(s.split())

    def lower(s):
        return str(s).lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def strip_links(text: str) -> str:
    """
    Remove HTML anchor tags from a text string.

    Args:
        text (str): Input HTML-formatted string.

    Returns:
        str: String with anchor tags removed.
    """
    return re.sub(r"</?a[^>]*>", "", text).strip()


def convert_seconds(total_seconds: float) -> Tuple[int, int, int]:
    """
    Convert a time duration from seconds to hours, minutes, and seconds.

    Args:
        total_seconds (float): Duration in seconds.

    Returns:
        Tuple[int, int, int]: Hours, minutes, seconds.
    """
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return hours, minutes, seconds


def setup_logging() -> logging.Logger:
    """
    Set up logging to file and console using configuration in CONFIG.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_file_path = CONFIG.energy_dir / "experiment.log"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logging.basicConfig(
        level=CONFIG.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("energy_eval")


def ensure_config_dirs() -> None:
    """
    Ensure that all required directories defined in CONFIG exist.
    """
    dirs_to_check = [
        CONFIG.wiki_dir,
        CONFIG.corpus_cache.parent,
        CONFIG.tfidf_cache.parent,
        CONFIG.index_cache.parent,
        CONFIG.energy_dir,
        CONFIG.result_dir,
        CONFIG.viz_dir,
    ]
    for d in dirs_to_check:
        d.mkdir(parents=True, exist_ok=True)


def extract_first_paragraph(html_content):
    """Extract the first paragraph from HTML content."""
    soup = BeautifulSoup(html_content, "html.parser")
    first_p = soup.find("p")
    return first_p.get_text().strip() if first_p else ""
