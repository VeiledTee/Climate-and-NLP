import bz2
import json
import logging
import re
from collections import Counter, defaultdict
from typing import Any
import random

import joblib
from codecarbon import EmissionsTracker
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    HashingVectorizer,
    TfidfVectorizer,
)

from config import CONFIG
from utils import normalize, strip_links

logger = logging.getLogger("codecarbon")
logger.setLevel(logging.ERROR)
logger.propagate = False
random.seed(CONFIG.seed)


def retrieve_from_wiki_dump(
    question: str,
    vectorizer: HashingVectorizer,
    tfidf_matrix: Any,
    titles: list[str],
    inv_index: dict[str, list[int]],
) -> tuple[list[str], dict[str, float]]:
    """
    Retrieve top candidate documents for a HotpotQA-style question.

    Args:
        question (str): Input question.
        vectorizer (HashingVectorizer): Pre-built TF-IDF vectorizer.
        tfidf_matrix (sparse matrix): Document-term matrix for all Wikipedia docs.
        titles (list[str]): Page titles aligned with the TF-IDF matrix.
        inv_index (dict[str, list[int]]): Inverted index mapping tokens/bigrams to doc IDs.

    Returns:
        top_titles (list[str]): Top-10 candidate document titles.
        metrics (dict[str, float]): Duration, energy_consumed, and emissions.
    """
    tokens = re.findall(CONFIG.token_pattern, normalize(question))
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    ngrams = tokens + [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]

    with EmissionsTracker(save_to_file=False) as tracker:
        counter = Counter()
        for ng in ngrams:
            counter.update(inv_index.get(ng, []))
        cand_ids = [doc_id for doc_id, _ in counter.most_common(5000)]

        q_vec = vectorizer.transform([question])

        if cand_ids:
            sub_mat = tfidf_matrix[cand_ids]
            scores = (q_vec @ sub_mat.T).toarray().flatten()
            top_indices = scores.argsort()[-10:][::-1]
            top_titles = [titles[cand_ids[i]] for i in top_indices]
        else:
            top_titles = []

    data = tracker.final_emissions_data
    return top_titles, {
        "duration": data.duration,
        "energy_consumed": data.energy_consumed,
        "emissions": data.emissions,
    }
