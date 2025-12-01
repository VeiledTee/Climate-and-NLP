import json
import random
import math
import pandas as pd

from datasets import Dataset, load_dataset

from config import CONFIG

# Use N_SAMPLES from CONFIG, defaulting to 128
N_SAMPLES = CONFIG.dataset_size or 128


def save_jsonl(data: list, path: str):
    """Saves a list of dictionaries to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for example in data:
            json.dump(example, f)
            f.write("\n")


def calculate_stratified_counts(
    full_counts: dict, subset_size: int, total_full_samples: int
) -> dict:
    """Calculates the proportional counts for the subset, ensuring the total is exact."""

    float_targets = {}

    # 1. Calculate the float target size for each type
    for q_type, count in full_counts.items():
        ratio = count / total_full_samples
        float_targets[q_type] = ratio * subset_size

    # 2. Round down all counts (initial floor)
    target_counts_floored = {k: math.floor(v) for k, v in float_targets.items()}

    # 3. Determine how many remaining samples are needed
    remaining_needed = subset_size - sum(target_counts_floored.values())

    # 4. Use fractional parts for tie-breaking to allocate remaining samples
    fractional_parts = {
        k: v - target_counts_floored[k] for k, v in float_targets.items()
    }
    sorted_fractions = sorted(
        fractional_parts.items(), key=lambda item: item[1], reverse=True
    )

    final_counts = target_counts_floored

    # Assign the remaining needed samples one by one
    for i in range(remaining_needed):
        q_type_to_add = sorted_fractions[i][0]
        final_counts[q_type_to_add] += 1

    return final_counts


def process_nq(dataset: Dataset) -> list:
    filtered_dataset = dataset.filter(
        lambda example: any(
            len(inner_list) > 0 for inner_list in example["short_answers"]
        )
    )

    mini_dataset = filtered_dataset.shuffle(seed=CONFIG.seed).select(
        range(CONFIG.dataset_size)
    )

    return mini_dataset.to_list()


def main():
    dataset_name = CONFIG.dataset_name.lower()
    dataset = load_dataset(
        CONFIG.dataset_name, split=CONFIG.split, trust_remote_code=True
    )
    subset = process_nq(dataset)

    if subset:
        # Construct output file path
        output_name = f"{CONFIG.dataset_name.split('/')[-1]}_mini_{'dev_' if CONFIG.split == 'dev' else ''}{N_SAMPLES}.jsonl"
        output_path = CONFIG.data_dir / output_name

        save_jsonl(subset, output_path)
        print(f"\nSubset successfully created and saved to: {output_path}")


if __name__ == "__main__":
    main()
