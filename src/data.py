from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

from datasets import load_dataset


@dataclass
class CandidateSample:
    text: str
    label: int
    source: str
    sample_id: str


def set_seed(seed: int) -> None:
    random.seed(seed)


def load_real_text_dataset(dataset_name: str, dataset_config: str, text_column: str):
    dataset = load_dataset(dataset_name, dataset_config)
    train_texts = [t.strip() for t in dataset["train"][text_column] if t and t.strip()]
    test_texts = [t.strip() for t in dataset["test"][text_column] if t and t.strip()]
    return train_texts, test_texts


def generate_canary(idx: int, seed: int) -> str:
    rng = random.Random(seed + idx)
    token = rng.randint(100000, 999999)
    key = rng.randint(1000, 9999)
    return f"CANARY_SEQUENCE_{idx}_TOKEN_{token}_KEY_{key}"


def insert_canaries(
    train_texts: List[str],
    num_canaries: int,
    repetitions: int,
    seed: int,
) -> Tuple[List[str], List[str]]:
    augmented = list(train_texts)
    canaries = [generate_canary(i, seed) for i in range(num_canaries)]
    insertion_pool: List[str] = []
    for canary in canaries:
        insertion_pool.extend([canary] * repetitions)

    rng = random.Random(seed)
    rng.shuffle(insertion_pool)

    for canary in insertion_pool:
        pos = rng.randint(0, len(augmented))
        augmented.insert(pos, canary)

    return augmented, canaries


def sample_candidates(
    train_texts: List[str],
    test_texts: List[str],
    canaries: List[str],
    member_sample_size: int,
    nonmember_sample_size: int,
    seed: int,
) -> List[CandidateSample]:
    rng = random.Random(seed)

    natural_members = rng.sample(train_texts, k=min(member_sample_size, len(train_texts)))
    natural_nonmembers = rng.sample(test_texts, k=min(nonmember_sample_size, len(test_texts)))

    candidates: List[CandidateSample] = []
    for i, text in enumerate(canaries):
        candidates.append(CandidateSample(text=text, label=1, source="canary", sample_id=f"canary_{i}"))

    for i, text in enumerate(natural_members):
        candidates.append(CandidateSample(text=text, label=1, source="train", sample_id=f"member_{i}"))

    for i, text in enumerate(natural_nonmembers):
        candidates.append(CandidateSample(text=text, label=0, source="test", sample_id=f"nonmember_{i}"))

    rng.shuffle(candidates)
    return candidates


def truncate_list(values: List[str], max_items: int) -> List[str]:
    if max_items <= 0:
        return values
    return values[: max_items]


