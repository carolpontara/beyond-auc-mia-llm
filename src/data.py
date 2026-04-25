from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset


@dataclass
class CandidateSample:
    text: str
    label: int
    source: str
    sample_id: str


def set_seed(seed: int) -> None:
    random.seed(seed)


def _stream_texts(split, text_column: str, limit: int) -> List[str]:
    """Collect up to `limit` non-empty texts from a streaming split."""
    texts: List[str] = []
    for item in split:
        raw = item.get(text_column, "") or ""
        t = raw.strip()
        if t:
            texts.append(t)
        if len(texts) >= limit:
            break
    return texts


def load_real_text_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    text_column: str,
    max_train: int = 0,
    max_eval: int = 0,
) -> Tuple[List[str], List[str]]:
    """Load a HuggingFace text dataset using streaming.

    Streaming avoids downloading the full dataset into memory — only the
    texts actually needed (up to max_train + max_eval) are fetched.  This
    is critical for large datasets such as OpenWebText (~24 GB).

    For datasets without a predefined test split (e.g. OpenWebText) the
    train stream is split 90/10.
    """
    load_kwargs: Dict = {"streaming": True}
    if dataset_config:
        load_kwargs["name"] = dataset_config

    dataset = load_dataset(dataset_name, **load_kwargs)

    want_train = max_train if max_train > 0 else 10_000
    want_eval = max_eval if max_eval > 0 else 1_000

    if "test" in dataset:
        train_texts = _stream_texts(dataset["train"], text_column, want_train)
        test_texts = _stream_texts(dataset["test"], text_column, want_eval)
    else:
        # Datasets like OpenWebText have only a train split — stream exactly
        # want_train + want_eval texts and slice directly.  A proportional
        # 90/10 split would give only (want_train+want_eval)*0.1 for eval,
        # which is less than want_eval whenever want_train >> want_eval.
        all_texts = _stream_texts(dataset["train"], text_column, want_train + want_eval)
        train_texts = all_texts[:want_train]
        test_texts = all_texts[want_train:]

    if max_train > 0:
        train_texts = train_texts[:max_train]
    if max_eval > 0:
        test_texts = test_texts[:max_eval]

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
