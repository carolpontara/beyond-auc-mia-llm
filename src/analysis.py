"""Sample vulnerability analysis module.

Computes per-sample features that help explain *why* certain texts are
more vulnerable to membership inference attacks:
  - text length (number of tokens)
  - token rarity (mean inverse document frequency across corpus)
  - word frequency (mean unigram frequency in training corpus)
  - loss difference (target model loss minus reference model loss)
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def token_length(text: str) -> int:
    """Number of whitespace-delimited tokens."""
    return len(text.split())


def build_idf(corpus: Sequence[str]) -> Dict[str, float]:
    """Build inverse document frequency from a corpus of texts."""
    n_docs = len(corpus)
    doc_freq: Counter = Counter()
    for text in corpus:
        unique_tokens = set(text.lower().split())
        for tok in unique_tokens:
            doc_freq[tok] += 1
    idf: Dict[str, float] = {}
    for tok, df in doc_freq.items():
        idf[tok] = math.log((n_docs + 1) / (df + 1)) + 1
    return idf


def mean_token_rarity(text: str, idf: Dict[str, float]) -> float:
    """Mean IDF of tokens in the text (higher = rarer tokens)."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    default_idf = max(idf.values()) if idf else 1.0
    return float(np.mean([idf.get(tok, default_idf) for tok in tokens]))


def build_unigram_freq(corpus: Sequence[str]) -> Dict[str, float]:
    """Build normalised unigram frequency table from corpus."""
    counter: Counter = Counter()
    total = 0
    for text in corpus:
        toks = text.lower().split()
        counter.update(toks)
        total += len(toks)
    if total == 0:
        return {}
    return {tok: count / total for tok, count in counter.items()}


def mean_word_frequency(text: str, freq: Dict[str, float]) -> float:
    """Mean unigram frequency of tokens (higher = more common words)."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return float(np.mean([freq.get(tok, 0.0) for tok in tokens]))


def compute_sample_features(
    texts: Sequence[str],
    sample_ids: Sequence[str],
    labels: Sequence[int],
    sources: Sequence[str],
    scores: Sequence[float],
    predictions: Sequence[int],
    target_losses: Sequence[float],
    ref_losses: Sequence[float],
    train_corpus: Sequence[str],
    attack_name: str,
) -> pd.DataFrame:
    """Build a DataFrame with per-sample vulnerability features."""
    idf = build_idf(train_corpus)
    freq = build_unigram_freq(train_corpus)

    rows = []
    for i, text in enumerate(texts):
        rows.append({
            "sample_id": sample_ids[i],
            "label": labels[i],
            "source": sources[i],
            "attack": attack_name,
            "score": scores[i],
            "prediction": predictions[i],
            "correct": int(predictions[i] == labels[i]),
            "text_length": token_length(text),
            "token_rarity": mean_token_rarity(text, idf),
            "word_frequency": mean_word_frequency(text, freq),
            "target_loss": target_losses[i] if i < len(target_losses) else float("nan"),
            "ref_loss": ref_losses[i] if i < len(ref_losses) else float("nan"),
            "loss_diff": (
                (target_losses[i] - ref_losses[i])
                if i < len(target_losses) and i < len(ref_losses)
                else float("nan")
            ),
        })
    return pd.DataFrame(rows)
