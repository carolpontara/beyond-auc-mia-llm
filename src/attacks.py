from __future__ import annotations

import math
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F


def _device_of(model) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def sequence_loss(model, tokenizer, text: str, max_length: int) -> float:
    device = _device_of(model)
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    outputs = model(**encoded, labels=encoded["input_ids"])
    return float(outputs.loss.detach().cpu().item())


@torch.no_grad()
def token_logprobs(model, tokenizer, text: str, max_length: int) -> List[float]:
    device = _device_of(model)
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    gathered = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return gathered.squeeze(0).detach().cpu().tolist()


# ---------------------------------------------------------------------------
# Original attacks
# ---------------------------------------------------------------------------

def min_k_score(model, tokenizer, text: str, max_length: int, ratio: float) -> float:
    lps = token_logprobs(model, tokenizer, text, max_length=max_length)
    if not lps:
        return float("inf")
    k = max(1, math.ceil(len(lps) * ratio))
    smallest = sorted(lps)[:k]
    return -float(sum(smallest) / len(smallest))


def reference_gap_score(target_model, ref_model, tokenizer, text: str, max_length: int) -> float:
    target_loss = sequence_loss(target_model, tokenizer, text, max_length)
    ref_loss = sequence_loss(ref_model, tokenizer, text, max_length)
    return target_loss - ref_loss


def perturb_text(text: str, rng: random.Random) -> str:
    tokens = text.split()
    if len(tokens) < 4:
        return text
    tokens = tokens[:]
    i = rng.randint(0, len(tokens) - 2)
    tokens[i], tokens[i + 1] = tokens[i + 1], tokens[i]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# New SOTA attacks
# ---------------------------------------------------------------------------

def min_k_pp_score(model, tokenizer, text: str, max_length: int, ratio: float) -> float:
    """Min-K%++ (Zhang et al. 2025): z-score normalised token log-probs."""
    lps = token_logprobs(model, tokenizer, text, max_length=max_length)
    if not lps:
        return float("inf")
    arr = np.array(lps, dtype=np.float64)
    mu = arr.mean()
    sigma = arr.std()
    if sigma < 1e-12:
        z_scores = arr - mu
    else:
        z_scores = (arr - mu) / sigma
    k = max(1, math.ceil(len(z_scores) * ratio))
    smallest = np.sort(z_scores)[:k]
    return -float(smallest.mean())


def recall_score(
    target_model,
    ref_model,
    tokenizer,
    text: str,
    max_length: int,
) -> float:
    """ReCaLL (Xie et al. 2024): relative conditional log-likelihood.

    Computes the difference of mean token log-prob under the target model
    vs. the reference model. A higher value indicates a member.
    """
    target_lps = token_logprobs(target_model, tokenizer, text, max_length)
    ref_lps = token_logprobs(ref_model, tokenizer, text, max_length)
    if not target_lps or not ref_lps:
        return 0.0
    min_len = min(len(target_lps), len(ref_lps))
    target_mean = float(np.mean(target_lps[:min_len]))
    ref_mean = float(np.mean(ref_lps[:min_len]))
    return target_mean - ref_mean


def lira_score(
    target_model,
    shadow_models: List,
    tokenizer,
    text: str,
    max_length: int,
) -> float:
    """LiRA (Carlini et al. 2022): simplified likelihood ratio attack.

    Computes target loss and compares it to the distribution of losses
    from shadow models. Returns z-score (lower loss = more likely a member,
    so we negate to align with 'smaller is member').
    """
    target_loss = sequence_loss(target_model, tokenizer, text, max_length)
    if not shadow_models:
        return -target_loss
    shadow_losses = [
        sequence_loss(sm, tokenizer, text, max_length) for sm in shadow_models
    ]
    mu = float(np.mean(shadow_losses))
    sigma = float(np.std(shadow_losses))
    if sigma < 1e-12:
        return -(target_loss - mu)
    return -(target_loss - mu) / sigma


# ---------------------------------------------------------------------------
# Unified scoring function
# ---------------------------------------------------------------------------

def compute_attack_scores(
    attack_name: str,
    target_model,
    ref_model,
    tokenizer,
    texts: List[str],
    max_length: int,
    min_k_ratio: float,
    rng_seed: int,
    neighborhood_variants: int,
    shadow_models: Optional[List] = None,
) -> List[float]:
    """Compute scores for a single attack on a list of texts."""
    rng = random.Random(rng_seed)
    scores: List[float] = []
    for text in texts:
        if attack_name == "loss":
            scores.append(sequence_loss(target_model, tokenizer, text, max_length))
        elif attack_name == "ref_gap":
            score = reference_gap_score(target_model, ref_model, tokenizer, text, max_length)
            scores.append(score)
        elif attack_name == "min_k":
            perturbed = [perturb_text(text, rng) for _ in range(max(1, neighborhood_variants - 1))]
            candidates = [text] + perturbed
            variant_scores = [min_k_score(target_model, tokenizer, x, max_length, min_k_ratio) for x in candidates]
            scores.append(sum(variant_scores) / len(variant_scores))
        elif attack_name == "min_k_pp":
            scores.append(min_k_pp_score(target_model, tokenizer, text, max_length, min_k_ratio))
        elif attack_name == "recall":
            scores.append(recall_score(target_model, ref_model, tokenizer, text, max_length))
        elif attack_name == "lira":
            scores.append(lira_score(target_model, shadow_models or [], tokenizer, text, max_length))
        else:
            raise ValueError(f"Unsupported attack: {attack_name}")
    return scores


# Backward-compatible alias
def repeated_scores_for_stability(
    attack_name: str,
    target_model,
    ref_model,
    tokenizer,
    texts: List[str],
    max_length: int,
    min_k_ratio: float,
    rng_seed: int,
    neighborhood_variants: int,
    shadow_models: Optional[List] = None,
) -> List[float]:
    return compute_attack_scores(
        attack_name=attack_name,
        target_model=target_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=max_length,
        min_k_ratio=min_k_ratio,
        rng_seed=rng_seed,
        neighborhood_variants=neighborhood_variants,
        shadow_models=shadow_models,
    )


