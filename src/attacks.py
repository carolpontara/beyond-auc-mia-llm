from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List

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
) -> List[float]:
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
        else:
            raise ValueError(f"Unsupported attack: {attack_name}")
    return scores


