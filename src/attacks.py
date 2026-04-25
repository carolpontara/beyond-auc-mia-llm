from __future__ import annotations

import contextlib
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
    try:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        if encoded["input_ids"].shape[1] < 2:
            return 0.0
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(**encoded, labels=encoded["input_ids"])
        loss = float(outputs.loss.detach().cpu().item())
        if math.isnan(loss) or math.isinf(loss):
            return 0.0
        return loss
    except Exception:
        return 0.0


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
# Batched inference utilities (GPU-efficient)
# ---------------------------------------------------------------------------

@torch.no_grad()
def batch_sequence_loss(
    model,
    tokenizer,
    texts: List[str],
    max_length: int,
    batch_size: int = 8,
) -> List[float]:
    """Per-sample cross-entropy loss for a list of texts, computed in batches.

    Equivalent to [sequence_loss(model, tokenizer, t, max_length) for t in texts]
    but issues a single forward pass per mini-batch, giving significant GPU
    utilisation improvements over per-sample calls.  Uses fp16 autocast on CUDA.
    """
    device = _device_of(model)
    results: List[float] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            enc = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
                return_attention_mask=True,
            )
            input_ids = enc["input_ids"].to(device)           # [B, T]
            attention_mask = enc["attention_mask"].to(device)  # [B, T]
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()                      # [B, T, V]
            shift_logits = logits[:, :-1, :].contiguous()        # [B, T-1, V]
            shift_labels = input_ids[:, 1:].contiguous()         # [B, T-1]
            shift_mask = attention_mask[:, 1:].float()           # [B, T-1]
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            per_token_loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            ).reshape(shift_labels.size()) * shift_mask
            token_counts = shift_mask.sum(-1).clamp(min=1)
            per_sample = (per_token_loss.sum(-1) / token_counts).tolist()
            for v in per_sample:
                results.append(0.0 if math.isnan(v) or math.isinf(v) else v)
        except Exception:
            for text in batch:
                results.append(sequence_loss(model, tokenizer, text, max_length))
    return results


@torch.no_grad()
def batch_token_logprobs(
    model,
    tokenizer,
    texts: List[str],
    max_length: int,
    batch_size: int = 8,
) -> List[List[float]]:
    """Per-token log-probs for a list of texts, computed in batches.

    Equivalent to [token_logprobs(model, tokenizer, t, max_length) for t in texts]
    but uses a single forward pass per mini-batch.  Padding tokens are stripped
    from each output list so lengths match the real token counts.
    """
    device = _device_of(model)
    results: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            enc = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
                return_attention_mask=True,
            )
            input_ids = enc["input_ids"].to(device)           # [B, T]
            attention_mask = enc["attention_mask"].to(device)  # [B, T]
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()         # [B, T, V] — fp32 for stability
            shift_logits = logits[:, :-1, :]        # [B, T-1, V]
            shift_labels = input_ids[:, 1:]         # [B, T-1]
            shift_mask = attention_mask[:, 1:].bool()  # [B, T-1]
            log_probs = F.log_softmax(shift_logits, dim=-1)
            gathered = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
            gathered_cpu = gathered.detach().cpu()
            mask_cpu = shift_mask.detach().cpu()
            for b in range(len(batch)):
                # mask_cpu selects only non-padding positions (works for any padding side)
                results.append(gathered_cpu[b][mask_cpu[b]].tolist())
        except Exception:
            for text in batch:
                try:
                    results.append(token_logprobs(model, tokenizer, text, max_length))
                except Exception:
                    results.append([])
    return results


# ---------------------------------------------------------------------------
# Original attacks
# ---------------------------------------------------------------------------

def min_k_score(model, tokenizer, text: str, max_length: int, ratio: float) -> float:
    try:
        lps = token_logprobs(model, tokenizer, text, max_length=max_length)
        if not lps:
            return 0.0
        k = max(1, math.ceil(len(lps) * ratio))
        smallest = sorted(lps)[:k]
        score = -float(sum(smallest) / len(smallest))
        if math.isnan(score) or math.isinf(score):
            return 0.0
        return score
    except Exception:
        return 0.0


def reference_gap_score(target_model, ref_model, tokenizer, text: str, max_length: int) -> float:
    try:
        target_loss = sequence_loss(target_model, tokenizer, text, max_length)
        ref_loss = sequence_loss(ref_model, tokenizer, text, max_length)
        score = target_loss - ref_loss
        if math.isnan(score) or math.isinf(score):
            return 0.0
        return score
    except Exception:
        return 0.0


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
    try:
        lps = token_logprobs(model, tokenizer, text, max_length=max_length)
        if not lps:
            return 0.0
        arr = np.array(lps, dtype=np.float64)
        mu = arr.mean()
        sigma = arr.std()
        if sigma < 1e-12:
            z_scores = arr - mu
        else:
            z_scores = (arr - mu) / sigma
        k = max(1, math.ceil(len(z_scores) * ratio))
        smallest = np.sort(z_scores)[:k]
        score = float(smallest.mean())
        if math.isnan(score) or math.isinf(score):
            return 0.0
        return score
    except Exception:
        return 0.0


def recall_score(
    target_model,
    ref_model,
    tokenizer,
    text: str,
    max_length: int,
) -> float:
    """ReCaLL (Xie et al. 2024): relative conditional log-likelihood.

    Computes the ratio of mean token log-prob under the target model
    vs. the reference model. Using ratio instead of difference normalises
    for per-text difficulty: a hard text with high ref_loss would create
    spurious signal under the difference formulation.

    Both means are negative; a member's target_mean is closer to 0
    (higher log-probs) than ref_mean, so the ratio < 1.
    A smaller ratio indicates a stronger member signal.
    """
    try:
        target_lps = token_logprobs(target_model, tokenizer, text, max_length)
        ref_lps = token_logprobs(ref_model, tokenizer, text, max_length)
        if not target_lps or not ref_lps:
            return 0.0
        min_len = min(len(target_lps), len(ref_lps))
        target_mean = float(np.mean(target_lps[:min_len]))
        ref_mean = float(np.mean(ref_lps[:min_len]))
        if abs(ref_mean) < 1e-12:
            return 0.0
        score = target_mean / ref_mean
        if math.isnan(score) or math.isinf(score):
            return 0.0
        return score
    except Exception:
        return 0.0


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
    try:
        target_loss = sequence_loss(target_model, tokenizer, text, max_length)
        # LiRA requires shadow models to compute meaningful z-score
        if not shadow_models or len(shadow_models) < 1:
            # Fallback: insufficient shadow models, return 0 to avoid spurious signal
            return 0.0
        
        shadow_losses = [
            sequence_loss(sm, tokenizer, text, max_length) for sm in shadow_models
        ]
        mu = float(np.mean(shadow_losses))
        # Bessel's correction (ddof=1): unbiased std estimate.
        # np.std default (ddof=0) underestimates std by sqrt(N/(N-1)),
        # which is sqrt(2) with N=2, inflating z-scores significantly.
        ddof = 1 if len(shadow_losses) > 1 else 0
        sigma = float(np.std(shadow_losses, ddof=ddof))
        if sigma < 1e-12:
            score = -(target_loss - mu)
        else:
            score = -(target_loss - mu) / sigma
        
        if math.isnan(score) or math.isinf(score):
            return 0.0
        return score
    except Exception:
        return 0.0


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
    batch_size: int = 8,
) -> List[float]:
    """Compute MIA scores for all texts using batched GPU inference.

    All forward passes are batched (batch_size texts per GPU call), giving a
    significant speedup over per-sample calls.  fp16 autocast is applied on
    CUDA for further speed and memory savings.
    """
    n = len(texts)

    if attack_name == "loss":
        return batch_sequence_loss(target_model, tokenizer, texts, max_length, batch_size)

    elif attack_name == "ref_gap":
        t_losses = batch_sequence_loss(target_model, tokenizer, texts, max_length, batch_size)
        r_losses = batch_sequence_loss(ref_model, tokenizer, texts, max_length, batch_size)
        return [
            0.0 if math.isnan(t - r) or math.isinf(t - r) else (t - r)
            for t, r in zip(t_losses, r_losses)
        ]

    elif attack_name == "min_k":
        rng = random.Random(rng_seed)
        n_variants = max(1, neighborhood_variants)
        all_variants: List[str] = []
        counts: List[int] = []
        for text in texts:
            variants = [text] + [perturb_text(text, rng) for _ in range(n_variants - 1)]
            all_variants.extend(variants)
            counts.append(len(variants))
        all_lps = batch_token_logprobs(target_model, tokenizer, all_variants, max_length, batch_size)
        scores: List[float] = []
        idx = 0
        for count in counts:
            variant_lps = all_lps[idx : idx + count]
            idx += count
            vscore_list = []
            for lps in variant_lps:
                if not lps:
                    vscore_list.append(0.0)
                    continue
                k = max(1, math.ceil(len(lps) * min_k_ratio))
                s = -float(sum(sorted(lps)[:k]) / k)
                vscore_list.append(0.0 if math.isnan(s) or math.isinf(s) else s)
            scores.append(sum(vscore_list) / len(vscore_list) if vscore_list else 0.0)
        return scores

    elif attack_name == "min_k_pp":
        all_lps = batch_token_logprobs(target_model, tokenizer, texts, max_length, batch_size)
        scores = []
        for lps in all_lps:
            if not lps:
                scores.append(0.0)
                continue
            arr = np.array(lps, dtype=np.float64)
            mu, sigma = arr.mean(), arr.std()
            z = (arr - mu) / sigma if sigma >= 1e-12 else arr - mu
            k = max(1, math.ceil(len(z) * min_k_ratio))
            s = float(np.sort(z)[:k].mean())
            scores.append(0.0 if math.isnan(s) or math.isinf(s) else s)
        return scores

    elif attack_name == "recall":
        t_lps = batch_token_logprobs(target_model, tokenizer, texts, max_length, batch_size)
        r_lps = batch_token_logprobs(ref_model, tokenizer, texts, max_length, batch_size)
        scores = []
        for tlp, rlp in zip(t_lps, r_lps):
            if not tlp or not rlp:
                scores.append(0.0)
                continue
            min_len = min(len(tlp), len(rlp))
            t_mean = float(np.mean(tlp[:min_len]))
            r_mean = float(np.mean(rlp[:min_len]))
            if abs(r_mean) < 1e-12:
                scores.append(0.0)
                continue
            s = t_mean / r_mean
            scores.append(0.0 if math.isnan(s) or math.isinf(s) else s)
        return scores

    elif attack_name == "lira":
        sms = shadow_models or []
        if not sms:
            return [0.0] * n
        t_losses = batch_sequence_loss(target_model, tokenizer, texts, max_length, batch_size)
        shadow_loss_matrix = [
            batch_sequence_loss(sm, tokenizer, texts, max_length, batch_size)
            for sm in sms
        ]
        scores = []
        for i in range(n):
            tl = t_losses[i]
            sl = [shadow_loss_matrix[j][i] for j in range(len(sms))]
            mu = float(np.mean(sl))
            ddof = 1 if len(sl) > 1 else 0
            sigma = float(np.std(sl, ddof=ddof))
            s = -(tl - mu) if sigma < 1e-12 else -(tl - mu) / sigma
            scores.append(0.0 if math.isnan(s) or math.isinf(s) else s)
        return scores

    else:
        raise ValueError(f"Unsupported attack: {attack_name}")


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


