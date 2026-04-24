from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attacks import compute_attack_scores, sequence_loss
from analysis import compute_sample_features
from data import (
    CandidateSample,
    insert_canaries,
    load_real_text_dataset,
    sample_candidates,
    set_seed,
    truncate_list,
)
from metrics import (
    aggregate_run_metrics,
    classification_metrics,
    coverage,
    coverage_canaries,
    predictions_from_scores,
    stability,
)
from plots import generate_all_plots
from train import train_model


ATTACK_SMALLER_IS_MEMBER: Dict[str, bool] = {
    "loss": True,
    "ref_gap": True,
    "min_k": True,
    "min_k_pp": True,
    "recall": False,
    "lira": False,
}


def read_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_model(model: Any, device: torch.device) -> Any:
    model.to(device)
    model.eval()
    return model


def load_model_and_tokenizer(model_name: str, device: torch.device):
    """Load a pre-trained causal LM and its tokenizer."""
    print(f"[LOG]   Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = move_model(model, device)
    return tokenizer, model


def dataset_key(ds_cfg: Dict[str, Any]) -> str:
    """Create a short human-readable key for a dataset config."""
    cfg = ds_cfg.get("dataset_config") or ds_cfg["dataset_name"]
    return str(cfg).replace("/", "_")


# ------------------------------------------------------------------
# Single model+dataset experiment
# ------------------------------------------------------------------

def run_single_experiment(
    model_name: str,
    ds_cfg: Dict[str, Any],
    config: Dict[str, Any],
    device: torch.device,
    output_base: str,
    canary_rep: int,
) -> Dict[str, Any]:
    """Run the full pipeline for one model, one dataset, one canary repetition."""
    ds_label = dataset_key(ds_cfg)
    exp_dir = os.path.join(output_base, f"{model_name.replace('/', '_')}_{ds_label}_r{canary_rep}")
    ensure_dir(exp_dir)

    # Build a single-model config for training
    single_cfg = copy.deepcopy(config)
    single_cfg["model_name"] = model_name
    single_cfg["dataset_name"] = ds_cfg["dataset_name"]
    single_cfg["dataset_config"] = ds_cfg.get("dataset_config")
    single_cfg["text_column"] = ds_cfg.get("text_column", "text")
    single_cfg["canary_repetitions"] = canary_rep
    single_cfg["output_dir"] = exp_dir

    seed = config["seed"]
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. Load dataset
    print(f"[LOG] Loading dataset {ds_label}")
    train_texts, test_texts = load_real_text_dataset(
        dataset_name=single_cfg["dataset_name"],
        dataset_config=single_cfg["dataset_config"],
        text_column=single_cfg["text_column"],
        max_train=config["max_train_texts"],
        max_eval=config["max_eval_texts"],
    )
    train_texts = truncate_list(train_texts, config["max_train_texts"])
    test_texts = truncate_list(test_texts, config["max_eval_texts"])
    print(f"[LOG]   train={len(train_texts)}, test={len(test_texts)}")

    # 2. Canaries
    augmented_train, canaries = insert_canaries(
        train_texts=train_texts,
        num_canaries=config["num_canaries"],
        repetitions=canary_rep,
        seed=seed,
    )

    # 3. Candidates
    candidates = sample_candidates(
        train_texts=augmented_train,
        test_texts=test_texts,
        canaries=canaries,
        member_sample_size=config["member_sample_size"],
        nonmember_sample_size=config["nonmember_sample_size"],
        seed=seed,
    )

    labels = [c.label for c in candidates]
    sample_ids = [c.sample_id for c in candidates]
    sources = [c.source for c in candidates]
    texts = [c.text for c in candidates]

    # 4. Train target model
    print(f"[LOG] Training target model {model_name} on {ds_label} (r={canary_rep})")
    tokenizer, trained_model = train_model(
        config=single_cfg,
        train_texts=augmented_train,
        eval_texts=test_texts,
    )
    trained_model = move_model(trained_model, device)

    # 5. Load reference model (pre-trained, no fine-tuning)
    _, ref_model = load_model_and_tokenizer(model_name, device)

    # 6. Shadow models for LiRA
    shadow_models: List[Any] = []
    num_shadow = config.get("num_shadow_models", 0)
    if "lira" in config["attacks"] and num_shadow > 0:
        print(f"[LOG]   Training {num_shadow} shadow models for LiRA")
        for si in range(num_shadow):
            shadow_cfg = copy.deepcopy(single_cfg)
            shadow_cfg["seed"] = seed + 1000 + si
            shadow_cfg["output_dir"] = os.path.join(exp_dir, f"shadow_{si}")
            ensure_dir(shadow_cfg["output_dir"])
            rng = random.Random(shadow_cfg["seed"])
            shadow_train = rng.sample(train_texts, k=min(len(train_texts), config["max_train_texts"]))
            _, shadow_m = train_model(
                config=shadow_cfg,
                train_texts=shadow_train,
                eval_texts=test_texts,
            )
            shadow_m = move_model(shadow_m, device)
            shadow_models.append(shadow_m)

    # 7. Run attacks
    all_rows: List[Dict] = []
    attack_summaries: Dict[str, Dict] = {}
    attack_stability: Dict[str, float] = {}

    for attack_name in config["attacks"]:
        smaller = ATTACK_SMALLER_IS_MEMBER.get(attack_name, True)
        print(f"[LOG]   Attack: {attack_name} (smaller_is_member={smaller})")
        run_predictions: List[List[int]] = []
        run_metrics_list: List[Dict[str, float]] = []

        for run_idx in range(config["num_attack_runs"]):
            scores = compute_attack_scores(
                attack_name=attack_name,
                target_model=trained_model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                texts=texts,
                max_length=config["max_length"],
                min_k_ratio=config["min_k_ratio"],
                rng_seed=seed + run_idx,
                neighborhood_variants=config["neighborhood_variants"],
                shadow_models=shadow_models if attack_name == "lira" else None,
            )
            metrics = classification_metrics(labels, scores, smaller)
            preds = predictions_from_scores(scores, metrics["threshold"], smaller)
            run_predictions.append(preds)
            metrics["coverage_all_members"] = coverage(sample_ids, labels, preds)
            metrics["coverage_canaries"] = coverage_canaries(sample_ids, sources, preds)
            run_metrics_list.append(metrics)

            for ci, (cand, sc, pr) in enumerate(zip(candidates, scores, preds)):
                all_rows.append({
                    "model": model_name,
                    "dataset": ds_label,
                    "canary_rep": canary_rep,
                    "attack": attack_name,
                    "run": run_idx,
                    "sample_id": cand.sample_id,
                    "source": cand.source,
                    "label": cand.label,
                    "prediction": pr,
                    "score": sc,
                    "text": cand.text[:200],
                })

            print(
                f"[LOG]     run {run_idx + 1}: "
                f"AUC={metrics['auc']:.3f} F1={metrics['f1']:.3f} "
                f"F1_best={metrics['f1_f1']:.3f} "
                f"canary_cov={metrics['coverage_canaries']:.3f}"
            )

        stab = stability(run_predictions, sample_ids)
        attack_stability[attack_name] = stab
        agg = aggregate_run_metrics(run_metrics_list)
        attack_summaries[attack_name] = agg
        attack_summaries[attack_name]["stability"] = stab

    # 8. Sample vulnerability analysis (first run, first attack with loss)
    analysis_frames: List[pd.DataFrame] = []
    target_losses = [
        sequence_loss(trained_model, tokenizer, t, config["max_length"]) for t in texts
    ]
    ref_losses = [
        sequence_loss(ref_model, tokenizer, t, config["max_length"]) for t in texts
    ]
    for attack_name in config["attacks"]:
        smaller = ATTACK_SMALLER_IS_MEMBER.get(attack_name, True)
        scores = compute_attack_scores(
            attack_name=attack_name,
            target_model=trained_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            texts=texts,
            max_length=config["max_length"],
            min_k_ratio=config["min_k_ratio"],
            rng_seed=seed,
            neighborhood_variants=config["neighborhood_variants"],
            shadow_models=shadow_models if attack_name == "lira" else None,
        )
        metrics = classification_metrics(labels, scores, smaller)
        preds = predictions_from_scores(scores, metrics["threshold"], smaller)
        df_feat = compute_sample_features(
            texts=texts,
            sample_ids=sample_ids,
            labels=labels,
            sources=sources,
            scores=scores,
            predictions=preds,
            target_losses=target_losses,
            ref_losses=ref_losses,
            train_corpus=train_texts[:500],
            attack_name=attack_name,
        )
        df_feat["model"] = model_name
        df_feat["dataset"] = ds_label
        df_feat["canary_rep"] = canary_rep
        analysis_frames.append(df_feat)

    # 9. Clean up GPU memory
    del trained_model, ref_model
    for sm in shadow_models:
        del sm
    shadow_models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 10. Save per-experiment results
    df_scores = pd.DataFrame(all_rows)
    df_scores.to_csv(os.path.join(exp_dir, "attack_scores.csv"), index=False)

    df_analysis = pd.concat(analysis_frames, ignore_index=True) if analysis_frames else pd.DataFrame()
    if not df_analysis.empty:
        df_analysis.to_csv(os.path.join(exp_dir, "sample_analysis.csv"), index=False)

    save_json(os.path.join(exp_dir, "metrics_summary.json"), attack_summaries)
    save_json(os.path.join(exp_dir, "run_config.json"), single_cfg)

    return {
        "model": model_name,
        "dataset": ds_label,
        "canary_rep": canary_rep,
        "metrics": attack_summaries,
        "stability": attack_stability,
        "scores_df": df_scores,
        "analysis_df": df_analysis,
    }


# ------------------------------------------------------------------
# Main orchestrator
# ------------------------------------------------------------------

def main(config_path: str) -> None:
    print(f"[LOG] Loading config from: {config_path}")
    config = read_config(config_path)
    output_base = config["output_dir"]
    ensure_dir(output_base)
    device = choose_device()
    print(f"[LOG] Device: {device}")

    models = config.get("models", [config.get("model_name", "distilgpt2")])
    datasets = config.get("datasets", [{
        "dataset_name": config.get("dataset_name", "wikitext"),
        "dataset_config": config.get("dataset_config", "wikitext-2-raw-v1"),
        "text_column": config.get("text_column", "text"),
    }])
    canary_reps = config.get("canary_repetitions", [4])
    if isinstance(canary_reps, int):
        canary_reps = [canary_reps]

    print(f"[LOG] Models: {models}")
    print(f"[LOG] Datasets: {[dataset_key(d) for d in datasets]}")
    print(f"[LOG] Canary repetitions: {canary_reps}")
    print(f"[LOG] Attacks: {config['attacks']}")
    print(f"[LOG] Runs per attack: {config['num_attack_runs']}")

    # Collect all results
    all_results: List[Dict[str, Any]] = []
    all_scores_frames: List[pd.DataFrame] = []
    all_analysis_frames: List[pd.DataFrame] = []

    total = len(models) * len(datasets) * len(canary_reps)
    idx = 0
    for model_name in models:
        for ds_cfg in datasets:
            for crep in canary_reps:
                idx += 1
                print(f"\n[LOG] ====== Experiment {idx}/{total} ======")
                print(f"[LOG]   model={model_name}  dataset={dataset_key(ds_cfg)}  r={crep}")
                result = run_single_experiment(
                    model_name=model_name,
                    ds_cfg=ds_cfg,
                    config=config,
                    device=device,
                    output_base=output_base,
                    canary_rep=crep,
                )
                all_results.append(result)
                all_scores_frames.append(result["scores_df"])
                if not result["analysis_df"].empty:
                    all_analysis_frames.append(result["analysis_df"])

    # ------------------------------------------------------------------
    # Aggregate into full_summary.json
    # ------------------------------------------------------------------
    print("\n[LOG] Building full summary...")
    full_summary: Dict[str, Any] = {"by_dataset": {}, "canary_by_repetition": {}}

    for res in all_results:
        ds = res["dataset"]
        mdl = res["model"]
        crep = res["canary_rep"]

        if ds not in full_summary["by_dataset"]:
            full_summary["by_dataset"][ds] = {"by_model": {}}
        if mdl not in full_summary["by_dataset"][ds]["by_model"]:
            full_summary["by_dataset"][ds]["by_model"][mdl] = {
                "metrics": {},
                "stability": {},
            }

        dest = full_summary["by_dataset"][ds]["by_model"][mdl]
        for atk, atk_data in res["metrics"].items():
            dest["metrics"][atk] = atk_data
        dest["stability"] = res["stability"]

        # Canary by repetition
        if mdl not in full_summary["canary_by_repetition"]:
            full_summary["canary_by_repetition"][mdl] = {}
        canary_cov_by_atk: Dict[str, float] = {}
        for atk, atk_data in res["metrics"].items():
            cc = atk_data.get("coverage_canaries", {})
            if isinstance(cc, dict):
                canary_cov_by_atk[atk] = cc.get("mean", 0)
            else:
                canary_cov_by_atk[atk] = float(cc)
        full_summary["canary_by_repetition"][mdl][str(crep)] = canary_cov_by_atk

    save_json(os.path.join(output_base, "full_summary.json"), full_summary)

    # Save combined CSVs
    if all_scores_frames:
        combined_scores = pd.concat(all_scores_frames, ignore_index=True)
        combined_scores.to_csv(os.path.join(output_base, "attack_scores.csv"), index=False)

    if all_analysis_frames:
        combined_analysis = pd.concat(all_analysis_frames, ignore_index=True)
        combined_analysis.to_csv(os.path.join(output_base, "sample_analysis.csv"), index=False)

    save_json(os.path.join(output_base, "run_config_snapshot.json"), config)

    # ------------------------------------------------------------------
    # Generate plots
    # ------------------------------------------------------------------
    print("\n[LOG] Generating plots...")
    try:
        generated_plots = generate_all_plots(output_base)
        print(f"[LOG] Generated {len(generated_plots)} plots")
    except Exception as exc:
        print(f"[WARN] Plot generation failed: {exc}")

    print("\n[LOG] ========== PIPELINE COMPLETED ==========")
    print(json.dumps(full_summary, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced MIA pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file.")
    args = parser.parse_args()
    main(args.config)
