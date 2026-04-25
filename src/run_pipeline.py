from __future__ import annotations

import argparse
import copy
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attacks import batch_sequence_loss, compute_attack_scores, sequence_loss
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
    "recall": True,
    "lira": False,
}

# Attacks whose scores are independent of rng_seed (no randomness in scoring).
# We compute them once per experiment and replicate across runs, saving
# (num_attack_runs - 1) full GPU scoring passes.
DETERMINISTIC_ATTACKS: frozenset = frozenset({"loss", "ref_gap", "min_k_pp", "recall", "lira"})


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
# Checkpoint/resume helpers
# ------------------------------------------------------------------

def experiment_exists(exp_dir: str) -> bool:
    """Check if experiment has been completed."""
    required_files = [
        os.path.join(exp_dir, "attack_scores.csv"),
        os.path.join(exp_dir, "metrics_summary.json"),
    ]
    return all(os.path.exists(f) for f in required_files)


def load_existing_experiment(exp_dir: str, model_name: str, ds_label: str, canary_rep: int) -> Optional[Dict[str, Any]]:
    """Load results from an already-completed experiment."""
    try:
        # Load attack scores
        scores_csv = os.path.join(exp_dir, "attack_scores.csv")
        df_scores = pd.read_csv(scores_csv)
        
        # Load metrics
        metrics_json = os.path.join(exp_dir, "metrics_summary.json")
        with open(metrics_json, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        
        # Load analysis if it exists
        analysis_csv = os.path.join(exp_dir, "sample_analysis.csv")
        df_analysis = pd.read_csv(analysis_csv) if os.path.exists(analysis_csv) else pd.DataFrame()
        
        # Reconstruct attack stability info (estimate from metrics)
        stability_dict = {}
        for attack_name in metrics.keys():
            stability_dict[attack_name] = metrics[attack_name].get("stability", {})
        
        return {
            "model": model_name,
            "dataset": ds_label,
            "canary_rep": canary_rep,
            "metrics": metrics,
            "stability": stability_dict,
            "scores_df": df_scores,
            "analysis_df": df_analysis,
        }
    except Exception as e:
        print(f"[WARN] Failed to load existing experiment from {exp_dir}: {e}")
        return None


# ------------------------------------------------------------------
# Parallel attack execution helpers
# ------------------------------------------------------------------

def _run_attack_iteration(
    run_idx: int,
    attack_name: str,
    target_model: Any,
    ref_model: Any,
    tokenizer: Any,
    texts: List[str],
    labels: List[int],
    candidates: List[Any],
    sample_ids: List[str],
    sources: List[str],
    config: Dict[str, Any],
    seed: int,
    smaller_is_member: bool,
    shadow_models: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Run a single attack iteration (one run)."""
    current_seed = seed + run_idx
    print(f"[DEBUG] Starting attack '{attack_name}' run {run_idx + 1}/{config['num_attack_runs']}, seed={current_seed}")
    
    scores = compute_attack_scores(
        attack_name=attack_name,
        target_model=target_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        texts=texts,
        max_length=config["max_length"],
        min_k_ratio=config["min_k_ratio"],
        rng_seed=current_seed,
        neighborhood_variants=config["neighborhood_variants"],
        shadow_models=shadow_models,
    )
    metrics = classification_metrics(labels, scores, smaller_is_member)
    preds = predictions_from_scores(scores, metrics["threshold"], smaller_is_member)
    metrics["coverage_all_members"] = coverage(sample_ids, labels, preds)
    metrics["coverage_canaries"] = coverage_canaries(sample_ids, sources, preds)
    
    rows = []
    for ci, (cand, sc, pr) in enumerate(zip(candidates, scores, preds)):
        rows.append({
            "model_name": None,  # Will be filled later
            "dataset": None,  # Will be filled later
            "canary_rep": None,  # Will be filled later
            "attack": attack_name,
            "run": run_idx,
            "sample_id": cand.sample_id,
            "source": cand.source,
            "label": cand.label,
            "prediction": pr,
            "score": sc,
            "text": cand.text[:200],
        })
    
    return {
        "run_idx": run_idx,
        "scores": scores,
        "metrics": metrics,
        "preds": preds,
        "rows": rows,
    }


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
            # Halve epochs for shadow models: same total compute but doubles
            # the number of shadow models you can afford, improving z-score quality.
            shadow_cfg["num_train_epochs"] = max(1, config["num_train_epochs"] // 2)
            rng = random.Random(shadow_cfg["seed"])
            # Use 50% of train_texts per shadow model so each sample has
            # ~50% probability of being "out" — this is required for the
            # LiRA z-score to have a meaningful signal on regular members.
            # Training on the full set makes target_loss ≈ shadow_loss for
            # all members, collapsing all z-scores to zero.
            shadow_size = max(len(train_texts) // 2, 1)
            shadow_train = rng.sample(train_texts, k=min(shadow_size, len(train_texts)))
            _, shadow_m = train_model(
                config=shadow_cfg,
                train_texts=shadow_train,
                eval_texts=test_texts,
            )
            shadow_m = move_model(shadow_m, device)
            shadow_models.append(shadow_m)

    # 7. Run attacks
    # Deterministic attacks (loss, ref_gap, min_k_pp, recall, lira) produce
    # identical scores regardless of rng_seed, so we compute them once and
    # replicate across runs.  Only min_k varies (uses rng for perturbations).
    all_rows: List[Dict] = []
    attack_summaries: Dict[str, Dict] = {}
    attack_stability: Dict[str, float] = {}
    # Cache run-0 scores per attack; reused in the analysis step.
    first_run_scores: Dict[str, List[float]] = {}

    for attack_name in config["attacks"]:
        smaller = ATTACK_SMALLER_IS_MEMBER.get(attack_name, True)
        print(f"[LOG]   Attack: {attack_name} (smaller_is_member={smaller})")

        run_predictions: List[List[int]] = []
        run_metrics_list: List[Dict[str, float]] = []

        if attack_name in DETERMINISTIC_ATTACKS:
            # Compute once — all runs are identical for deterministic attacks.
            result = _run_attack_iteration(
                run_idx=0,
                attack_name=attack_name,
                target_model=trained_model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                texts=texts,
                labels=labels,
                candidates=candidates,
                sample_ids=sample_ids,
                sources=sources,
                config=config,
                seed=seed,
                smaller_is_member=smaller,
                shadow_models=shadow_models if attack_name == "lira" else None,
            )
            first_run_scores[attack_name] = result["scores"]
            for run_idx in range(config["num_attack_runs"]):
                run_predictions.append(result["preds"])
                run_metrics_list.append(result["metrics"])
                for row in result["rows"]:
                    all_rows.append({
                        **row,
                        "run": run_idx,
                        "model": model_name,
                        "dataset": ds_label,
                        "canary_rep": canary_rep,
                    })
            print(
                f"[LOG]     (×{config['num_attack_runs']} runs, deterministic) "
                f"AUC={result['metrics']['auc']:.3f} F1={result['metrics']['f1']:.3f} "
                f"F1_best={result['metrics']['f1_f1']:.3f} "
                f"canary_cov={result['metrics']['coverage_canaries']:.3f}"
            )
        else:
            # Non-deterministic (min_k): run sequentially.
            # Batched GPU inference already maximises device utilisation.
            for run_idx in range(config["num_attack_runs"]):
                result = _run_attack_iteration(
                    run_idx=run_idx,
                    attack_name=attack_name,
                    target_model=trained_model,
                    ref_model=ref_model,
                    tokenizer=tokenizer,
                    texts=texts,
                    labels=labels,
                    candidates=candidates,
                    sample_ids=sample_ids,
                    sources=sources,
                    config=config,
                    seed=seed,
                    smaller_is_member=smaller,
                    shadow_models=None,
                )
                if run_idx == 0:
                    first_run_scores[attack_name] = result["scores"]
                run_predictions.append(result["preds"])
                run_metrics_list.append(result["metrics"])
                for row in result["rows"]:
                    all_rows.append({
                        **row,
                        "model": model_name,
                        "dataset": ds_label,
                        "canary_rep": canary_rep,
                    })
                print(
                    f"[LOG]     run {run_idx + 1}: "
                    f"AUC={result['metrics']['auc']:.3f} F1={result['metrics']['f1']:.3f} "
                    f"F1_best={result['metrics']['f1_f1']:.3f} "
                    f"canary_cov={result['metrics']['coverage_canaries']:.3f}"
                )

        stab = stability(run_predictions, sample_ids)
        attack_stability[attack_name] = stab
        agg = aggregate_run_metrics(run_metrics_list)
        attack_summaries[attack_name] = agg
        attack_summaries[attack_name]["stability"] = stab

    # 8. Sample vulnerability analysis
    # Reuse cached run-0 scores — no GPU re-computation needed.
    # compute_sample_features is CPU-only so ThreadPoolExecutor gives real parallelism.
    analysis_frames: List[pd.DataFrame] = []
    target_losses = batch_sequence_loss(trained_model, tokenizer, texts, config["max_length"])
    ref_losses = batch_sequence_loss(ref_model, tokenizer, texts, config["max_length"])

    def _compute_analysis_for_attack(attack_name: str) -> pd.DataFrame:
        smaller = ATTACK_SMALLER_IS_MEMBER.get(attack_name, True)
        scores = first_run_scores[attack_name]
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
        return df_feat

    with ThreadPoolExecutor(max_workers=min(len(config["attacks"]), os.cpu_count() or 4)) as executor:
        futures = {executor.submit(_compute_analysis_for_attack, atk): atk for atk in config["attacks"]}
        for future in as_completed(futures):
            try:
                df = future.result()
                analysis_frames.append(df)
            except Exception as e:
                atk = futures[future]
                print(f"[ERROR] Analysis for attack {atk} failed: {e}")
                raise

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
    skipped = 0
    for model_name in models:
        for ds_cfg in datasets:
            for crep in canary_reps:
                idx += 1
                ds_label = dataset_key(ds_cfg)
                exp_dir = os.path.join(output_base, f"{model_name.replace('/', '_')}_{ds_label}_r{crep}")
                
                print(f"\n[LOG] ====== Experiment {idx}/{total} ======")
                print(f"[LOG]   model={model_name}  dataset={ds_label}  r={crep}")
                
                # Check if experiment already exists
                if experiment_exists(exp_dir):
                    print(f"[LOG]   ✓ Experiment already completed, loading results...")
                    result = load_existing_experiment(exp_dir, model_name, ds_label, crep)
                    if result:
                        skipped += 1
                        all_results.append(result)
                        all_scores_frames.append(result["scores_df"])
                        if not result["analysis_df"].empty:
                            all_analysis_frames.append(result["analysis_df"])
                        print(f"[LOG]   Results loaded successfully")
                        continue
                    else:
                        print(f"[LOG]   ! Failed to load results, re-running experiment")
                
                # Run new experiment
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
    
    print(f"\n[LOG] Experiments completed: {idx - skipped} new, {skipped} loaded from checkpoint")

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
