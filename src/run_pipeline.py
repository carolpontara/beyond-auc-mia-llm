from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from attacks import repeated_scores_for_stability
from data import CandidateSample, insert_canaries, load_real_text_dataset, sample_candidates, set_seed, truncate_list
from metrics import classification_metrics, coverage, coverage_canaries, predictions_from_scores, stability
from train import train_model


ATTACK_SMALLER_IS_MEMBER = {
    "loss": True,
    "ref_gap": True,
    "min_k": True,
}


def read_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def ensure_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_model_to_device(model, device):
    model.to(device)
    model.eval()
    return model


def main(config_path: str) -> None:
    print(f"[LOG] Loading config from: {config_path}")
    config = read_config(config_path)
    print(f"[LOG] Config loaded successfully")
    
    ensure_output_dir(config["output_dir"])
    print(f"[LOG] Output directory created: {config['output_dir']}")
    
    set_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    print(f"[LOG] Random seeds set to: {config['seed']}")

    print(f"[LOG] Loading dataset: {config['dataset_name']} ({config['dataset_config']})")
    train_texts, test_texts = load_real_text_dataset(
        dataset_name=config["dataset_name"],
        dataset_config=config["dataset_config"],
        text_column=config["text_column"],
    )
    print(f"[LOG] Dataset loaded - train: {len(train_texts)} samples, test: {len(test_texts)} samples")
    
    train_texts = truncate_list(train_texts, config["max_train_texts"])
    test_texts = truncate_list(test_texts, config["max_eval_texts"])
    print(f"[LOG] Dataset truncated - train: {len(train_texts)} samples, test: {len(test_texts)} samples")

    print(f"[LOG] Inserting canaries - num: {config['num_canaries']}, repetitions: {config['canary_repetitions']}")
    augmented_train, canaries = insert_canaries(
        train_texts=train_texts,
        num_canaries=config["num_canaries"],
        repetitions=config["canary_repetitions"],
        seed=config["seed"],
    )
    print(f"[LOG] Canaries inserted - augmented train size: {len(augmented_train)}")

    print(f"[LOG] Sampling candidates - members: {config['member_sample_size']}, non-members: {config['nonmember_sample_size']}")
    candidates = sample_candidates(
        train_texts=augmented_train,
        test_texts=test_texts,
        canaries=canaries,
        member_sample_size=config["member_sample_size"],
        nonmember_sample_size=config["nonmember_sample_size"],
        seed=config["seed"],
    )
    print(f"[LOG] Candidates sampled - total: {len(candidates)}")

    print(f"[LOG] Starting model training on {config['num_train_epochs']} epochs")
    tokenizer, trained_model = train_model(
        config=config,
        train_texts=augmented_train,
        eval_texts=test_texts,
    )
    print(f"[LOG] Model training completed")

    print(f"[LOG] Setting up device and moving models")
    device = choose_device()
    print(f"[LOG] Device: {device}")
    trained_model = move_model_to_device(trained_model, device)
    print(f"[LOG] Trained model moved to device")

    print(f"[LOG] Loading reference model: {config['model_name']}")
    ref_tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if ref_tokenizer.pad_token is None:
        ref_tokenizer.pad_token = ref_tokenizer.eos_token
    ref_model = AutoModelForCausalLM.from_pretrained(config["model_name"])
    ref_model.config.pad_token_id = ref_tokenizer.pad_token_id
    ref_model = move_model_to_device(ref_model, device)
    print(f"[LOG] Reference model loaded and moved to device")

    labels = [c.label for c in candidates]
    sample_ids = [c.sample_id for c in candidates]
    sources = [c.source for c in candidates]
    texts = [c.text for c in candidates]

    print(f"[LOG] Starting attack execution")
    all_rows: List[Dict] = []
    summary: Dict[str, Dict] = {}

    for attack_name in config["attacks"]:
        print(f"[LOG] === Starting attack: {attack_name} ===")
        run_predictions: List[List[int]] = []
        run_metrics: List[Dict] = []

        for run_idx in range(config["num_attack_runs"]):
            print(f"[LOG]   Run {run_idx + 1}/{config['num_attack_runs']} for {attack_name}...")
            scores = repeated_scores_for_stability(
                attack_name=attack_name,
                target_model=trained_model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                texts=texts,
                max_length=config["max_length"],
                min_k_ratio=config["min_k_ratio"],
                rng_seed=config["seed"] + run_idx,
                neighborhood_variants=config["neighborhood_variants"],
            )
            print(f"[LOG]   Scores computed, calculating metrics...")
            metric_values = classification_metrics(
                labels=labels,
                scores=scores,
                smaller_is_member=ATTACK_SMALLER_IS_MEMBER[attack_name],
            )
            preds = predictions_from_scores(
                scores=scores,
                threshold=metric_values["threshold"],
                smaller_is_member=ATTACK_SMALLER_IS_MEMBER[attack_name],
            )
            run_predictions.append(preds)
            metric_values["coverage_all_members"] = coverage(sample_ids, labels, preds)
            metric_values["coverage_canaries"] = coverage_canaries(sample_ids, sources, preds)
            run_metrics.append(metric_values)
            print(f"[LOG]   Run {run_idx + 1} metrics: AUC={metric_values['auc']:.3f}, F1={metric_values['f1']:.3f}, Canary Coverage={metric_values['coverage_canaries']:.3f}")

            for candidate, score, pred in zip(candidates, scores, preds):
                all_rows.append(
                    {
                        "attack": attack_name,
                        "run": run_idx,
                        "sample_id": candidate.sample_id,
                        "source": candidate.source,
                        "label": candidate.label,
                        "prediction": pred,
                        "score": score,
                        "text": candidate.text,
                    }
                )

        summary[attack_name] = {
            "mean_auc": float(np.nanmean([m["auc"] for m in run_metrics])),
            "mean_precision": float(np.mean([m["precision"] for m in run_metrics])),
            "mean_recall": float(np.mean([m["recall"] for m in run_metrics])),
            "mean_f1": float(np.mean([m["f1"] for m in run_metrics])),
            "mean_coverage_all_members": float(np.mean([m["coverage_all_members"] for m in run_metrics])),
            "mean_coverage_canaries": float(np.mean([m["coverage_canaries"] for m in run_metrics])),
            "stability": stability(run_predictions, sample_ids),
        }
        print(f"[LOG] === Attack {attack_name} completed ===")

    print(f"[LOG] All attacks completed, saving results...")
    df = pd.DataFrame(all_rows)
    print(f"[LOG] DataFrame created with {len(df)} rows")
    
    csv_path = os.path.join(config["output_dir"], "attack_scores.csv")
    df.to_csv(csv_path, index=False)
    print(f"[LOG] Saved attack_scores.csv to {csv_path}")
    
    metrics_path = os.path.join(config["output_dir"], "metrics_summary.json")
    save_json(metrics_path, summary)
    print(f"[LOG] Saved metrics_summary.json to {metrics_path}")
    
    config_snapshot_path = os.path.join(config["output_dir"], "run_config_snapshot.json")
    save_json(config_snapshot_path, config)
    print(f"[LOG] Saved run_config_snapshot.json to {config_snapshot_path}")

    print("[LOG] ========== PIPELINE COMPLETED ==========")
    print("Finished. Summary metrics:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file.")
    args = parser.parse_args()
    main(args.config)
