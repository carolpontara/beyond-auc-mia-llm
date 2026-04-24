"""Visualization module for MIA experiment results."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _short_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def plot_auc_by_attack_model(
    summary: Dict[str, Dict],
    output_dir: str,
    dataset_label: str = "",
) -> str:
    """Bar chart comparing AUC across models for each attack."""
    models = sorted(summary.keys())
    if not models:
        return ""
    attacks = sorted(summary[models[0]].keys())
    x = np.arange(len(attacks))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(max(10, len(attacks) * 2), 6))
    for i, model in enumerate(models):
        aucs = [summary[model].get(atk, {}).get("auc", {}).get("mean", 0) for atk in attacks]
        stds = [summary[model].get(atk, {}).get("auc", {}).get("std", 0) for atk in attacks]
        ax.bar(x + i * width, aucs, width, yerr=stds, label=_short_name(model), capsize=3)
    ax.set_ylabel("AUC")
    title = "AUC by Attack and Model"
    if dataset_label:
        title += " -- " + dataset_label
    ax.set_title(title)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(attacks, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    suffix = "_" + dataset_label if dataset_label else ""
    path = os.path.join(output_dir, f"auc_by_attack_model{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_heatmap_attack_dataset(
    results: Dict[str, Dict],
    metric_key: str,
    output_dir: str,
    model_label: str = "",
) -> str:
    """Heatmap of a metric across attacks and datasets."""
    datasets = sorted(results.keys())
    if not datasets:
        return ""
    attacks = sorted(results[datasets[0]].keys())
    matrix = []
    for atk in attacks:
        row = []
        for ds in datasets:
            val = results[ds].get(atk, {}).get(metric_key, {}).get("mean", float("nan"))
            row.append(val)
        matrix.append(row)
    matrix_np = np.array(matrix)
    fig, ax = plt.subplots(figsize=(max(8, len(datasets) * 2.5), max(5, len(attacks) * 0.8)))
    im = ax.imshow(matrix_np, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_yticks(range(len(attacks)))
    ax.set_yticklabels(attacks)
    for i in range(len(attacks)):
        for j in range(len(datasets)):
            val = matrix_np[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=9)
    title = f"{metric_key.upper()} -- Attack x Dataset"
    if model_label:
        title += f" ({_short_name(model_label)})"
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    suffix = "_" + _short_name(model_label) if model_label else ""
    path = os.path.join(output_dir, f"heatmap_{metric_key}{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_canary_by_repetition(
    canary_results: Dict[int, Dict[str, float]],
    output_dir: str,
    model_label: str = "",
) -> str:
    """Line chart of canary coverage vs repetition count."""
    reps = sorted(canary_results.keys())
    if not reps:
        return ""
    attacks = sorted(canary_results[reps[0]].keys())
    fig, ax = plt.subplots(figsize=(8, 5))
    for atk in attacks:
        vals = [canary_results[r].get(atk, 0.0) for r in reps]
        ax.plot(reps, vals, marker="o", label=atk)
    ax.set_xlabel("Canary Repetitions (r)")
    ax.set_ylabel("Canary Coverage")
    title = "Canary Detection by Repetition"
    if model_label:
        title += " -- " + _short_name(model_label)
    ax.set_title(title)
    ax.set_xticks(reps)
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    suffix = "_" + _short_name(model_label) if model_label else ""
    path = os.path.join(output_dir, f"canary_by_repetition{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_f1_threshold_comparison(
    summary: Dict[str, Dict],
    output_dir: str,
    label: str = "",
) -> str:
    """F1 comparison: median threshold vs best-F1 threshold."""
    attacks = sorted(summary.keys())
    if not attacks:
        return ""
    f1_median = [summary[a].get("f1_median", {}).get("mean", 0) for a in attacks]
    f1_best = [summary[a].get("f1_f1", {}).get("mean", 0) for a in attacks]
    x = np.arange(len(attacks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(attacks) * 1.5), 5))
    ax.bar(x - width / 2, f1_median, width, label="Median Threshold")
    ax.bar(x + width / 2, f1_best, width, label="Best-F1 Threshold")
    ax.set_ylabel("F1 Score")
    title = "F1: Median vs Best-F1 Threshold"
    if label:
        title += " -- " + label
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    suffix = "_" + label if label else ""
    path = os.path.join(output_dir, f"f1_threshold_comparison{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_tpr_at_fpr(
    summary: Dict[str, Dict],
    output_dir: str,
    label: str = "",
) -> str:
    """TPR at various FPR thresholds per attack."""
    attacks = sorted(summary.keys())
    if not attacks:
        return ""
    fpr_keys = sorted([k for k in summary[attacks[0]] if k.startswith("tpr@fpr=")])
    if not fpr_keys:
        return ""
    x = np.arange(len(attacks))
    width = 0.8 / max(len(fpr_keys), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(attacks) * 2), 5))
    for i, fk in enumerate(fpr_keys):
        vals = [summary[a].get(fk, {}).get("mean", 0) for a in attacks]
        ax.bar(x + i * width, vals, width, label=fk)
    ax.set_ylabel("TPR")
    title = "TPR at Low FPR"
    if label:
        title += " -- " + label
    ax.set_title(title)
    ax.set_xticks(x + width * (len(fpr_keys) - 1) / 2)
    ax.set_xticklabels(attacks, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    suffix = "_" + label if label else ""
    path = os.path.join(output_dir, f"tpr_at_fpr{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_stability_comparison(
    stability_data: Dict[str, Dict[str, float]],
    output_dir: str,
) -> str:
    """Stability (Jaccard) across models and attacks."""
    models = sorted(stability_data.keys())
    if not models:
        return ""
    attacks = sorted(stability_data[models[0]].keys())
    x = np.arange(len(attacks))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(max(10, len(attacks) * 2), 6))
    for i, model in enumerate(models):
        vals = [stability_data[model].get(a, 0) for a in attacks]
        ax.bar(x + i * width, vals, width, label=_short_name(model))
    ax.set_ylabel("Stability (Jaccard)")
    ax.set_title("Stability Across Models")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(attacks, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "stability_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_vulnerability_scatter(
    df: pd.DataFrame,
    output_dir: str,
    label: str = "",
) -> str:
    """Scatter plot of text length vs token rarity coloured by membership."""
    if df.empty:
        return ""
    fig, ax = plt.subplots(figsize=(9, 6))
    members = df[df["label"] == 1]
    nonmembers = df[df["label"] == 0]
    ax.scatter(
        nonmembers["text_length"], nonmembers["token_rarity"],
        alpha=0.4, s=15, label="Non-member", color="blue",
    )
    ax.scatter(
        members["text_length"], members["token_rarity"],
        alpha=0.4, s=15, label="Member", color="red",
    )
    ax.set_xlabel("Text Length (tokens)")
    ax.set_ylabel("Token Rarity (mean IDF)")
    title = "Sample Vulnerability"
    if label:
        title += " -- " + label
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    suffix = "_" + label if label else ""
    path = os.path.join(output_dir, f"vulnerability_scatter{suffix}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_coverage_comparison(
    coverage_data: Dict[str, Dict[str, float]],
    output_dir: str,
    metric_label: str = "coverage",
) -> str:
    """Coverage comparison across models."""
    models = sorted(coverage_data.keys())
    if not models:
        return ""
    attacks = sorted(coverage_data[models[0]].keys())
    x = np.arange(len(attacks))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(max(10, len(attacks) * 2), 6))
    for i, model in enumerate(models):
        vals = [coverage_data[model].get(a, 0) for a in attacks]
        ax.bar(x + i * width, vals, width, label=_short_name(model))
    ax.set_ylabel(metric_label.replace("_", " ").title())
    ax.set_title(metric_label.replace("_", " ").title() + " Across Models")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(attacks, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{metric_label}_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_all_plots(output_dir: str) -> List[str]:
    """Master function: reads full_summary.json and generates all plots."""
    _safe_mkdir(os.path.join(output_dir, "plots"))
    plots_dir = os.path.join(output_dir, "plots")
    generated: List[str] = []

    full_summary_path = os.path.join(output_dir, "full_summary.json")
    if not os.path.exists(full_summary_path):
        return generated

    with open(full_summary_path, "r") as f:
        full = json.load(f)

    # 1. AUC by attack and model (per dataset)
    for ds_key, ds_data in full.get("by_dataset", {}).items():
        model_attack_auc: Dict[str, Dict] = {}
        for model_key, model_data in ds_data.get("by_model", {}).items():
            model_attack_auc[model_key] = {}
            for atk, atk_data in model_data.get("metrics", {}).items():
                model_attack_auc[model_key][atk] = atk_data
        path = plot_auc_by_attack_model(model_attack_auc, plots_dir, dataset_label=ds_key)
        if path:
            generated.append(path)

    # 2. Heatmap attack x dataset (per model)
    all_models = set()
    for ds_key, ds_data in full.get("by_dataset", {}).items():
        all_models.update(ds_data.get("by_model", {}).keys())
    for model in all_models:
        ds_atk: Dict[str, Dict] = {}
        for ds_key, ds_data in full.get("by_dataset", {}).items():
            model_data = ds_data.get("by_model", {}).get(model, {})
            ds_atk[ds_key] = model_data.get("metrics", {})
        path = plot_heatmap_attack_dataset(ds_atk, "auc", plots_dir, model_label=model)
        if path:
            generated.append(path)

    # 3. Canary by repetition
    canary_data = full.get("canary_by_repetition", {})
    if canary_data:
        for model_key, rep_data in canary_data.items():
            int_rep_data = {int(k): v for k, v in rep_data.items()}
            path = plot_canary_by_repetition(int_rep_data, plots_dir, model_label=model_key)
            if path:
                generated.append(path)

    # 4. F1 threshold comparison and TPR at FPR (per model x dataset)
    for ds_key, ds_data in full.get("by_dataset", {}).items():
        for model_key, model_data in ds_data.get("by_model", {}).items():
            lbl = _short_name(model_key) + "_" + ds_key
            path = plot_f1_threshold_comparison(
                model_data.get("metrics", {}), plots_dir, label=lbl,
            )
            if path:
                generated.append(path)
            path2 = plot_tpr_at_fpr(
                model_data.get("metrics", {}), plots_dir, label=lbl,
            )
            if path2:
                generated.append(path2)

    # 5. Stability comparison
    stability_all: Dict[str, Dict] = {}
    for ds_key, ds_data in full.get("by_dataset", {}).items():
        for model_key, model_data in ds_data.get("by_model", {}).items():
            key = _short_name(model_key) + "_" + ds_key
            stability_all[key] = model_data.get("stability", {})
    if stability_all:
        path = plot_stability_comparison(stability_all, plots_dir)
        if path:
            generated.append(path)

    # 6. Coverage comparison
    coverage_all: Dict[str, Dict] = {}
    for ds_key, ds_data in full.get("by_dataset", {}).items():
        for model_key, model_data in ds_data.get("by_model", {}).items():
            key = _short_name(model_key) + "_" + ds_key
            cov: Dict[str, float] = {}
            for atk, atk_data in model_data.get("metrics", {}).items():
                cov[atk] = atk_data.get("coverage_all_members", {}).get("mean", 0)
            coverage_all[key] = cov
    if coverage_all:
        path = plot_coverage_comparison(coverage_all, plots_dir, "coverage")
        if path:
            generated.append(path)

    # 7. Vulnerability scatter
    analysis_path = os.path.join(output_dir, "sample_analysis.csv")
    if os.path.exists(analysis_path):
        df = pd.read_csv(analysis_path)
        path = plot_vulnerability_scatter(df, plots_dir, label="all")
        if path:
            generated.append(path)

    print(f"[PLOTS] Generated {len(generated)} charts in {plots_dir}")
    return generated
