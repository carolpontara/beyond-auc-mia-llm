import sys
import os
import json
import glob
import subprocess
from pathlib import Path

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "default_config.json"
OUTPUT_DIR = ROOT / "output"
FULL_SUMMARY = OUTPUT_DIR / "full_summary.json"
SCORES_FILE = OUTPUT_DIR / "attack_scores.csv"
ANALYSIS_FILE = OUTPUT_DIR / "sample_analysis.csv"
PLOTS_DIR = OUTPUT_DIR / "plots"


def ensure_default_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return cfg


def run_pipeline():
    """Execute the enhanced multi-model/multi-dataset pipeline."""
    ensure_default_config()
    cmd = [
        sys.executable,
        "src/run_pipeline.py",
        "--config",
        str(CONFIG_PATH),
    ]
    process = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)

    logs = []
    logs.append("=== STDOUT ===\n")
    logs.append(process.stdout or "(no stdout)\n")
    logs.append("\n=== STDERR ===\n")
    logs.append(process.stderr or "(no stderr)\n")
    logs.append(f"\n=== RETURN CODE: {process.returncode} ===\n")

    summary_text, scores_df, analysis_df, plot_paths = _load_results()
    return "".join(logs), summary_text, scores_df, analysis_df, plot_paths


def _load_results():
    """Load existing pipeline results from output directory."""
    summary_text = "No results yet."
    scores_df = pd.DataFrame([{"status": "No scores file found."}])
    analysis_df = pd.DataFrame([{"status": "No analysis file found."}])
    plot_paths = []

    if FULL_SUMMARY.exists():
        with open(FULL_SUMMARY, "r", encoding="utf-8") as f:
            summary_text = json.dumps(json.load(f), indent=2, ensure_ascii=False)

    if SCORES_FILE.exists():
        scores_df = pd.read_csv(SCORES_FILE)

    if ANALYSIS_FILE.exists():
        analysis_df = pd.read_csv(ANALYSIS_FILE)

    if PLOTS_DIR.exists():
        plot_paths = sorted(glob.glob(str(PLOTS_DIR / "*.png")))

    return summary_text, scores_df, analysis_df, plot_paths


def show_existing_results():
    summary_text, scores_df, analysis_df, plot_paths = _load_results()
    return "", summary_text, scores_df, analysis_df, plot_paths


with gr.Blocks(title="Beyond AUC: MIA LLM Audit") as demo:
    gr.Markdown("# Beyond AUC: MIA LLM Audit Pipeline")
    gr.Markdown(
        "Multi-model, multi-dataset membership inference attack evaluation. "
        "Click **Run Pipeline** to train models and run all attacks, "
        "or **Load Results** to view existing output."
    )

    with gr.Row():
        run_btn = gr.Button("Run Pipeline", variant="primary")
        refresh_btn = gr.Button("Load Results")

    logs_box = gr.Textbox(label="Execution Logs", lines=15, max_lines=30)

    with gr.Tabs():
        with gr.TabItem("Summary"):
            summary_box = gr.Code(label="full_summary.json", language="json")
        with gr.TabItem("Attack Scores"):
            scores_table = gr.Dataframe(label="attack_scores.csv", interactive=False)
        with gr.TabItem("Sample Analysis"):
            analysis_table = gr.Dataframe(label="sample_analysis.csv", interactive=False)
        with gr.TabItem("Plots"):
            gallery = gr.Gallery(label="Generated Charts", columns=2, height="auto")

    run_btn.click(
        fn=run_pipeline,
        inputs=None,
        outputs=[logs_box, summary_box, scores_table, analysis_table, gallery],
    )

    refresh_btn.click(
        fn=show_existing_results,
        inputs=None,
        outputs=[logs_box, summary_box, scores_table, analysis_table, gallery],
    )

demo.queue()
demo.launch()
