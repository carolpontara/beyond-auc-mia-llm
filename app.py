import sys
import os
import json
import subprocess
from pathlib import Path

import gradio as gr
import pandas as pd

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "default_config.json"
OUTPUT_DIR = ROOT / "outputs"
METRICS_FILE = OUTPUT_DIR / "metrics_summary.json"
SCORES_FILE = OUTPUT_DIR / "attack_scores.csv"


def ensure_default_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config não encontrado: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return cfg


def run_training():
    ensure_default_config()

    cmd = [
        sys.executable,
        "src/run_pipeline.py",
        "--config",
        str(CONFIG_PATH),
    ]

    process = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True
    )

    logs = []
    logs.append("=== STDOUT ===\n")
    logs.append(process.stdout or "(sem stdout)\n")
    logs.append("\n=== STDERR ===\n")
    logs.append(process.stderr or "(sem stderr)\n")
    logs.append(f"\n=== RETURN CODE: {process.returncode} ===\n")

    metrics_text = "Arquivo de métricas ainda não gerado."
    scores_df = pd.DataFrame([{"status": "Arquivo CSV ainda não gerado."}])

    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics_text = json.dumps(json.load(f), indent=2, ensure_ascii=False)

    if SCORES_FILE.exists():
        scores_df = pd.read_csv(SCORES_FILE)

    return "".join(logs), metrics_text, scores_df


def show_existing_results():
    metrics_text = "Arquivo de métricas ainda não encontrado."
    scores_df = pd.DataFrame([{"status": "Arquivo CSV ainda não encontrado."}])

    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics_text = json.dumps(json.load(f), indent=2, ensure_ascii=False)

    if SCORES_FILE.exists():
        scores_df = pd.read_csv(SCORES_FILE)

    return metrics_text, scores_df


with gr.Blocks(title="MIA LLM Audit Trainer") as demo:
    gr.Markdown("# 🔐 MIA LLM Audit Trainer")
    gr.Markdown(
        "Clique em **Run Training Pipeline** para executar o treino e a auditoria diretamente no Space."
    )

    with gr.Row():
        run_btn = gr.Button("Run Training Pipeline", variant="primary")
        refresh_btn = gr.Button("Load Existing Results")

    logs_box = gr.Textbox(
        label="Execution Logs",
        lines=20,
        max_lines=30
    )

    metrics_box = gr.Code(
        label="metrics_summary.json",
        language="json"
    )

    scores_df = gr.Dataframe(
        label="attack_scores.csv",
        interactive=False
    )

    run_btn.click(
        fn=run_training,
        inputs=None,
        outputs=[logs_box, metrics_box, scores_df]
    )

    refresh_btn.click(
        fn=lambda: ("", *show_existing_results()),
        inputs=None,
        outputs=[logs_box, metrics_box, scores_df]
    )

demo.queue()
demo.launch()