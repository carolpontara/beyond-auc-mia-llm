---
title: MIA LLM Audit Trainer
emoji: 🔐
colorFrom: blue
colorTo: purple
sdk: gradio
python_version: "3.10"
app_file: app.py
pinned: false
suggested_hardware: cpu-upgrade
---

# Beyond AUC: Evaluating Coverage and Stability of Membership Inference Attacks in Large Language Models

## Authors

- **Ana Caroline Silva Pontara** — Universidade Estadual Paulista (UNESP), Brazil
- **Daniel Pederzini**

## Overview

This repository contains the code and experimental outputs for the paper *"Beyond AUC: Evaluating Coverage and Stability of Membership Inference Attacks in Large Language Models"*.

We propose a **multi-attack auditing protocol** that evaluates privacy leakage in Large Language Models (LLMs) beyond traditional aggregate metrics (AUC, precision, recall, F1). The protocol introduces two complementary evaluation dimensions:

- **Coverage**: the fraction of true training members detected by individual attacks and their union, including canary-specific coverage.
- **Stability**: the consistency of attack predictions across independent runs, measured by pairwise Jaccard similarity.

## Attack Strategies

We implement and evaluate three membership inference attacks:

| Attack | Signal | Deterministic? |
|---|---|---|
| **Loss-based** | Cross-entropy loss of the fine-tuned model | Yes |
| **Reference-gap** | Difference in loss between fine-tuned and pre-trained models | Yes |
| **Min-K%** | Average of the k smallest token log-probabilities, with neighbourhood perturbations | No |

## Key Results

- Attacks with **identical** precision/recall/F1 detect substantially **different** samples (Jaccard overlap as low as 0.55)
- The **union** of three attacks detects **38% more** unique memorized samples than any single attack
- All attacks achieve **100% canary detection**, but diverge on natural training data
- Deterministic attacks have **perfect stability** (1.0); Min-K% has stability of **0.87**

## Project Structure

```
.
├── app.py                      # Gradio web interface
├── config/
│   └── default_config.json     # Pipeline configuration
├── src/
│   ├── data.py                 # Dataset loading, canary generation, candidate sampling
│   ├── train.py                # Model fine-tuning via HuggingFace Trainer
│   ├── attacks.py              # Loss, reference-gap, and Min-K% attack implementations
│   ├── metrics.py              # Classification metrics, coverage, and stability
│   └── run_pipeline.py         # End-to-end pipeline orchestration
├── output/                     # Experimental results
│   ├── attack_scores.csv       # Per-sample scores (2925 rows)
│   ├── metrics_summary.json    # Aggregate metrics per attack
│   └── run_config_snapshot.json# Configuration snapshot
└── requirements.txt            # Python dependencies
```

## Setup

### Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

### Installation

```bash
pip install -r requirements.txt
```

### Running the Pipeline

**Option 1 — Command line:**

```bash
python src/run_pipeline.py --config config/default_config.json
```

**Option 2 — Gradio web interface:**

```bash
python app.py
```

Then open the local URL shown in the terminal. The interface provides two buttons:
- **Run Training Pipeline** — launches the full pipeline and displays logs, metrics, and scores
- **Load Existing Results** — loads previously generated output files

### Hugging Face Spaces

This application can also be deployed on [Hugging Face Spaces](https://huggingface.co/spaces) using the Gradio SDK.

## Configuration

All parameters are controlled via `config/default_config.json`:

| Parameter | Value | Description |
|---|---|---|
| `model_name` | `distilgpt2` | Pre-trained model to fine-tune |
| `dataset_name` / `dataset_config` | `wikitext` / `wikitext-2-raw-v1` | Training dataset |
| `max_train_texts` | 4000 | Number of training texts sampled |
| `num_canaries` | 25 | Canary sequences inserted |
| `canary_repetitions` | 4 | Repetitions per canary |
| `member_sample_size` | 150 | Members in candidate set |
| `nonmember_sample_size` | 150 | Non-members in candidate set |
| `num_attack_runs` | 3 | Independent runs per attack |
| `attacks` | `["loss", "ref_gap", "min_k"]` | Attack strategies to run |
| `min_k_ratio` | 0.2 | Fraction of tokens for Min-K% |
| `neighbourhood_variants` | 3 | Variants for Min-K% perturbation |

## License

This project is released for academic and research purposes.
