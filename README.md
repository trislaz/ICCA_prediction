# GigaSSL Reproduction

Clean, standalone reproduction of GigaSSL linear probing experiments.

## Prerequisites

You will need [uv](https://docs.astral.sh/uv/getting-started/installation/) to run these scripts.

## Quick Start - Run Complete Pipeline

**The easiest way to reproduce all results:**

```bash
# 1. Clone this repository and navigate to it
cd ICCA_prediction

# 2. Install dependencies
uv sync

# 3. Run the complete pipeline (downloads data, trains all models, analyzes results)
./main.sh
```

This script will:
1. Download all training and validation datasets from Google Drive
2. Run the full pipeline for all 5 labels (HepaticStem_like, Desert_like, Immune_classical, Inflammatory_stroma, Tumor_classical)
3. Perform binary classification analysis

Results will be saved in `outputs/` directory.

## Manual Run

If you want to run individual steps:

```bash
# Download datasets only
uv run downloads.py

# Run cross-validation for a specific label
uv run run_pipeline.py --label_name HepaticStem_like

# Analyze binary classification results
uv run analyze_binary_classifications.py
```

## Paper Items (run from root)

```bash
# Generate Figure 3 for TCGA
uv run python gigassl_reproduction/paper_items_scripts/plot_figure3.py TCGA

# Generate Figure 3 for Mondor
uv run python gigassl_reproduction/paper_items_scripts/plot_figure3.py mondor

# Extract main CV performance table (OPB slidetype)
uv run python gigassl_reproduction/paper_items_scripts/extract_main_table.py

# Extract validation results for specific slidetype
uv run python gigassl_reproduction/paper_items_scripts/extract_validation_results.py OPB

# Generate everything at once
uv run python gigassl_reproduction/paper_items_scripts/generate_all_paper_items.py
```
