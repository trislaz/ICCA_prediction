#!/bin/bash

set -e  # Exit on error

echo "=========================================="
echo "ICCA Reproduction Pipeline - Full Run"
echo "=========================================="

# Step 1: Download datasets
echo ""
echo "[1/3] Downloading datasets..."
uv run downloads.py

# Step 2: Run pipeline for all labels
echo ""
echo "[2/3] Running pipeline for all labels..."
labels=("HepaticStem_like" "Desert_like" "Immune_classical" "Inflammatory_stroma" "Tumor_classical")

for label in "${labels[@]}"; do
    echo ""
    echo ">>> Running pipeline for label: $label"
    uv run run_pipeline.py --label_name "$label"
done

# Step 3: Analyze binary classification results
echo ""
echo "[3/3] Analyzing binary classification results..."
uv run analyze_binary_classifications.py

echo ""
echo "=========================================="
echo "All steps completed successfully!"
echo "=========================================="