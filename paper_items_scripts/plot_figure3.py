#!/usr/bin/env python3
"""
Script to reproduce Figure 3 for any validation dataset.
Creates a horizontal bar plot showing ROC AUC scores for different slide types.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
from pathlib import Path

def create_figure3(base_path, label_name, dataset_name, output_dir):
    """Create Figure 3 for specified validation dataset."""

    # Read the validation results
    results_path = Path(base_path) / label_name / "validation_results" / f"validation_summary_{label_name}.csv"
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        return False

    df = pd.read_csv(results_path)

    # Filter data: SlideLevel, same tile_filter for train and valid, filter type, specified dataset
    filtered_df = df.query(
        'level_train == "SlideLevel" and '
        'tile_filter_train == tile_filter_valid and '
        'tile_filter_valid == "filter" and '
        f'dataset_valid == "{dataset_name}"'
    )

    if filtered_df.empty:
        print(f"Warning: No data found matching the criteria for dataset '{dataset_name}'")
        return False

    # Define slide type order (from best to worst performance typically)
    order = ["O", "OB", "OPB", "OP", "P", "PB", "B"]

    # Set color based on dataset
    colors = {
        'TCGA': 'mediumpurple',
        'mondor': 'lightcoral',
        'TCGA_frozen': 'skyblue',
        'mondor_2': 'lightgreen'
    }
    color = colors.get(dataset_name, 'gray')

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=filtered_df,
        x='roc_auc_score',
        y='slidetype_train',
        orient="h",
        order=order,
        color=color
    )

    # Customize the plot
    plt.xlim([0.6, 0.9])
    plt.xlabel('ROC AUC Score')
    plt.ylabel('Slide Type (Train)')
    plt.title(f'Figure 3: ROC AUC Performance by Slide Type - {dataset_name.upper()} Validation')
    plt.grid(axis='x', alpha=0.3)

    # Add value labels on bars
    for i, (idx, row) in enumerate(filtered_df.set_index('slidetype_train').reindex(order).iterrows()):
        if pd.notna(row['roc_auc_score']):
            plt.text(row['roc_auc_score'] + 0.005, i, f'{row["roc_auc_score"]:.3f}',
                    va='center', fontsize=10)

    plt.tight_layout()

    # Save the plot
    output_path_dir = Path(output_dir)
    output_path_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_path_dir / f"figure3_{dataset_name}_{label_name}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

    # Also save as PDF
    output_path_pdf = output_path_dir / f"figure3_{dataset_name}_{label_name}.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Figure saved to: {output_path_pdf}")

    plt.close()

    # Print summary statistics
    print(f"\nSummary Statistics ({dataset_name.upper()}):")
    summary = filtered_df[['slidetype_train', 'roc_auc_score']].sort_values('roc_auc_score', ascending=False)
    print(summary.to_string(index=False))

    return True

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Figure 3 for a specific validation dataset"
    )
    parser.add_argument(
        '--base_path',
        type=str,
        required=True,
        help='Base directory containing results (e.g., ./outputs)'
    )
    parser.add_argument(
        '--label',
        type=str,
        required=True,
        help='Label name to process'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Validation dataset name (e.g., TCGA, mondor, TCGA_frozen, mondor_2)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for generated files'
    )

    args = parser.parse_args()

    success = create_figure3(args.base_path, args.label, args.dataset, args.output_dir)

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()