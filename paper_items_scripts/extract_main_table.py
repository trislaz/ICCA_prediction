#!/usr/bin/env python3
"""
Script to extract the main table showing CV performances for each tile_filter with slidetype OPB.
Creates a table comparing different tile filtering approaches.
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path

def extract_main_table(base_path, label_name, output_dir):
    """Extract main CV performance table for slidetype OPB across different tile filters."""

    # Read the CV results
    cv_stats_path = Path(base_path) / label_name / "cv_results" / f"cv_summary_stats_{label_name}.csv"
    if not cv_stats_path.exists():
        print(f"Error: CV stats file not found at {cv_stats_path}")
        return None, None

    df = pd.read_csv(cv_stats_path)

    # Filter for SlideLevel and OPB slidetype only
    filtered_df = df.query('level == "SlideLevel" and slidetype == "OPB"')

    if filtered_df.empty:
        print("Warning: No data found for SlideLevel and OPB slidetype")
        return None, None

    # Select relevant columns and create the main table
    table_columns = [
        'tile_filter',
        'roc_auc_score_mean',
        'roc_auc_score_std',
        'specificity_mean',
        'specificity_std',
        'sensitivity_mean',
        'sensitivity_std',
        'valeur_predictive_positive_mean',
        'valeur_predictive_positive_std',
        'valeur_predictive_negative_mean',
        'valeur_predictive_negative_std'
    ]

    main_table = filtered_df[table_columns].copy()

    # Round numerical values for better presentation
    numeric_cols = [col for col in main_table.columns if col != 'tile_filter']
    main_table[numeric_cols] = main_table[numeric_cols].round(3)

    # Sort by ROC AUC performance (descending)
    main_table = main_table.sort_values('roc_auc_score_mean', ascending=False)

    # Create a formatted version with mean ± std format
    formatted_table = pd.DataFrame()
    formatted_table['Tile Filter'] = main_table['tile_filter']

    # Format as mean ± std for each metric
    formatted_table['ROC AUC'] = main_table.apply(
        lambda row: f"{row['roc_auc_score_mean']:.3f} ± {row['roc_auc_score_std']:.3f}", axis=1
    )
    formatted_table['Specificity'] = main_table.apply(
        lambda row: f"{row['specificity_mean']:.3f} ± {row['specificity_std']:.3f}", axis=1
    )
    formatted_table['Sensitivity'] = main_table.apply(
        lambda row: f"{row['sensitivity_mean']:.3f} ± {row['sensitivity_std']:.3f}", axis=1
    )
    formatted_table['PPV'] = main_table.apply(
        lambda row: f"{row['valeur_predictive_positive_mean']:.3f} ± {row['valeur_predictive_positive_std']:.3f}", axis=1
    )
    formatted_table['NPV'] = main_table.apply(
        lambda row: f"{row['valeur_predictive_negative_mean']:.3f} ± {row['valeur_predictive_negative_std']:.3f}", axis=1
    )

    # Save both versions
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_path_raw = output_path / f"main_table_OPB_raw_{label_name}.csv"
    main_table.to_csv(output_path_raw, index=False)
    print(f"Raw table saved to: {output_path_raw}")

    output_path_formatted = output_path / f"main_table_OPB_formatted_{label_name}.csv"
    formatted_table.to_csv(output_path_formatted, index=False)
    print(f"Formatted table saved to: {output_path_formatted}")

    # Print the formatted table
    print("\nMain Table - CV Performance for Slidetype OPB (SlideLevel):")
    print("=" * 80)
    print(formatted_table.to_string(index=False))
    print("=" * 80)

    # Print performance ranking
    print(f"\nPerformance Ranking by ROC AUC:")
    for i, (idx, row) in enumerate(main_table.iterrows(), 1):
        print(f"{i}. {row['tile_filter']}: {row['roc_auc_score_mean']:.3f} ± {row['roc_auc_score_std']:.3f}")

    return main_table, formatted_table

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract main CV performance table for slidetype OPB"
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
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for generated files'
    )

    args = parser.parse_args()

    # Extract main table
    results = extract_main_table(args.base_path, args.label, args.output_dir)

    if results[0] is None:
        sys.exit(1)

if __name__ == "__main__":
    main()
