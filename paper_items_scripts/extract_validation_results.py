#!/usr/bin/env python3
"""
Script to extract validation results for models trained and validated with the same tile filter,
across all available validation datasets. The slidetype used in training is provided as an argument.
"""

import pandas as pd
import argparse
import sys
from pathlib import Path

def extract_validation_results(base_path, label_name, train_slidetype, output_dir):
    """
    Extract validation results for a specific training slidetype across all validation datasets.

    Args:
        base_path: Base directory containing results
        label_name: Label name being processed
        train_slidetype: The slidetype used during training (e.g., 'OPB', 'O', 'P', etc.)
        output_dir: Output directory for generated files
    """
    # Read the validation results
    results_path = Path(base_path) / label_name / "validation_results" / f"validation_summary_{label_name}.csv"
    if not results_path.exists():
        print(f"Error: Results file not found at {results_path}")
        return None, None

    df = pd.read_csv(results_path)

    # Filter for the specified training slidetype, SlideLevel, and matching tile filters
    filtered_df = df.query(
        f'slidetype_train == "{train_slidetype}" and '
        'level_train == "SlideLevel" and '
        'tile_filter_train == tile_filter_valid'
    )

    if filtered_df.empty:
        print(f"Warning: No data found for training slidetype '{train_slidetype}' with matching tile filters")
        return None, None

    # Select relevant columns for the output
    output_columns = [
        'dataset_valid',
        'tile_filter_train',
        'tile_filter_valid',
        'roc_auc_score',
        'f1_score',
        'precision_score',
        'recall_score',
        'slidetype_train',
        'level_train'
    ]

    result_table = filtered_df[output_columns].copy()

    # Sort by validation dataset and tile filter
    result_table = result_table.sort_values(['dataset_valid', 'tile_filter_valid'])

    # Round numerical values
    numeric_cols = ['roc_auc_score', 'f1_score', 'precision_score', 'recall_score']
    result_table[numeric_cols] = result_table[numeric_cols].round(3)

    # Create a summary pivot table
    pivot_table = result_table.pivot_table(
        index='tile_filter_train',
        columns='dataset_valid',
        values='roc_auc_score',
        aggfunc='first'
    ).round(3)

    # Save results
    output_path_dir = Path(output_dir)
    output_path_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_path_dir / f"validation_results_{train_slidetype}_{label_name}.csv"
    result_table.to_csv(output_path, index=False)
    print(f"Validation results saved to: {output_path}")

    output_path_pivot = output_path_dir / f"validation_results_{train_slidetype}_{label_name}_pivot.csv"
    pivot_table.to_csv(output_path_pivot)
    print(f"Pivot table saved to: {output_path_pivot}")

    # Print results
    print(f"\nValidation Results for Training Slidetype: {train_slidetype}")
    print("=" * 80)
    print("Detailed Results:")
    print(result_table.to_string(index=False))
    print("\n" + "=" * 80)
    print("ROC AUC Summary (Pivot Table):")
    print(pivot_table.to_string())
    print("=" * 80)

    # Print performance summary
    print(f"\nPerformance Summary by Validation Dataset:")
    for dataset in result_table['dataset_valid'].unique():
        dataset_data = result_table[result_table['dataset_valid'] == dataset]
        best_performance = dataset_data.loc[dataset_data['roc_auc_score'].idxmax()]
        print(f"\n{dataset}:")
        print(f"  Best: {best_performance['tile_filter_train']} filter (ROC AUC: {best_performance['roc_auc_score']:.3f})")
        for _, row in dataset_data.iterrows():
            print(f"    {row['tile_filter_train']}: ROC AUC={row['roc_auc_score']:.3f}, "
                  f"F1={row['f1_score']:.3f}, Precision={row['precision_score']:.3f}, "
                  f"Recall={row['recall_score']:.3f}")

    return result_table, pivot_table

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract validation results for a specific training slidetype"
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
        '--slidetype',
        type=str,
        required=True,
        help='Slidetype used in training (e.g., OPB, O, P, OP, PB, B)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for generated files'
    )

    args = parser.parse_args()

    # Validate slidetype argument
    valid_slidetypes = ['O', 'OB', 'OPB', 'OP', 'P', 'PB', 'B']
    if args.slidetype not in valid_slidetypes:
        print(f"Error: Invalid slidetype '{args.slidetype}'. "
              f"Valid options are: {', '.join(valid_slidetypes)}")
        sys.exit(1)

    # Extract validation results
    results = extract_validation_results(args.base_path, args.label, args.slidetype, args.output_dir)

    if results[0] is None:
        sys.exit(1)

if __name__ == "__main__":
    main()