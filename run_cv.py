"""
Simplified script to run cross-validation experiments on multiple datasets.
Replaces the complex parameter parsing with a cleaner configuration approach.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
try:
    from .crossval import cross_validate, save_results
    from .utils import LabelColumn
except ImportError:
    # When running as standalone script from root
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from crossval import cross_validate, save_results
    from utils import LabelColumn


def parse_dataset_params(dataset_path):
    """Extract parameters from dataset path following the original naming convention."""
    path_str = str(dataset_path)
    parts = path_str.split('/')
    
    # Find the relevant parts (keeping compatibility with original structure)
    try:
        slidetype = parts[-2] if parts[-2] in {'B', 'OP', 'OPB', 'P', 'O', 'rO', 'PB', 'OB'} else 'unknown'
        level = parts[-3] if parts[-3] in {'SlideLevel', 'PatientLevel'} else 'unknown'
        tile_filter = parts[-4] if parts[-4] in {'QC', 'none', 'filter', 'QC_filter'} else 'unknown'
        
        return {
            'slidetype': slidetype,
            'level': level, 
            'tile_filter': tile_filter,
            'dataset_path': str(dataset_path)
        }
    except IndexError:
        return {
            'slidetype': 'unknown',
            'level': 'unknown',
            'tile_filter': 'unknown',
            'dataset_path': str(dataset_path)
        }


def get_label_csv_for_level(level, slide_level_csv=None, patient_level_csv=None):
    """Get the appropriate label CSV based on the level."""
    if level == 'SlideLevel' and slide_level_csv:
        return slide_level_csv
    elif level == 'PatientLevel' and patient_level_csv:
        return patient_level_csv
    else:
        # Return the first available CSV
        return slide_level_csv or patient_level_csv


def run_cv_experiment(datasets_dir, label_name, output_dir, 
                     slide_level_csv=None, patient_level_csv=None,
                     cv_type='fixed', score_types=None, n_splits=5):
    """
    Run cross-validation experiments on multiple datasets.
    
    Args:
        datasets_dir: Directory containing dataset subdirectories
        label_name: Name of the label column
        output_dir: Directory to save results
        slide_level_csv: CSV file for slide-level labels
        patient_level_csv: CSV file for patient-level labels
        cv_type: Type of cross-validation ('fixed' or 'random')
        score_types: List of score types to calculate
        n_splits: Number of folds for random CV
    """
    if score_types is None:
        score_types = ['roc_auc_score', 'specificity', 'sensitivity', 
                      'valeur_predictive_positive', 'valeur_predictive_negative']
    
    datasets_dir = Path(datasets_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all dataset directories
    dataset_paths = list(datasets_dir.rglob('dataset'))
    if not dataset_paths:
        # Try to find directories with embeddings.npy
        dataset_paths = [p.parent for p in datasets_dir.rglob('embeddings.npy')]
    
    print(f"Found {len(dataset_paths)} datasets to process")
    
    all_results = []
    
    for dataset_path in dataset_paths:
        print(f"\nProcessing: {dataset_path}")
        
        # Extract parameters from path
        params = parse_dataset_params(dataset_path)
        
        # Get appropriate label CSV
        label_csv = get_label_csv_for_level(
            params['level'], slide_level_csv, patient_level_csv
        )
        
        if not label_csv or not Path(label_csv).exists():
            print(f"Warning: No label CSV found for {dataset_path}")
            continue
        
        try:
            # Run cross-validation
            scores_df, classifiers, predictions_df, normalizer = cross_validate(
                dataset_path=dataset_path,
                label_csv=label_csv,
                cv_type=cv_type,
                label_name=label_name,
                n_splits=n_splits,
                score_types=score_types
            )
            
            # Add dataset parameters to results
            for param, value in params.items():
                scores_df[param] = value
            
            # Save individual results
            name_prefix = f"{params['tile_filter']}_{params['level']}_{params['slidetype']}"
            save_results(
                scores_df, classifiers, predictions_df, normalizer,
                output_dir / "individual_results",
                name_prefix, label_name
            )
            
            # Add to combined results (exclude summary rows for now)
            fold_results = scores_df[scores_df['fold'].apply(lambda x: str(x).isdigit())].copy()
            all_results.append(fold_results)
            
        except Exception as e:
            print(f"Error processing {dataset_path}: {e}")
            continue
    
    # Save combined results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(output_dir / f"cv_summary_{label_name}.csv", index=False)
        print(f"\nCombined results saved to {output_dir / f'cv_summary_{label_name}.csv'}")
        
        # Calculate and save aggregated statistics
        numeric_cols = combined_df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'fold']
        
        summary_stats = []
        for params_combo in combined_df.groupby(['tile_filter', 'level', 'slidetype']):
            group_name, group_data = params_combo
            stats = {'tile_filter': group_name[0], 'level': group_name[1], 'slidetype': group_name[2]}
            for col in numeric_cols:
                stats[f'{col}_mean'] = group_data[col].mean()
                stats[f'{col}_std'] = group_data[col].std()
            summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv(output_dir / f"cv_summary_stats_{label_name}.csv", index=False)
        
    return output_dir


def main():
    """Main entry point for CV experiments."""
    parser = argparse.ArgumentParser(description='Run cross-validation experiments')
    parser.add_argument('--datasets_dir', type=str, required=True,
                       help='Directory containing dataset subdirectories')
    parser.add_argument('--label_name', type=str,
                       choices=LabelColumn.get_all_labels(),
                       default=LabelColumn.LABEL.value,
                       help=f'Name of the label column. Options: {LabelColumn.get_all_labels()}')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    parser.add_argument('--slide_level_csv', type=str,
                       default=LabelColumn.get_default_csv_path(),
                       help=f'CSV file for slide-level labels (default: {LabelColumn.get_default_csv_path()})')
    parser.add_argument('--patient_level_csv', type=str,
                       help='CSV file for patient-level labels')
    parser.add_argument('--cv_type', type=str, choices=['fixed', 'random'],
                       help='Type of cross-validation (auto-determined based on label type if not specified)')
    parser.add_argument('--score_types', type=str, nargs='+',
                       default=['roc_auc_score', 'specificity', 'sensitivity', 
                               'valeur_predictive_positive', 'valeur_predictive_negative'],
                       help='Score types to calculate')
    parser.add_argument('--n_splits', type=int, default=5,
                       help='Number of folds for random CV')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.datasets_dir).exists():
        print(f"Error: datasets_dir '{args.datasets_dir}' does not exist")
        sys.exit(1)
    
    if args.slide_level_csv and not Path(args.slide_level_csv).exists():
        print(f"Error: slide_level_csv '{args.slide_level_csv}' does not exist")
        sys.exit(1)
    
    if args.patient_level_csv and not Path(args.patient_level_csv).exists():
        print(f"Error: patient_level_csv '{args.patient_level_csv}' does not exist")
        sys.exit(1)
    
    if not args.slide_level_csv and not args.patient_level_csv:
        print("Error: At least one of slide_level_csv or patient_level_csv must be provided")
        sys.exit(1)

    # Determine cv_type automatically based on label type if not specified
    cv_type = args.cv_type
    if cv_type is None:
        cv_type = 'fixed' if LabelColumn.uses_fixed_split(args.label_name) else 'random'
        print(f"Auto-determined CV type '{cv_type}' based on label '{args.label_name}'")

    # Run experiments
    output_path = run_cv_experiment(
        datasets_dir=args.datasets_dir,
        label_name=args.label_name,
        output_dir=args.output_dir,
        slide_level_csv=args.slide_level_csv,
        patient_level_csv=args.patient_level_csv,
        cv_type=cv_type,
        score_types=args.score_types,
        n_splits=args.n_splits
    )
    
    print(f"\nAll experiments completed. Results saved to: {output_path}")


if __name__ == '__main__':
    main()
