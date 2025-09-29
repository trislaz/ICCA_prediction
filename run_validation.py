"""
Simplified script to run validation experiments using saved classifiers.
Tests trained models on external validation datasets.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
try:
    from .crossval import test_classifier
    from .utils import LabelColumn
except ImportError:
    # When running as standalone script from root
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from crossval import test_classifier
    from utils import LabelColumn


def parse_train_params(classifier_path):
    """Extract training parameters from classifier path."""
    # Get the filename and parse it - format should be: model_TILE_FILTER_LEVEL_SLIDETYPE_label.pkl
    filename = Path(classifier_path).name

    try:
        # Remove the 'model_' prefix and '_label.pkl' suffix
        if filename.startswith('model_'):
            filename = filename[6:]  # Remove 'model_'
        if filename.endswith('_label.pkl'):
            filename = filename[:-10]  # Remove '_label.pkl'
        elif filename.endswith('_label.npy'):
            filename = filename[:-10]  # Remove '_label.npy'

        # Split the remaining parts
        parts = filename.split('_')

        if len(parts) >= 3:
            # Expected format: TILE_FILTER_LEVEL_SLIDETYPE
            tile_filter = parts[0]
            level = parts[1]
            slidetype = parts[2]

            return {
                'slidetype_train': slidetype if slidetype in {'B', 'OP', 'OPB', 'P', 'O', 'rO', 'PB', 'OB'} else 'unknown',
                'level_train': level if level in {'SlideLevel', 'PatientLevel'} else 'unknown',
                'tile_filter_train': tile_filter if tile_filter in {'QC', 'none', 'filter', 'QC_filter'} else 'unknown'
            }

        # Fallback: try to parse from full path
        path_str = str(classifier_path)
        parts = path_str.split('/')

        slidetype = None
        level = None
        tile_filter = None

        for part in parts:
            if part in {'B', 'OP', 'OPB', 'P', 'O', 'rO', 'PB', 'OB'}:
                slidetype = part
            elif part in {'SlideLevel', 'PatientLevel'}:
                level = part
            elif part in {'QC', 'none', 'filter', 'QC_filter'}:
                tile_filter = part

        return {
            'slidetype_train': slidetype or 'unknown',
            'level_train': level or 'unknown',
            'tile_filter_train': tile_filter or 'unknown'
        }

    except Exception:
        return {
            'slidetype_train': 'unknown',
            'level_train': 'unknown',
            'tile_filter_train': 'unknown'
        }


def parse_valid_params(dataset_path):
    """Extract validation parameters from dataset path."""
    path_str = str(dataset_path)
    parts = path_str.split('/')
    
    try:
        # Extract validation dataset name and tile filter
        dataset_valid = 'unknown'
        tile_filter_valid = 'unknown'
        
        # Look for known validation datasets
        for part in parts:
            if part in {'TCGA', 'mondor', 'TCGA_frozen', 'mondor_2'}:
                dataset_valid = part
            elif part in {'QC', 'none', 'filter', 'QC_filter', 'noQC'}:
                tile_filter_valid = part
        
        return {
            'dataset_valid': dataset_valid,
            'tile_filter_valid': tile_filter_valid
        }
    except Exception:
        return {
            'dataset_valid': 'unknown',
            'tile_filter_valid': 'unknown'
        }


class ValidationConfig:
    """Configuration for validation experiments."""
    
    def __init__(self):
        # Default label CSV mappings for different validation datasets
        self.label_csv_map = {
            'mondor': None,
            'TCGA': None,
            'TCGA_frozen': None,
            'mondor_2': None
        }
        
        self.score_types = ['roc_auc_score', 'f1_score', 'precision_score', 'recall_score']
    
    def set_label_csv(self, dataset_name, csv_path):
        """Set label CSV for a validation dataset."""
        if dataset_name in self.label_csv_map:
            self.label_csv_map[dataset_name] = csv_path
        else:
            print(f"Warning: Unknown dataset '{dataset_name}'. Adding to config.")
            self.label_csv_map[dataset_name] = csv_path
    
    def get_label_csv(self, dataset_name):
        """Get label CSV for a validation dataset."""
        return self.label_csv_map.get(dataset_name)


def run_validation_experiment(classifiers_dir, validation_datasets_dir, output_dir,
                             config=None, label_name='label'):
    """
    Run validation experiments using saved classifiers.
    
    Args:
        classifiers_dir: Directory containing saved model files
        validation_datasets_dir: Directory containing validation datasets
        output_dir: Directory to save results
        config: ValidationConfig instance
        label_name: Name of the label column
    """
    if config is None:
        config = ValidationConfig()
    
    classifiers_dir = Path(classifiers_dir)
    validation_datasets_dir = Path(validation_datasets_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all saved classifiers
    classifier_files = list(classifiers_dir.rglob(f'*{label_name}*.pkl'))
    if not classifier_files:
        # Fallback to .npy files for backward compatibility
        classifier_files = list(classifiers_dir.rglob(f'*{label_name}*.npy'))
    
    print(f"Found {len(classifier_files)} classifiers to test")
    
    all_results = []
    predictions_data = []
    
    for classifier_path in classifier_files:
        print(f"\nTesting classifier: {classifier_path.name}")
        
        # Extract training parameters
        train_params = parse_train_params(classifier_path)
        
        # Find validation datasets
        validation_paths = list(validation_datasets_dir.rglob('dataset'))
        if not validation_paths:
            validation_paths = [p.parent for p in validation_datasets_dir.rglob('embeddings.npy')]
        
        for valid_path in validation_paths:
            # Extract validation parameters
            valid_params = parse_valid_params(valid_path)
            dataset_name = valid_params['dataset_valid']
            
            # Get corresponding label CSV
            label_csv = config.get_label_csv(dataset_name)
            if not label_csv or not Path(label_csv).exists():
                # Only print warning once per dataset (not for every classifier)
                if dataset_name not in getattr(run_validation_experiment, '_warned_datasets', set()):
                    if not hasattr(run_validation_experiment, '_warned_datasets'):
                        run_validation_experiment._warned_datasets = set()
                    run_validation_experiment._warned_datasets.add(dataset_name)
                    print(f"  Warning: No label CSV configured for {dataset_name}, skipping validation")
                continue
            
            try:
                # Run validation
                print(f"  Validating on: {dataset_name} ({valid_params['tile_filter_valid']})")
                
                scores_df, predictions_df = test_classifier(
                    dataset_path=valid_path,
                    label_csv=label_csv,
                    classifier_path=classifier_path,
                    score_types=config.score_types,
                    label_name=label_name
                )
                
                # Add parameter information
                combined_params = {**train_params, **valid_params}
                combined_params['classifier_name'] = classifier_path.name
                combined_params['validation_dataset'] = str(valid_path)
                
                for param, value in combined_params.items():
                    scores_df[param] = value
                
                all_results.append(scores_df)
                
                # Store predictions with metadata
                predictions_df_with_params = predictions_df.copy()
                for param, value in combined_params.items():
                    predictions_df_with_params[param] = value
                predictions_data.append(predictions_df_with_params)
                
                # Save individual results
                result_name = f"{train_params['slidetype_train']}_{train_params['level_train']}_{train_params['tile_filter_train']}_on_{dataset_name}_{valid_params['tile_filter_valid']}"
                
                individual_dir = output_dir / "individual_results"
                individual_dir.mkdir(exist_ok=True)
                scores_df.to_csv(individual_dir / f"scores_{result_name}.csv", index=False)
                predictions_df_with_params.to_csv(individual_dir / f"predictions_{result_name}.csv", index=False)
                
            except Exception as e:
                print(f"Error testing {classifier_path.name} on {dataset_name}: {e}")
                continue
    
    # Save combined results
    if all_results:
        combined_scores = pd.concat(all_results, ignore_index=True)
        combined_scores.to_csv(output_dir / f"validation_summary_{label_name}.csv", index=False)
        
        combined_predictions = pd.concat(predictions_data, ignore_index=True)
        combined_predictions.to_csv(output_dir / f"validation_predictions_{label_name}.csv", index=False)
        
        print(f"\nCombined results saved:")
        print(f"  Scores: {output_dir / f'validation_summary_{label_name}.csv'}")
        print(f"  Predictions: {output_dir / f'validation_predictions_{label_name}.csv'}")
        
        # Create summary statistics
        create_validation_summary(combined_scores, output_dir, label_name)
    
    return output_dir


def create_validation_summary(results_df, output_dir, label_name):
    """Create summary statistics for validation results."""
    numeric_cols = results_df.select_dtypes(include=['float64', 'int64']).columns
    
    # Group by training and validation parameters
    groupby_cols = ['slidetype_train', 'level_train', 'tile_filter_train', 
                   'dataset_valid', 'tile_filter_valid']
    
    summary_stats = []
    for group_name, group_data in results_df.groupby(groupby_cols):
        stats = dict(zip(groupby_cols, group_name))
        stats['n_experiments'] = len(group_data)
        
        for col in numeric_cols:
            if col in group_data.columns:
                stats[f'{col}_mean'] = group_data[col].mean()
                stats[f'{col}_std'] = group_data[col].std()
        
        summary_stats.append(stats)
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / f"validation_summary_stats_{label_name}.csv", index=False)


def main():
    """Main entry point for validation experiments."""
    parser = argparse.ArgumentParser(description='Run validation experiments')
    parser.add_argument('--classifiers_dir', type=str, required=True,
                       help='Directory containing saved classifiers')
    parser.add_argument('--validation_datasets_dir', type=str, required=True,
                       help='Directory containing validation datasets')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    parser.add_argument('--label_name', type=str,
                       choices=LabelColumn.get_all_labels(),
                       default=LabelColumn.LABEL.value,
                       help=f'Name of the label column. Options: {LabelColumn.get_all_labels()}')

    # Label CSV configurations
    parser.add_argument('--mondor_csv', type=str,
                       default='csv_labels/validation_mondor.csv',
                       help='Label CSV for mondor validation dataset (default: csv_labels/validation_mondor.csv)')
    parser.add_argument('--tcga_csv', type=str,
                       default='csv_labels/TCGA_chol_final_transcripto_hemstem.csv',
                       help='Label CSV for TCGA validation dataset (default: csv_labels/TCGA_chol_final_transcripto_hemstem.csv)')
    parser.add_argument('--tcga_frozen_csv', type=str,
                       help='Label CSV for TCGA_frozen validation dataset')
    parser.add_argument('--mondor_2_csv', type=str,
                       help='Label CSV for mondor_2 validation dataset')
    
    parser.add_argument('--score_types', type=str, nargs='+',
                       default=['roc_auc_score', 'f1_score', 'precision_score', 'recall_score'],
                       help='Score types to calculate')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.classifiers_dir).exists():
        print(f"Error: classifiers_dir '{args.classifiers_dir}' does not exist")
        sys.exit(1)
    
    if not Path(args.validation_datasets_dir).exists():
        print(f"Error: validation_datasets_dir '{args.validation_datasets_dir}' does not exist")
        sys.exit(1)
    
    # Set up configuration
    config = ValidationConfig()
    config.score_types = args.score_types
    
    if args.mondor_csv:
        config.set_label_csv('mondor', args.mondor_csv)
    if args.tcga_csv:
        config.set_label_csv('TCGA', args.tcga_csv)
    if args.tcga_frozen_csv:
        config.set_label_csv('TCGA_frozen', args.tcga_frozen_csv)
    if args.mondor_2_csv:
        config.set_label_csv('mondor_2', args.mondor_2_csv)
    
    # Run validation experiments
    output_path = run_validation_experiment(
        classifiers_dir=args.classifiers_dir,
        validation_datasets_dir=args.validation_datasets_dir,
        output_dir=args.output_dir,
        config=config,
        label_name=args.label_name
    )
    
    print(f"\nAll validation experiments completed. Results saved to: {output_path}")


if __name__ == '__main__':
    main()
