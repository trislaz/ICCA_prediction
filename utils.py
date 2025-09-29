"""
Utility functions for data handling and model management.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple
from enum import Enum


class LabelColumn(Enum):
    """Enumeration of available label columns in the CSV files."""
    LABEL = "HepaticStem_like"
    DESERT_LIKE = "Desert_like"
    IMMUNE_CLASSICAL = "Immune_classical"
    INFLAMMATORY_STROMA = "Inflammatory_stroma"
    TUMOR_CLASSICAL = "Tumor_classical"

    @classmethod
    def get_all_labels(cls):
        """Get all available label column names."""
        return [item.value for item in cls]

    @classmethod
    def uses_fixed_split(cls, label_name: str) -> bool:
        """Check if a label uses fixed split (only 'label' does, others use on-the-fly)."""
        return label_name == cls.LABEL.value

    @classmethod
    def get_default_csv_path(cls) -> str:
        """Get default CSV path for cross-validation."""
        return "csv_labels/heptastem_testsplit.csv"


def load_and_merge_results(results_dir: Path, pattern: str = "*.csv") -> pd.DataFrame:
    """
    Load and merge multiple CSV result files.
    
    Args:
        results_dir: Directory containing result files
        pattern: File pattern to match
        
    Returns:
        Combined DataFrame
    """
    results_dir = Path(results_dir)
    csv_files = list(results_dir.glob(pattern))
    
    if not csv_files:
        print(f"No files found matching pattern '{pattern}' in {results_dir}")
        return pd.DataFrame()
    
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df['source_file'] = csv_file.name
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def save_model_ensemble(classifiers: List, normalizer, metadata: Dict, save_path: Path):
    """
    Save an ensemble of classifiers with metadata.
    
    Args:
        classifiers: List of trained sklearn classifiers
        normalizer: Fitted normalizer
        metadata: Dictionary with experiment metadata
        save_path: Path to save the model
    """
    model_data = {
        'classifiers': classifiers,
        'normalizer': normalizer,
        'metadata': metadata,
        'n_classifiers': len(classifiers),
        'model_type': 'ensemble' if len(classifiers) > 1 else 'single'
    }
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_data, save_path)
    print(f"Model saved to: {save_path}")


def load_model_ensemble(model_path: Path) -> Tuple[List, Any, Dict]:
    """
    Load an ensemble of classifiers with metadata.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Tuple of (classifiers, normalizer, metadata)
    """
    model_data = joblib.load(model_path)
    
    if isinstance(model_data, dict):
        return (
            model_data.get('classifiers', []),
            model_data.get('normalizer'),
            model_data.get('metadata', {})
        )
    else:
        # Handle legacy format (just classifiers)
        return model_data, None, {}


def create_summary_table(results_df: pd.DataFrame, 
                        group_cols: List[str], 
                        metric_cols: List[str]) -> pd.DataFrame:
    """
    Create a summary table with mean and std for metrics.
    
    Args:
        results_df: DataFrame with results
        group_cols: Columns to group by
        metric_cols: Metric columns to summarize
        
    Returns:
        Summary DataFrame
    """
    summary_rows = []
    
    for group_vals, group_data in results_df.groupby(group_cols):
        row = dict(zip(group_cols, group_vals))
        row['n_samples'] = len(group_data)
        
        for metric in metric_cols:
            if metric in group_data.columns:
                row[f'{metric}_mean'] = group_data[metric].mean()
                row[f'{metric}_std'] = group_data[metric].std()
                row[f'{metric}_min'] = group_data[metric].min()
                row[f'{metric}_max'] = group_data[metric].max()
        
        summary_rows.append(row)
    
    return pd.DataFrame(summary_rows)


def export_results_to_excel(results_dict: Dict[str, pd.DataFrame], 
                          output_path: Path,
                          sheet_names: Dict[str, str] = None):
    """
    Export multiple DataFrames to Excel with different sheets.
    
    Args:
        results_dict: Dictionary mapping keys to DataFrames
        output_path: Path for output Excel file
        sheet_names: Optional mapping of keys to sheet names
    """
    if sheet_names is None:
        sheet_names = {k: k for k in results_dict.keys()}
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for key, df in results_dict.items():
            sheet_name = sheet_names.get(key, key)[:31]  # Excel sheet name limit
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"Results exported to: {output_path}")


def validate_dataset_structure(dataset_path: Path) -> bool:
    """
    Validate that a dataset directory has the required structure.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        True if valid, False otherwise
    """
    required_files = ['embeddings.npy', 'ids.npy']
    
    for file in required_files:
        if not (dataset_path / file).exists():
            print(f"Missing required file: {dataset_path / file}")
            return False
    
    # Check if arrays can be loaded and have compatible shapes
    try:
        embs = np.load(dataset_path / 'embeddings.npy')
        ids = np.load(dataset_path / 'ids.npy')
        
        if len(embs) != len(ids):
            print(f"Shape mismatch: embeddings {embs.shape}, ids {ids.shape}")
            return False
        
        print(f"Dataset valid: {len(embs)} samples, {embs.shape[1]} features")
        return True
        
    except Exception as e:
        print(f"Error loading dataset files: {e}")
        return False


def create_experiment_config(config_dict: Dict, save_path: Path):
    """
    Save experiment configuration as JSON for reproducibility.
    
    Args:
        config_dict: Configuration dictionary
        save_path: Path to save configuration
    """
    import json
    from datetime import datetime
    
    config_dict['timestamp'] = datetime.now().isoformat()
    config_dict['version'] = '0.1.0'
    
    # Convert Path objects to strings for JSON serialization
    def convert_paths(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(v) for v in obj]
        return obj
    
    config_serializable = convert_paths(config_dict)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(config_serializable, f, indent=2)
    
    print(f"Configuration saved to: {save_path}")


def get_best_models(results_df: pd.DataFrame, 
                   metric_col: str, 
                   group_cols: List[str] = None,
                   higher_is_better: bool = True) -> pd.DataFrame:
    """
    Find best models based on a metric.
    
    Args:
        results_df: DataFrame with results
        metric_col: Column name for the metric to optimize
        group_cols: Columns to group by (find best within each group)
        higher_is_better: Whether higher values are better
        
    Returns:
        DataFrame with best models
    """
    if group_cols:
        best_models = []
        for _, group_data in results_df.groupby(group_cols):
            if higher_is_better:
                best_idx = group_data[metric_col].idxmax()
            else:
                best_idx = group_data[metric_col].idxmin()
            best_models.append(group_data.loc[best_idx])
        return pd.DataFrame(best_models)
    else:
        if higher_is_better:
            best_idx = results_df[metric_col].idxmax()
        else:
            best_idx = results_df[metric_col].idxmin()
        return results_df.loc[[best_idx]]