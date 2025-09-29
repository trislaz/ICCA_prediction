#!/usr/bin/env python3
"""
Analysis of binary classification behavior across slides.

This script investigates:
1. How many times each slide is misclassified across all binary classifiers
2. Slides that are never predicted as positive in any binary classification
3. Misclassification matrix: Among False Positives, what are their true labels

Uses: OPB slidetype, filter for both train and validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Configuration
LABEL_NAMES = ['Desert_like', 'Immune_classical', 'Inflammatory_stroma', 'Tumor_classical', "HepaticStem_like"]
DATASETS = ['TCGA', 'mondor']
PREDICTION_THRESHOLD = 0.5

# Paths
BASE_PATH = Path('outputs')


def load_predictions(label_name: str, dataset: str) -> pd.DataFrame:
    """Load predictions for a specific label and dataset."""
    file_path = (BASE_PATH / label_name / 'validation_results' / 'individual_results' /
                 f'predictions_OPB_SlideLevel_filter_on_{dataset}_filter.csv')

    if not file_path.exists():
        console.print(f"[yellow]Warning: File not found: {file_path}[/yellow]")
        return None

    df = pd.read_csv(file_path)
    df['binary_class'] = label_name
    df['predicted'] = (df['proba_1'] > PREDICTION_THRESHOLD).astype(int)
    return df[['ID', 'label', 'proba_0', 'proba_1', 'predicted', 'binary_class', 'dataset_valid']]


def aggregate_all_predictions() -> pd.DataFrame:
    """Aggregate predictions from all binary classifiers."""
    all_data = []

    for dataset in DATASETS:
        for label_name in LABEL_NAMES:
            df = load_predictions(label_name, dataset)
            if df is not None:
                all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)
    return combined


def get_true_label_per_slide(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each slide, get its single ground truth label.
    The true label is the binary_class where label=1.
    """
    true_labels = df[df['label'] == 1][['ID', 'binary_class', 'dataset_valid']].copy()
    true_labels.columns = ['ID', 'true_label', 'dataset_valid']

    # Check for slides with no positive label or multiple positive labels
    label_counts = df[df['label'] == 1].groupby('ID').size()

    if len(label_counts[label_counts > 1]) > 0:
        console.print(f"[yellow]Warning: {len(label_counts[label_counts > 1])} slides have multiple true labels[/yellow]")

    # Get all slides to identify those with no positive labels
    all_slides = df[['ID', 'dataset_valid']].drop_duplicates()
    true_labels = all_slides.merge(true_labels, on=['ID', 'dataset_valid'], how='left')

    return true_labels


def count_misclassifications_per_slide(df: pd.DataFrame, dataset: str = None) -> pd.DataFrame:
    """
    For each slide, count how many binary classifiers misclassified it.

    Returns: DataFrame with columns [ID, true_label, n_misclassifications, n_classifiers, misclass_rate]
    """
    if dataset is not None:
        df = df[df['dataset_valid'] == dataset].copy()

    # Get true label for each slide
    true_labels = get_true_label_per_slide(df)

    # For each slide, count misclassifications
    results = []

    for slide_id in df['ID'].unique():
        slide_data = df[df['ID'] == slide_id]
        true_label = true_labels[true_labels['ID'] == slide_id]['true_label'].values
        true_label = true_label[0] if len(true_label) > 0 and pd.notna(true_label[0]) else None
        dataset_val = slide_data['dataset_valid'].iloc[0]

        n_classifiers = len(slide_data)
        n_errors = 0

        for _, row in slide_data.iterrows():
            # Error if: (1) true class but predicted 0, or (2) not true class but predicted 1
            if row['binary_class'] == true_label:
                # This is the true class, should predict 1
                if row['predicted'] == 0:
                    n_errors += 1
            else:
                # This is not the true class, should predict 0
                if row['predicted'] == 1:
                    n_errors += 1

        results.append({
            'ID': slide_id,
            'true_label': true_label,
            'dataset': dataset_val,
            'n_misclassifications': n_errors,
            'n_classifiers': n_classifiers,
            'misclass_rate': n_errors / n_classifiers if n_classifiers > 0 else 0
        })

    return pd.DataFrame(results)


def find_never_positive_slides(df: pd.DataFrame, dataset: str = None) -> pd.DataFrame:
    """Find slides that are never predicted as positive in any binary classification."""
    if dataset is not None:
        df = df[df['dataset_valid'] == dataset].copy()

    # Slides with at least one positive prediction
    slides_with_positive = df[df['predicted'] == 1]['ID'].unique()

    # All slides
    all_slides = df['ID'].unique()

    # Never positive
    never_positive = set(all_slides) - set(slides_with_positive)

    # Get their true labels
    true_labels = get_true_label_per_slide(df)
    never_positive_df = true_labels[true_labels['ID'].isin(never_positive)].copy()

    return never_positive_df


def build_fp_misclassification_matrix(df: pd.DataFrame, dataset: str = None) -> pd.DataFrame:
    """
    Build misclassification matrix: Among False Positives, what were their true labels?

    Rows: True label
    Columns: What was falsely predicted as positive
    Values: Count of FP predictions
    """
    if dataset is not None:
        df = df[df['dataset_valid'] == dataset].copy()

    # Get true labels
    true_labels = get_true_label_per_slide(df)

    # Merge true labels with predictions
    df_with_true = df.merge(true_labels[['ID', 'true_label']], on='ID', how='left')

    # Find False Positives: predicted=1 but binary_class != true_label
    false_positives = df_with_true[
        (df_with_true['predicted'] == 1) &
        (df_with_true['binary_class'] != df_with_true['true_label'])
    ].copy()

    # Count FPs: true_label -> predicted_as (binary_class)
    fp_matrix = false_positives.groupby(['true_label', 'binary_class']).size().reset_index(name='count')

    # Pivot to matrix form
    matrix = fp_matrix.pivot(index='true_label', columns='binary_class', values='count').fillna(0).astype(int)

    # Add rows/columns for all labels
    for label in LABEL_NAMES:
        if label not in matrix.index:
            matrix.loc[label] = 0
        if label not in matrix.columns:
            matrix[label] = 0

    # Add row for slides with no true label (if any)
    if pd.isna(false_positives['true_label']).any():
        no_label_fps = false_positives[pd.isna(false_positives['true_label'])].groupby('binary_class').size()
        matrix.loc['<no_label>'] = 0
        for label, count in no_label_fps.items():
            matrix.loc['<no_label>', label] = count

    matrix = matrix[sorted(matrix.columns)]
    matrix = matrix.sort_index()

    return matrix


def analyze_per_class_errors(df: pd.DataFrame, dataset: str = None) -> pd.DataFrame:
    """
    For each binary class, compute:
    - True Positives, False Positives, True Negatives, False Negatives
    - Precision, Recall, F1, AUC
    """
    results = []

    # Filter by dataset if specified
    if dataset is not None:
        df = df[df['dataset_valid'] == dataset].copy()

    for label_name in LABEL_NAMES:
        class_df = df[df['binary_class'] == label_name].copy()

        if len(class_df) == 0:
            continue

        tp = ((class_df['label'] == 1) & (class_df['predicted'] == 1)).sum()
        fp = ((class_df['label'] == 0) & (class_df['predicted'] == 1)).sum()
        tn = ((class_df['label'] == 0) & (class_df['predicted'] == 0)).sum()
        fn = ((class_df['label'] == 1) & (class_df['predicted'] == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate AUC
        try:
            auc = roc_auc_score(class_df['label'], class_df['proba_1'])
        except ValueError:
            # In case there's only one class present
            auc = np.nan

        result = {
            'binary_class': label_name,
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'AUC': auc,
            'total_positives': tp + fn,
            'total_negatives': tn + fp
        }

        if dataset is not None:
            result['dataset'] = dataset

        results.append(result)

    return pd.DataFrame(results)


def main():
    console.print(Panel.fit(
        "[bold cyan]BINARY CLASSIFICATION ANALYSIS[/bold cyan]\n"
        "Configuration: OPB slidetype, filter train/validation",
        border_style="cyan"
    ))
    console.print()

    # Load all predictions
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Loading predictions from all binary classifiers...", total=None)
        df = aggregate_all_predictions()
        progress.update(task, completed=True)

    console.print(f"[green]✓[/green] Loaded {len(df)} prediction records from {df['ID'].nunique()} unique slides")
    console.print(f"[green]✓[/green] Binary classifiers: {', '.join(LABEL_NAMES)}")
    console.print(f"[green]✓[/green] Validation datasets: {', '.join(df['dataset_valid'].unique())}")
    console.print()

    # ========================================
    # 1. PER-CLASS PERFORMANCE METRICS
    # ========================================
    console.print(Panel("[bold]PER-CLASS PERFORMANCE METRICS[/bold]", border_style="blue"))

    # Overall metrics
    console.print("\n[bold cyan]Overall (All Datasets)[/bold cyan]")
    class_metrics_overall = analyze_per_class_errors(df)

    table = Table(show_header=True, header_style="bold magenta")
    for col in class_metrics_overall.columns:
        table.add_column(col)

    for _, row in class_metrics_overall.iterrows():
        table.add_row(*[str(val) if not isinstance(val, float) else f"{val:.3f}" for val in row])

    console.print(table)

    # Per-dataset metrics
    all_class_metrics = [class_metrics_overall]
    for dataset in sorted(df['dataset_valid'].unique()):
        console.print(f"\n[bold cyan]{dataset.upper()} Dataset[/bold cyan]")
        class_metrics_dataset = analyze_per_class_errors(df, dataset=dataset)
        all_class_metrics.append(class_metrics_dataset)

        table = Table(show_header=True, header_style="bold magenta")
        for col in class_metrics_dataset.columns:
            table.add_column(col)

        for _, row in class_metrics_dataset.iterrows():
            table.add_row(*[str(val) if not isinstance(val, float) else f"{val:.3f}" for val in row])

        console.print(table)

    console.print()

    # ========================================
    # 2. MISCLASSIFICATIONS PER SLIDE
    # ========================================
    console.print(Panel("[bold]MISCLASSIFICATIONS PER SLIDE[/bold]", border_style="blue"))

    # Overall
    console.print("\n[bold cyan]Overall (All Datasets)[/bold cyan]")
    misclass_per_slide_overall = count_misclassifications_per_slide(df)

    # Summary stats
    stats_table = Table(show_header=False, box=None)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="yellow")
    stats_table.add_row("Total slides", str(len(misclass_per_slide_overall)))
    stats_table.add_row("Mean misclassifications per slide", f"{misclass_per_slide_overall['n_misclassifications'].mean():.2f}")
    stats_table.add_row("Median misclassifications per slide", f"{misclass_per_slide_overall['n_misclassifications'].median():.0f}")
    stats_table.add_row("Perfect slides (0 errors)", str((misclass_per_slide_overall['n_misclassifications'] == 0).sum()))
    console.print(stats_table)
    console.print()

    # Top misclassified slides
    console.print("[bold]Top 20 Most Misclassified Slides:[/bold]")
    top_misclass = misclass_per_slide_overall.nlargest(20, 'n_misclassifications')

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Slide ID", style="yellow")
    table.add_column("True Label", style="magenta")
    table.add_column("Errors", style="red")
    table.add_column("Rate", style="red")

    for _, row in top_misclass.iterrows():
        true_label_str = str(row['true_label']) if pd.notna(row['true_label']) else '<no_label>'
        table.add_row(
            row['ID'][:50],  # Truncate long IDs
            true_label_str,
            str(row['n_misclassifications']),
            f"{row['misclass_rate']:.2f}"
        )

    console.print(table)

    # Per-dataset analysis
    all_misclass_per_slide = {'overall': misclass_per_slide_overall}
    for dataset in sorted(df['dataset_valid'].unique()):
        console.print(f"\n[bold cyan]{dataset.upper()} Dataset[/bold cyan]")
        misclass_per_slide_ds = count_misclassifications_per_slide(df, dataset=dataset)
        all_misclass_per_slide[dataset] = misclass_per_slide_ds

        stats_table = Table(show_header=False, box=None)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="yellow")
        stats_table.add_row("Total slides", str(len(misclass_per_slide_ds)))
        stats_table.add_row("Mean misclassifications", f"{misclass_per_slide_ds['n_misclassifications'].mean():.2f}")
        stats_table.add_row("Perfect slides", str((misclass_per_slide_ds['n_misclassifications'] == 0).sum()))
        console.print(stats_table)

    console.print()

    # ========================================
    # 3. FALSE POSITIVE MISCLASSIFICATION MATRIX
    # ========================================
    console.print(Panel("[bold]FALSE POSITIVE MISCLASSIFICATION MATRIX[/bold]", border_style="blue"))
    console.print("[dim]Rows: True label | Columns: What was falsely predicted as positive | Values: Count of FPs[/dim]")
    console.print()

    # Overall
    console.print("[bold cyan]Overall (All Datasets)[/bold cyan]")
    fp_matrix_overall = build_fp_misclassification_matrix(df)

    matrix_table = Table(show_header=True, header_style="bold cyan")
    matrix_table.add_column("True Label", style="yellow")
    for col in fp_matrix_overall.columns:
        matrix_table.add_column(col)

    for idx, row in fp_matrix_overall.iterrows():
        matrix_table.add_row(str(idx), *[str(int(val)) for val in row])

    console.print(matrix_table)
    console.print()

    # Per-dataset
    all_fp_matrices = {'overall': fp_matrix_overall}
    for dataset in sorted(df['dataset_valid'].unique()):
        console.print(f"[bold cyan]{dataset.upper()} Dataset[/bold cyan]")
        fp_matrix_ds = build_fp_misclassification_matrix(df, dataset=dataset)
        all_fp_matrices[dataset] = fp_matrix_ds

        matrix_table = Table(show_header=True, header_style="bold cyan")
        matrix_table.add_column("True Label", style="yellow")
        for col in fp_matrix_ds.columns:
            matrix_table.add_column(col)

        for idx, row in fp_matrix_ds.iterrows():
            matrix_table.add_row(str(idx), *[str(int(val)) for val in row])

        console.print(matrix_table)
        console.print()

    # ========================================
    # 4. SLIDES NEVER PREDICTED POSITIVE
    # ========================================
    console.print(Panel("[bold]SLIDES NEVER PREDICTED AS POSITIVE[/bold]", border_style="blue"))

    # Overall
    console.print("\n[bold cyan]Overall (All Datasets)[/bold cyan]")
    never_positive_overall = find_never_positive_slides(df)
    console.print(f"[green]Found {len(never_positive_overall)} slides never predicted positive[/green]")

    if len(never_positive_overall) > 0:
        # Distribution of true labels
        true_label_dist = never_positive_overall['true_label'].value_counts().reset_index()
        true_label_dist.columns = ['True Label', 'Count']

        dist_table = Table(show_header=True, header_style="bold cyan")
        dist_table.add_column("True Label", style="yellow")
        dist_table.add_column("Count", style="green")

        for _, row in true_label_dist.iterrows():
            label_str = str(row['True Label']) if pd.notna(row['True Label']) else '<no_label>'
            dist_table.add_row(label_str, str(row['Count']))

        console.print(dist_table)

        # Show first 10 examples
        console.print("\n[dim]First 10 examples:[/dim]")
        examples_table = Table(show_header=True, header_style="bold cyan")
        examples_table.add_column("Slide ID", style="yellow")
        examples_table.add_column("True Label", style="magenta")

        for _, row in never_positive_overall.head(10).iterrows():
            label_str = str(row['true_label']) if pd.notna(row['true_label']) else '<no_label>'
            examples_table.add_row(row['ID'][:50], label_str)

        console.print(examples_table)

    # Per-dataset
    all_never_positive = {'overall': never_positive_overall}
    for dataset in sorted(df['dataset_valid'].unique()):
        console.print(f"\n[bold cyan]{dataset.upper()} Dataset[/bold cyan]")
        never_positive_ds = find_never_positive_slides(df, dataset=dataset)
        all_never_positive[dataset] = never_positive_ds
        console.print(f"[green]Found {len(never_positive_ds)} slides never predicted positive[/green]")

        if len(never_positive_ds) > 0:
            true_label_dist = never_positive_ds['true_label'].value_counts().reset_index()
            true_label_dist.columns = ['True Label', 'Count']

            dist_table = Table(show_header=True, header_style="bold cyan", box=None)
            dist_table.add_column("True Label", style="yellow")
            dist_table.add_column("Count", style="green")

            for _, row in true_label_dist.iterrows():
                label_str = str(row['True Label']) if pd.notna(row['True Label']) else '<no_label>'
                dist_table.add_row(label_str, str(row['Count']))

            console.print(dist_table)

    console.print()

    # ========================================
    # 5. SAVE RESULTS
    # ========================================
    output_dir = Path('outputs/analysis')
    output_dir.mkdir(exist_ok=True)

    # Save per-class metrics
    class_metrics_overall.to_csv(output_dir / 'per_class_metrics_overall.csv', index=False)
    combined_metrics = pd.concat(all_class_metrics, ignore_index=True)
    combined_metrics.to_csv(output_dir / 'per_class_metrics_combined.csv', index=False)

    for dataset in sorted(df['dataset_valid'].unique()):
        class_metrics_ds = analyze_per_class_errors(df, dataset=dataset)
        class_metrics_ds.to_csv(output_dir / f'per_class_metrics_{dataset}.csv', index=False)

    # Save misclassifications per slide
    misclass_per_slide_overall.to_csv(output_dir / 'misclassifications_per_slide_overall.csv', index=False)
    for dataset, misclass_df in all_misclass_per_slide.items():
        if dataset != 'overall':
            misclass_df.to_csv(output_dir / f'misclassifications_per_slide_{dataset}.csv', index=False)

    # Save FP matrices
    fp_matrix_overall.to_csv(output_dir / 'fp_misclassification_matrix_overall.csv')
    for dataset, fp_matrix in all_fp_matrices.items():
        if dataset != 'overall':
            fp_matrix.to_csv(output_dir / f'fp_misclassification_matrix_{dataset}.csv')

    # Save never-positive slides
    never_positive_overall.to_csv(output_dir / 'never_positive_slides_overall.csv', index=False)
    for dataset, never_pos_df in all_never_positive.items():
        if dataset != 'overall':
            never_pos_df.to_csv(output_dir / f'never_positive_slides_{dataset}.csv', index=False)

    console.print(Panel(
        "[bold green]Results saved to outputs/analysis/[/bold green]\n\n"
        "  [cyan]•[/cyan] per_class_metrics_*.csv: Performance metrics per binary classifier\n"
        "  [cyan]•[/cyan] misclassifications_per_slide_*.csv: Error counts per slide\n"
        "  [cyan]•[/cyan] fp_misclassification_matrix_*.csv: FP confusion patterns\n"
        "  [cyan]•[/cyan] never_positive_slides_*.csv: Slides never predicted positive",
        border_style="green"
    ))


if __name__ == '__main__':
    main()
