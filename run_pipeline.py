"""
Master script to run the full GigaSSL reproduction pipeline: CV + Validation.
Executes cross-validation experiments followed by validation on external datasets.
"""

import argparse
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

try:
    from .run_cv import run_cv_experiment
    from .run_validation import run_validation_experiment, ValidationConfig
    from .utils import create_experiment_config, export_results_to_excel, load_and_merge_results, LabelColumn
except ImportError:
    # When running as standalone script from root
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from run_cv import run_cv_experiment
    from run_validation import run_validation_experiment, ValidationConfig
    from utils import create_experiment_config, export_results_to_excel, load_and_merge_results, LabelColumn


def run_full_pipeline(config):
    """
    Run the complete pipeline: cross-validation followed by validation.
    
    Args:
        config: Configuration dictionary with all parameters
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create main output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration for reproducibility
    config_path = output_dir / f"pipeline_config_{timestamp}.json"
    create_experiment_config(config, config_path)

    # Display configuration
    config_table = Table(show_header=False, box=None)
    config_table.add_row("Output directory", str(output_dir))
    config_table.add_row("Label", config['label_name'])
    config_table.add_row("CV type", config['cv_type'])
    config_table.add_row("Score types", ', '.join(config['score_types']))
    config_table.add_row("Timestamp", timestamp)

    console.print(Panel(config_table, title="GigaSSL Reproduction Pipeline", border_style="blue"))
    
    # Step 1: Cross-Validation
    console.print("\n[bold blue]STEP 1: Cross-Validation Experiments[/bold blue]")

    cv_output_dir = output_dir / "cv_results"

    try:
        cv_results_path = run_cv_experiment(
            datasets_dir=config['datasets_dir'],
            label_name=config['label_name'],
            output_dir=cv_output_dir,
            slide_level_csv=config.get('slide_level_csv'),
            patient_level_csv=config.get('patient_level_csv'),
            cv_type=config['cv_type'],
            score_types=config['score_types'],
            n_splits=config['n_splits']
        )

        cv_success = True
        console.print(f"[green]Cross-validation completed[/green]")
        console.print(f"Results: {cv_results_path}")

    except Exception as e:
        console.print(f"[red]Cross-validation failed: {e}[/red]")
        cv_success = False
        return False
    
    # Step 2: Validation (only if CV succeeded)
    if cv_success and config.get('run_validation', True):
        console.print("\n[bold blue]STEP 2: Validation Experiments[/bold blue]")

        # Set up validation configuration
        val_config = ValidationConfig()
        val_config.score_types = config['validation_score_types']

        # Configure label CSVs for validation datasets
        validation_csvs = {
            'mondor': config.get('mondor_csv'),
            'TCGA': config.get('tcga_csv'),
            'TCGA_frozen': config.get('tcga_frozen_csv'),
            'mondor_2': config.get('mondor_2_csv')
        }

        for dataset_name, csv_path in validation_csvs.items():
            if csv_path:
                val_config.set_label_csv(dataset_name, csv_path)

        validation_output_dir = output_dir / "validation_results"

        try:
            validation_results_path = run_validation_experiment(
                classifiers_dir=cv_output_dir / "individual_results",
                validation_datasets_dir=config['validation_datasets_dir'],
                output_dir=validation_output_dir,
                config=val_config,
                label_name=config['label_name']
            )

            console.print(f"[green]Validation completed[/green]")
            console.print(f"Results: {validation_results_path}")

        except Exception as e:
            console.print(f"[red]Validation failed: {e}[/red]")
            console.print("Cross-validation results are still available.")
    
    # Step 3: Generate Summary Report
    console.print("\n[bold blue]STEP 3: Summary Report[/bold blue]")

    try:
        generate_pipeline_summary(output_dir, config)
        console.print("[green]Summary report generated[/green]")

    except Exception as e:
        console.print(f"[yellow]Warning: Summary generation failed: {e}[/yellow]")

    # Step 4: Generate Paper Items
    console.print("\n[bold blue]STEP 4: Paper Items[/bold blue]")

    try:
        run_paper_items(output_dir, config)
        console.print("[green]Paper items generated[/green]")

    except Exception as e:
        console.print(f"[yellow]Warning: Paper items generation failed: {e}[/yellow]")

    # Final summary
    total_time = time.time() - start_time

    summary_table = Table(show_header=False, box=None)
    summary_table.add_row("Execution time", f"{total_time:.1f}s ({total_time/60:.1f}m)")
    summary_table.add_row("Results location", str(output_dir))
    summary_table.add_row("Configuration", str(config_path))

    console.print(Panel(summary_table, title="Pipeline Completed", border_style="green"))

    # Show key output files
    console.print("\n[bold]Key output files:[/bold]")
    key_files = [
        output_dir / "cv_results" / f"cv_summary_{config['label_name']}.csv",
        output_dir / "validation_results" / f"validation_summary_{config['label_name']}.csv",
        output_dir / f"pipeline_summary_{config['label_name']}.xlsx",
    ]

    for file_path in key_files:
        if file_path.exists():
            console.print(f"  [green]✓[/green] {file_path}")
        else:
            console.print(f"  [yellow]![/yellow] {file_path} (not generated)")

    return True


def generate_pipeline_summary(output_dir, config):
    """Generate a comprehensive summary of pipeline results."""
    output_dir = Path(output_dir)
    label_name = config['label_name']
    
    # Collect all results
    results_dict = {}
    
    # CV Results
    cv_summary_path = output_dir / "cv_results" / f"cv_summary_{label_name}.csv"
    if cv_summary_path.exists():
        results_dict['CV_Results'] = load_and_merge_results(
            cv_summary_path.parent, f"cv_summary_{label_name}.csv"
        )
    
    cv_stats_path = output_dir / "cv_results" / f"cv_summary_stats_{label_name}.csv"
    if cv_stats_path.exists():
        results_dict['CV_Statistics'] = load_and_merge_results(
            cv_stats_path.parent, f"cv_summary_stats_{label_name}.csv"
        )
    
    # Validation Results
    val_summary_path = output_dir / "validation_results" / f"validation_summary_{label_name}.csv"
    if val_summary_path.exists():
        results_dict['Validation_Results'] = load_and_merge_results(
            val_summary_path.parent, f"validation_summary_{label_name}.csv"
        )
    
    val_stats_path = output_dir / "validation_results" / f"validation_summary_stats_{label_name}.csv"
    if val_stats_path.exists():
        results_dict['Validation_Statistics'] = load_and_merge_results(
            val_stats_path.parent, f"validation_summary_stats_{label_name}.csv"
        )
    
    # Export to Excel if we have results
    if results_dict:
        excel_path = output_dir / f"pipeline_summary_{label_name}.xlsx"
        export_results_to_excel(results_dict, excel_path)
        
        # Also create a simple text summary
        summary_text = create_text_summary(results_dict, config)
        text_summary_path = output_dir / f"pipeline_summary_{label_name}.txt"
        with open(text_summary_path, 'w') as f:
            f.write(summary_text)


def create_text_summary(results_dict, config):
    """Create a text summary of the pipeline results."""
    summary_lines = []
    summary_lines.append("GIGASSL REPRODUCTION PIPELINE SUMMARY")
    summary_lines.append("=" * 50)
    summary_lines.append(f"Experiment: {config['label_name']}")
    summary_lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"CV Type: {config['cv_type']}")
    summary_lines.append("")
    
    # CV Summary
    if 'CV_Statistics' in results_dict:
        cv_stats = results_dict['CV_Statistics']
        summary_lines.append("CROSS-VALIDATION RESULTS")
        summary_lines.append("-" * 30)
        if not cv_stats.empty and 'roc_auc_score_mean' in cv_stats.columns:
            best_auc = cv_stats['roc_auc_score_mean'].max()
            summary_lines.append(f"Best CV AUC: {best_auc:.3f}")
            summary_lines.append(f"Number of parameter combinations: {len(cv_stats)}")
        summary_lines.append("")
    
    # Validation Summary  
    if 'Validation_Statistics' in results_dict:
        val_stats = results_dict['Validation_Statistics']
        summary_lines.append("VALIDATION RESULTS")
        summary_lines.append("-" * 20)
        if not val_stats.empty:
            summary_lines.append(f"Number of validation experiments: {len(val_stats)}")
            if 'roc_auc_score_mean' in val_stats.columns:
                best_val_auc = val_stats['roc_auc_score_mean'].max()
                summary_lines.append(f"Best validation AUC: {best_val_auc:.3f}")
        summary_lines.append("")
    
    summary_lines.append("Files generated:")
    for name, df in results_dict.items():
        summary_lines.append(f"- {name}: {len(df)} rows")
    
    return "\n".join(summary_lines)


def run_paper_items(output_dir, config):
    """Run paper items scripts for the given label."""
    output_dir = Path(output_dir)
    label_name = config['label_name']

    # Get the script directory
    current_dir = Path(__file__).parent
    paper_scripts_dir = current_dir / "paper_items_scripts"

    if not paper_scripts_dir.exists():
        raise Exception(f"Paper items scripts directory not found: {paper_scripts_dir}")

    # Create paper items output directory
    paper_output_dir = output_dir / "paper_items"
    paper_output_dir.mkdir(parents=True, exist_ok=True)

    # Base path is the parent of the output directory (where all label directories are)
    base_path = output_dir.parent

    # List of scripts to run with their arguments
    scripts_to_run = [
        {
            'script': 'extract_main_table.py',
            'args': ['--base_path', str(base_path), '--label', label_name, '--output_dir', str(paper_output_dir)],
            'description': 'Extract main table'
        },
        {
            'script': 'extract_validation_results.py',
            'args': ['--base_path', str(base_path), '--label', label_name, '--slidetype', 'OPB', '--output_dir', str(paper_output_dir)],
            'description': 'Extract validation results (OPB)'
        },
        {
            'script': 'plot_figure3.py',
            'args': ['--base_path', str(base_path), '--label', label_name, '--dataset', 'TCGA', '--output_dir', str(paper_output_dir)],
            'description': 'Generate Figure 3 (TCGA)'
        },
        {
            'script': 'plot_figure3.py',
            'args': ['--base_path', str(base_path), '--label', label_name, '--dataset', 'mondor', '--output_dir', str(paper_output_dir)],
            'description': 'Generate Figure 3 (mondor)'
        }
    ]

    success_count = 0
    total_count = len(scripts_to_run)

    for script_info in scripts_to_run:
        script_path = paper_scripts_dir / script_info['script']
        if not script_path.exists():
            console.print(f"  [yellow]![/yellow] Script not found: {script_path}")
            continue

        try:
            cmd = [sys.executable, str(script_path)] + script_info['args']
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            if result.returncode == 0:
                console.print(f"  [green]✓[/green] {script_info['description']}")
                success_count += 1
            else:
                console.print(f"  [red]✗[/red] {script_info['description']}")
                if result.stderr:
                    console.print(f"    Error: {result.stderr.strip()[:200]}")

        except subprocess.TimeoutExpired:
            console.print(f"  [red]✗[/red] {script_info['description']} (timeout)")
        except Exception as e:
            console.print(f"  [red]✗[/red] {script_info['description']}: {e}")

    console.print(f"\nCompleted {success_count}/{total_count} paper items scripts")

    # List generated files
    generated_files = list(paper_output_dir.glob("*.csv")) + list(paper_output_dir.glob("*.png")) + list(paper_output_dir.glob("*.pdf"))
    if generated_files:
        console.print(f"Generated {len(generated_files)} files in {paper_output_dir}")


def main():
    """Main entry point for the full pipeline."""
    parser = argparse.ArgumentParser(description='Run complete GigaSSL reproduction pipeline')
    
    # Required arguments
    parser.add_argument('--datasets_dir', type=str, default="./assets/training",
                       help='Directory containing training datasets')
    parser.add_argument('--label_name', type=str,
                       choices=LabelColumn.get_all_labels(),
                       default=LabelColumn.LABEL.value,
                       help=f'Name of the label column. Options: {LabelColumn.get_all_labels()}')
    parser.add_argument('--output_dir', type=str, default="./outputs",
                       help='Main output directory for all results')
    
    # CV configuration
    parser.add_argument('--slide_level_csv', type=str,
                       default=LabelColumn.get_default_csv_path(),
                       help=f'CSV file for slide-level labels (default: {LabelColumn.get_default_csv_path()})')
    parser.add_argument('--patient_level_csv', type=str,
                       help='CSV file for patient-level labels')
    parser.add_argument('--cv_type', type=str, choices=['fixed', 'random'],
                       help='Type of cross-validation (auto-determined based on label type if not specified)')
    parser.add_argument('--n_splits', type=int, default=5,
                       help='Number of folds for random CV')
    parser.add_argument('--score_types', type=str, nargs='+',
                       default=['roc_auc_score', 'specificity', 'sensitivity', 
                               'valeur_predictive_positive', 'valeur_predictive_negative'],
                       help='Score types for CV')
    
    # Validation configuration
    parser.add_argument('--validation_datasets_dir', type=str, default="./assets/validation",
                       help='Directory containing validation datasets')
    parser.add_argument('--validation_score_types', type=str, nargs='+',
                       default=['roc_auc_score', 'f1_score', 'precision_score', 'recall_score'],
                       help='Score types for validation')
    parser.add_argument('--skip_validation', action='store_true',
                       help='Skip validation step (only run CV)')
    
    # Validation dataset CSVs
    parser.add_argument('--mondor_csv', type=str,
                       default='csv_labels/validation_mondor.csv',
                       help='Mondor validation labels (default: csv_labels/validation_mondor.csv)')
    parser.add_argument('--tcga_csv', type=str,
                       default='csv_labels/TCGA_chol_final_transcripto_hemstem.csv',
                       help='TCGA validation labels (default: csv_labels/TCGA_chol_final_transcripto_hemstem.csv)')
    parser.add_argument('--tcga_frozen_csv', type=str, help='TCGA frozen validation labels')
    parser.add_argument('--mondor_2_csv', type=str, help='Mondor 2 validation labels')
    
    args = parser.parse_args()
    
    # Validate required paths
    if not Path(args.datasets_dir).exists():
        console.print(f"[red]Error: datasets_dir '{args.datasets_dir}' does not exist[/red]")
        sys.exit(1)

    if not args.slide_level_csv and not args.patient_level_csv:
        console.print("[red]Error: At least one of slide_level_csv or patient_level_csv must be provided[/red]")
        sys.exit(1)

    # Validation is optional but if requested, validation_datasets_dir must exist
    run_validation = not args.skip_validation and args.validation_datasets_dir
    if run_validation and not Path(args.validation_datasets_dir).exists():
        console.print(f"[red]Error: validation_datasets_dir '{args.validation_datasets_dir}' does not exist[/red]")
        sys.exit(1)

    # Determine cv_type automatically based on label type if not specified
    cv_type = args.cv_type
    if cv_type is None:
        cv_type = 'fixed' if LabelColumn.uses_fixed_split(args.label_name) else 'random'
        console.print(f"CV type '{cv_type}' auto-determined for label '{args.label_name}'")

    # Create label-specific output directory
    label_output_dir = Path(args.output_dir) / args.label_name

    # Create configuration dictionary
    config = {
        'datasets_dir': args.datasets_dir,
        'label_name': args.label_name,
        'output_dir': str(label_output_dir),
        'slide_level_csv': args.slide_level_csv,
        'patient_level_csv': args.patient_level_csv,
        'cv_type': cv_type,
        'n_splits': args.n_splits,
        'score_types': args.score_types,
        'run_validation': run_validation,
        'validation_datasets_dir': args.validation_datasets_dir,
        'validation_score_types': args.validation_score_types,
        'mondor_csv': args.mondor_csv,
        'tcga_csv': args.tcga_csv,
        'tcga_frozen_csv': args.tcga_frozen_csv,
        'mondor_2_csv': args.mondor_2_csv,
    }
    
    # Run the pipeline
    success = run_full_pipeline(config)
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
