#!/usr/bin/env python3
"""
Master script to generate all paper items at once.
Runs all the individual scripts and generates a comprehensive report.
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def run_script(script_path, args=None):
    """Run a Python script and capture its output."""
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def main():
    """Generate all paper items."""
    console.print(Panel("Paper Items Generation", border_style="blue"))

    scripts_dir = Path("gigassl_reproduction/paper_items_scripts")
    results = {}

    # 1. Generate Figure 3 for both TCGA and mondor datasets
    console.print("\n[bold]1. Generating Figures[/bold]")
    validation_datasets = ['TCGA', 'mondor']
    for dataset in validation_datasets:
        success, output = run_script(scripts_dir / "plot_figure3.py", [dataset])
        results[f'figure3_{dataset.lower()}'] = success
        if success:
            console.print(f"  [green]✓[/green] Figure 3 - {dataset}")
        else:
            console.print(f"  [red]✗[/red] Figure 3 - {dataset}")
            if output:
                console.print(f"    {output[:200]}")

    # 2. Extract main table
    console.print("\n[bold]2. Extracting Main Table[/bold]")
    success, output = run_script(scripts_dir / "extract_main_table.py")
    results['main_table'] = success
    if success:
        console.print("  [green]✓[/green] Main table (OPB CV)")
    else:
        console.print("  [red]✗[/red] Main table")
        if output:
            console.print(f"    {output[:200]}")

    # 3. Extract validation results for key slidetypes
    console.print("\n[bold]3. Extracting Validation Results[/bold]")
    key_slidetypes = ['OPB', 'O', 'P', 'OP']
    for slidetype in key_slidetypes:
        success, output = run_script(scripts_dir / "extract_validation_results.py", [slidetype])
        results[f'validation_{slidetype}'] = success
        if success:
            console.print(f"  [green]✓[/green] Validation - {slidetype}")
        else:
            console.print(f"  [red]✗[/red] Validation - {slidetype}")
            if output:
                console.print(f"    {output[:200]}")

    # Summary
    success_count = sum(results.values())
    total_count = len(results)

    summary_table = Table(show_header=False, box=None)
    summary_table.add_row("Success rate", f"{success_count}/{total_count}")
    summary_table.add_row("Output directory", str(scripts_dir.absolute()))

    console.print(Panel(summary_table, title="Summary", border_style="green" if success_count == total_count else "yellow"))

    # Check generated files
    console.print("\n[bold]Generated Files:[/bold]")
    output_files = []

    # Figure files
    for dataset in validation_datasets:
        output_files.extend([
            f"figure3_{dataset}.png",
            f"figure3_{dataset}.pdf"
        ])

    # Table files
    output_files.extend([
        "main_table_OPB_raw.csv",
        "main_table_OPB_formatted.csv"
    ])

    # Validation files
    for slidetype in key_slidetypes:
        output_files.extend([
            f"validation_results_{slidetype}.csv",
            f"validation_results_{slidetype}_pivot.csv"
        ])

    for file in output_files:
        file_path = scripts_dir / file
        if file_path.exists():
            console.print(f"  [green]✓[/green] {file}")
        else:
            console.print(f"  [yellow]![/yellow] {file} (missing)")

    # Create a summary report
    summary_report = f"""
Paper Items Generation Report
=============================

Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Success Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)

Generated Items:
"""

    for item, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        summary_report += f"- {item}: {status}\n"

    summary_report += f"""
Output Directory: {scripts_dir.absolute()}

Files Generated:
"""

    for file in output_files:
        file_path = scripts_dir / file
        if file_path.exists():
            size = file_path.stat().st_size
            summary_report += f"- {file} ({size} bytes)\n"

    # Save summary report
    report_path = scripts_dir / "generation_report.txt"
    with open(report_path, 'w') as f:
        f.write(summary_report)

    console.print(f"\nReport saved to: {report_path}")

    if success_count == total_count:
        return 0
    else:
        console.print(f"\n[yellow]{total_count - success_count} items failed[/yellow]")
        return 1

if __name__ == "__main__":
    sys.exit(main())