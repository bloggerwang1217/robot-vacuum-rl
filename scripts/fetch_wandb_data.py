"""
Fetch training data from WandB and save to CSV files

This script downloads all training metrics from WandB runs,
including metrics not available in training.log files.
"""

import wandb
import pandas as pd
from pathlib import Path
import argparse


def fetch_wandb_runs(entity: str, project: str, output_dir: str = "wandb_data"):
    """
    Fetch all runs from a WandB project and save metrics to CSV files

    Args:
        entity: WandB entity (username or team)
        project: WandB project name
        output_dir: Directory to save CSV files
    """
    # Initialize WandB API
    api = wandb.Api()

    # Get all runs from the project
    runs = api.runs(f"{entity}/{project}")

    print(f"Found {len(runs)} runs in {entity}/{project}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Process each run
    for run in runs:
        run_name = run.name
        run_id = run.id

        print(f"\nProcessing run: {run_name} ({run_id})")

        # Get run history (all logged metrics over time)
        history = run.history()

        if history.empty:
            print(f"  No data for run {run_name}")
            continue

        # Save to CSV
        csv_path = output_path / f"{run_name}.csv"
        history.to_csv(csv_path, index=False)
        print(f"  Saved {len(history)} rows to {csv_path}")

        # Print available metrics
        print(f"  Available metrics: {', '.join(history.columns.tolist())}")

    print(f"\nAll data saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch training data from WandB"
    )

    parser.add_argument("--entity", type=str,
                       default="lazyhao-national-taiwan-university",
                       help="WandB entity (username or team)")
    parser.add_argument("--project", type=str,
                       default="robot-vacuum-rl",
                       help="WandB project name")
    parser.add_argument("--output", type=str,
                       default="wandb_data",
                       help="Output directory for CSV files")

    args = parser.parse_args()

    fetch_wandb_runs(args.entity, args.project, args.output)


if __name__ == "__main__":
    main()
