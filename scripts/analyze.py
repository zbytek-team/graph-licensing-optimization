"""Analysis script for benchmark results."""

import click
from datetime import datetime
from pathlib import Path

from scripts.common import (
    setup_logging,
    create_timestamped_path,
    create_metadata,
    save_results,
)


def run_analysis(input_dir: str = None) -> None:
    """Run analysis on benchmark results."""
    if not input_dir:
        input_dir = "results/benchmark"

    output_path = create_timestamped_path("results", "analysis")
    output_path.mkdir(parents=True, exist_ok=True)
    click.echo(f"Analysis results will be saved to: {output_path}")

    click.echo("Running analysis on benchmark results...")

    try:
        start_time = datetime.now()

        # Import here to avoid circular imports
        from src.graph_licensing.analysis import AnalysisRunner

        analyzer = AnalysisRunner(results_path=str(input_dir))
        analyzer.output_dir = output_path
        analyzer.generate_comprehensive_report()

        # Create metadata
        analysis_metadata = create_metadata(
            start_time,
            input_directory=str(input_dir),
            output_directory=str(output_path),
        )

        save_results(analysis_metadata, output_path, "analysis_metadata")

        click.echo(f"Analysis complete! Results saved to {output_path}")
        click.echo(f"Duration: {analysis_metadata['duration_seconds']:.2f} seconds")

    except Exception as e:
        click.echo(f"Analysis failed: {e}")


@click.command()
@click.option("--input-dir", type=click.Path(), help="Input directory with benchmark results")
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def analyze(input_dir, log_level):
    """Analyze benchmark results and generate reports."""
    setup_logging(log_level)
    run_analysis(input_dir)


if __name__ == "__main__":
    analyze()
