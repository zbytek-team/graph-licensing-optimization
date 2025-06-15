"""Main CLI entry point."""

import click

from scripts.run_single import single
from scripts.run_compare import compare
from scripts.run_tune import tune
from scripts.run_benchmark import benchmark
from scripts.run_dynamic import dynamic
from scripts.run_analyze import analyze


@click.group()
@click.version_option(version="0.1.0", prog_name="graph-licensing-optimization")
def cli():
    """Graph Licensing Optimization CLI.

    A comprehensive tool for optimizing software license allocation
    across graph structures using various algorithms.
    """
    pass


# Add all commands to the main CLI group
cli.add_command(single)
cli.add_command(compare)
cli.add_command(tune)
cli.add_command(benchmark)
cli.add_command(dynamic)
cli.add_command(analyze)


if __name__ == "__main__":
    cli()
