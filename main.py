"""Main CLI entry point."""

import click

from scripts.single import single
from scripts.compare import compare
from scripts.tune import tune  
from scripts.benchmark import benchmark
from scripts.dynamic import dynamic
from scripts.analyze import analyze


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
