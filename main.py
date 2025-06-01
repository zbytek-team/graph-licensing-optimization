#!/usr/bin/env python3

"""Main application for graph licensing optimization.

This is the main entry point for the Graph Licensing Optimization project.
It provides a command-line interface for running various algorithms and
benchmarks on different graph types for license optimization problems.
"""

import json
import logging
import sys
from pathlib import Path

import click

from graph_licensing.algorithms.approx import DominatingSetAlgorithm, GreedyAlgorithm, RandomizedAlgorithm
from graph_licensing.algorithms.exact import ILPAlgorithm, NaiveAlgorithm
from graph_licensing.algorithms.meta import GeneticAlgorithm, SimulatedAnnealingAlgorithm, TabuSearchAlgorithm
from graph_licensing.generators.graph_generator import GraphGenerator
from graph_licensing.models.license import LicenseConfig
from graph_licensing.utils import Benchmark, FileIO
from src.graph_licensing.visualizers.graph_visualizer import GraphVisualizer


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("graph_licensing.log"),
        ],
    )


def create_algorithms() -> dict:
    """Create all available algorithms.

    Returns:
        Dictionary mapping algorithm names to instances.
    """
    return {
        "ilp": ILPAlgorithm(),
        "naive": NaiveAlgorithm(),
        "greedy": GreedyAlgorithm(),
        "dominating_set": DominatingSetAlgorithm(),
        "randomized": RandomizedAlgorithm(seed=42),
        "genetic": GeneticAlgorithm(
            population_size=200,
            generations=500,
            mutation_rate=0.1,
            crossover_rate=0.8,
            seed=42,
        ),
        "simulated_annealing": SimulatedAnnealingAlgorithm(
            initial_temp=100.0,
            final_temp=0.1,
            max_iterations=5000,
            cooling_rate=0.995,
            seed=42,
        ),
        "tabu_search": TabuSearchAlgorithm(
            max_iterations=1000,
            max_no_improvement=50,
            seed=42,
        ),
    }


# Common options that are used across multiple commands
algorithm_option = click.option(
    "--algorithm",
    type=click.Choice(list(create_algorithms().keys())),
    help="Algorithm to run",
)

algorithms_option = click.option(
    "--algorithms",
    multiple=True,
    default=["ilp", "greedy", "dominating_set", "simulated_annealing"],
    # default=["greedy", "dominating_set", "randomized", "genetic", "simulated_annealing", "tabu_search"],
    type=click.Choice(list(create_algorithms().keys())),
    help="Algorithms to run (can specify multiple)",
)

graph_type_option = click.option(
    "--graph-type",
    default="random",
    type=click.Choice(["random", "scale_free", "small_world", "complete", "grid", "star", "path", "cycle", "facebook"]),
    help="Type of graph to generate",
)

graph_types_option = click.option(
    "--graph-types",
    multiple=True,
    default=["scale_free"],
    # default=["random", "scale_free", "small_world", "complete", "grid", "star", "path", "cycle", "facebook"],
    type=click.Choice(["random", "scale_free", "small_world", "complete", "grid", "star", "path", "cycle", "facebook"]),
    help="Types of graphs to test",
)

graph_size_option = click.option(
    "--graph-size",
    default=20,
    type=int,
    help="Number of nodes in graph",
)

graph_sizes_option = click.option(
    "--graph-sizes",
    multiple=True,
    default=[256, 512, 1024],
    # default=[8, 16, 32, 64, 128, 256, 512],
    type=int,
    help="Graph sizes to test",
)

seed_option = click.option(
    "--seed",
    default=42,
    type=int,
    help="Random seed for reproducibility",
)

solo_cost_option = click.option(
    "--solo-cost",
    default=1.0,
    type=float,
    help="Cost of solo license",
)

group_cost_option = click.option(
    "--group-cost",
    default=2.08,
    type=float,
    help="Cost of group license",
)

group_size_option = click.option(
    "--group-size",
    default=6,
    type=int,
    help="Maximum group size",
)

facebook_ego_option = click.option(
    "--facebook-ego",
    default=None,
    type=str,
    help="Specific Facebook ego network ID to load (e.g., '0', '107'). If not specified, loads random ego network.",
)


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    help="Logging level",
)
@click.pass_context
def cli(ctx, log_level):
    """Graph Licensing Optimization CLI.

    This tool provides various algorithms for solving graph licensing optimization problems.

    Examples:

    \b
    # Run single algorithm test
    uv run main.py single --algorithm greedy --graph-type random --graph-size 20

    \b
    # Run on Facebook ego network
    uv run main.py single --algorithm ilp --graph-type facebook --facebook-ego 0

    \b
    # Run benchmark on multiple algorithms
    uv run main.py benchmark --algorithms greedy --algorithms ilp --algorithms genetic --graph-types random --graph-types scale_free

    \b
    # Run with custom costs
    uv run main.py single --algorithm ilp --solo-cost 10 --group-cost 15 --group-size 5
    """  # noqa: E501
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level
    setup_logging(log_level)


@cli.command()
@algorithm_option
@graph_type_option
@graph_size_option
@seed_option
@solo_cost_option
@group_cost_option
@group_size_option
@facebook_ego_option
@click.pass_context
def single(
    ctx,
    algorithm,
    graph_type,
    graph_size,
    seed,
    solo_cost,
    group_cost,
    group_size,
    facebook_ego,
):
    """Run single algorithm test."""
    if not algorithm:
        click.echo("Error: --algorithm is required")
        return

    try:
        # Create graph
        generator = GraphGenerator()
        kwargs = {}
        if graph_type == "facebook" and facebook_ego:
            kwargs["ego_id"] = facebook_ego
        graph = generator.generate_graph(graph_type, graph_size, seed=seed, **kwargs)

        # Create configuration
        config = LicenseConfig(
            solo_price=solo_cost,
            group_price=group_cost,
            group_size=group_size,
        )

        # Get algorithm
        algorithms = create_algorithms()
        if algorithm not in algorithms:
            click.echo(f"Error: Unknown algorithm '{algorithm}'. Available algorithms: {list(algorithms.keys())}")
            return

        alg_instance = algorithms[algorithm]

        # Solve
        click.echo(f"Running {algorithm} algorithm on {graph_type} graph with {graph_size} nodes...")
        solution = alg_instance.solve(graph, config)

        if solution is None:
            click.echo(f"Error: Algorithm '{algorithm}' failed to find a solution")
            return

        # Display results
        total_cost = solution.calculate_cost(config)

        click.echo(f"\n=== Solution Results ===")  # noqa: F541
        click.echo(f"Algorithm: {algorithm}")
        click.echo(f"Graph: {graph_type} with {graph_size} nodes")
        click.echo(f"Total cost: {total_cost:.2f}")
        click.echo(
            f"Solo licenses: {len(solution.solo_nodes)} (cost: {len(solution.solo_nodes) * config.solo_price:.2f})"
        )
        click.echo(
            f"Group licenses: {len(solution.group_owners)} (cost: {len(solution.group_owners) * config.group_price:.2f})"
        )

        if solution.solo_nodes:
            click.echo(f"Solo nodes: {sorted(solution.solo_nodes)}")

        if solution.group_owners:
            click.echo("Groups:")
            for owner, members in solution.group_owners.items():
                click.echo(f"  Owner {owner}: {sorted(members)} ({len(members)} members)")

        click.echo(
            f"License configuration: solo={config.solo_price}, group={config.group_price}, max_size={config.group_size}"
        )
        click.echo("========================\n")

        # Save results
        output = Path("results/single/") / f"{algorithm}_{graph_type}_{graph_size}"

        output.mkdir(parents=True, exist_ok=True)

        FileIO.save_graph(graph, output / "graph.json")

        FileIO.save_solution(solution, output / "solution.json")

        FileIO.save_config(config, output / "config.json")

        visualizer = GraphVisualizer()
        visualizer.visualize_solution(
            graph, solution, config, title=f"{algorithm} Solution", save_path=output / "solution.png"
        )

    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user")
    except Exception as e:
        click.echo(f"Error: {e}")
        logging.exception(f"Error: {e}")
        if ctx.obj["log_level"] == "DEBUG":
            raise
        sys.exit(1)


@cli.command()
@algorithms_option
@graph_types_option
@graph_sizes_option
@seed_option
@solo_cost_option
@group_cost_option
@group_size_option
@click.pass_context
def benchmark(
    ctx,
    algorithms,
    graph_types,
    graph_sizes,
    seed,
    solo_cost,
    group_cost,
    group_size,
):
    """Run comprehensive benchmark tests."""
    try:
        # Create benchmark
        benchmark_obj = Benchmark()
        all_algorithms = create_algorithms()

        # Filter algorithms if specified
        if algorithms:
            selected_algorithms = {k: v for k, v in all_algorithms.items() if k in algorithms}
            if not selected_algorithms:
                click.echo("Error: No valid algorithms selected")
                return
            all_algorithms = selected_algorithms

        # Create test graphs
        generator = GraphGenerator()
        test_graphs = {}

        for graph_type in graph_types:
            test_graphs[graph_type] = []
            for size in graph_sizes:
                graph = generator.generate_graph(graph_type, size, seed=seed)
                test_graphs[graph_type].append((f"{graph_type}_{size}", graph))

        # Create configurations
        config = LicenseConfig(
            solo_price=solo_cost,
            group_price=group_cost,
            group_size=group_size,
        )

        # Run benchmarks
        total_tests = sum(len(graphs) for graphs in test_graphs.values()) * len(all_algorithms)
        click.echo(f"Running {total_tests} benchmark tests...")

        current_test = 0
        for graph_type, graphs in test_graphs.items():
            click.echo(f"Testing graph type: {graph_type}")
            for graph_name, graph in graphs:
                click.echo(f"  Graph: {graph_name}")
                for alg_name, algorithm in all_algorithms.items():
                    current_test += 1
                    click.echo(f"    [{current_test}/{total_tests}] Running {alg_name}...")
                    try:
                        benchmark_obj.run_single_test(
                            algorithm,
                            graph,
                            config,
                            f"{graph_name}_{alg_name}",
                        )
                    except Exception as e:
                        click.echo(f"      ✗ Exception: {e}")

        # Save results
        output = Path("results/benchmark")

        output.mkdir(parents=True, exist_ok=True)

        # Save benchmark results
        benchmark_obj.save_results(output / "benchmark_results.csv")

        # Generate summary
        summary = benchmark_obj.get_summary()
        if summary:
            summary_path = output / "benchmark_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

        click.echo(f"Results saved to {output}")

        click.echo("Benchmark completed!")

    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user")
    except Exception as e:
        click.echo(f"Error: {e}")
        logging.exception(f"Error: {e}")
        if ctx.obj["log_level"] == "DEBUG":
            raise
        sys.exit(1)


@cli.command()
@algorithms_option
@graph_type_option
@graph_size_option
@seed_option
@solo_cost_option
@group_cost_option
@group_size_option
@facebook_ego_option
@click.pass_context
def compare(ctx, algorithms, graph_type, graph_size, seed, solo_cost, group_cost, group_size, facebook_ego):
    """Compare multiple algorithms."""
    try:
        # Create graph
        generator = GraphGenerator()
        kwargs = {}
        if graph_type == "facebook" and facebook_ego:
            kwargs["ego_id"] = facebook_ego
        graph = generator.generate_graph(graph_type, graph_size, seed=seed, **kwargs)

        # Create configuration
        config = LicenseConfig(
            solo_price=solo_cost,
            group_price=group_cost,
            group_size=group_size,
        )

        # Get algorithms
        all_algorithms = create_algorithms()
        if algorithms:
            all_algorithms = {k: v for k, v in all_algorithms.items() if k in algorithms}
        else:
            # Default algorithms for comparison
            all_algorithms = {
                k: v for k, v in all_algorithms.items() if k in ["greedy", "genetic", "simulated_annealing"]
            }

        solutions = {}

        # Run all algorithms
        click.echo(f"Comparing {len(all_algorithms)} algorithms on {graph_type} graph with {graph_size} nodes...")

        for name, algorithm in all_algorithms.items():
            try:
                click.echo(f"Running {name}...")
                solution = algorithm.solve(graph, config)
                if solution:
                    solutions[name] = solution
                    cost = solution.calculate_cost(config)
                    click.echo(f"  {name}: cost = {cost:.2f}")
                else:
                    click.echo(f"  {name}: failed to find solution")
            except Exception as e:
                click.echo(f"  {name}: error - {e}")

        if not solutions:
            click.echo("No algorithms produced valid solutions")
            return

        # Show comparison
        click.echo(f"\n=== Algorithm Comparison ===")  # noqa: F541
        click.echo(f"Graph: {graph_type} with {graph_size} nodes")
        click.echo(
            f"License configuration: solo={config.solo_price}, group={config.group_price}, max_size={config.group_size}"
        )
        click.echo("Results:")

        sorted_solutions = sorted(solutions.items(), key=lambda x: x[1].calculate_cost(config))
        for name, solution in sorted_solutions:
            cost = solution.calculate_cost(config)
            click.echo(f"  {name}: {cost:.2f}")

        best_algorithm = sorted_solutions[0][0]
        click.echo(f"Best algorithm: {best_algorithm}")
        click.echo("============================\n")

        output = Path("results/comparison")

        visualizer = GraphVisualizer()
        visualizer.compare_solutions(
            graph,
            solutions,
            config,
            title="Algorithm Comparison",
            save_path=output / "comparison.png",
        )

    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user")
    except Exception as e:
        click.echo(f"Error: {e}")
        logging.exception(f"Error: {e}")
        if ctx.obj["log_level"] == "DEBUG":
            raise
        sys.exit(1)


if __name__ == "__main__":
    cli()
