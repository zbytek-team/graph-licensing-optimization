"""Dynamic graph analysis script."""

import click
from datetime import datetime

from graph_licensing.utils import Benchmark
from graph_licensing.visualizers import graph_visualizer
from scripts.common import (
    setup_logging,
    get_algorithms,
    create_license_config,
    create_test_graph,
    create_timestamped_path,
    create_metadata,
    save_results,
)


def run_dynamic_analysis(
    algorithm: str,
    graph_type: str = "random",
    initial_size: int = 20,
    iterations: int = 10,
    modification_prob: float = 1.0,
    create_gif: bool = False,
) -> None:
    """Run dynamic analysis of algorithm on evolving graph."""
    algorithms = get_algorithms()
    if algorithm not in algorithms:
        click.echo(f"Unknown algorithm: {algorithm}")
        return

    output_path = create_timestamped_path("results", "dynamic")
    output_path.mkdir(parents=True, exist_ok=True)
    click.echo(f"Results will be saved to: {output_path}")

    click.echo(f"Running dynamic analysis with {algorithm}")
    click.echo(f"Iterations: {iterations}, Modification probability: {modification_prob}")

    initial_graph = create_test_graph(graph_type, initial_size, seed=42)
    config = create_license_config()

    benchmark_runner = Benchmark()
    algo = algorithms[algorithm]
    start_time = datetime.now()

    try:
        # Run dynamic test
        if create_gif:
            results, graph_states, solutions = benchmark_runner.run_dynamic_test_with_states(
                algo, initial_graph, config, iterations, modification_prob
            )
        else:
            results = benchmark_runner.run_dynamic_test(algo, initial_graph, config, iterations, modification_prob)
            graph_states, solutions = None, None

        # Add timestamps
        for result in results:
            result["timestamp"] = datetime.now().isoformat()

        # Calculate statistics
        costs = [r["total_cost"] for r in results if r["success"]]
        if costs:
            click.echo(f"Average cost: {sum(costs) / len(costs):.2f}")
            click.echo(f"Cost range: {min(costs):.2f} - {max(costs):.2f}")

        # Print evolution summary
        if results:
            _print_evolution_summary(results, initial_size, initial_graph)

        # Create metadata
        end_time = datetime.now()
        dynamic_metadata = create_metadata(
            start_time,
            algorithm=algorithm,
            graph_type=graph_type,
            initial_size=initial_size,
            iterations=iterations,
            modification_prob=modification_prob,
            successful_runs=len([r for r in results if r["success"]]),
            total_runs=len(results),
        )

        # Save results
        save_results(results, output_path, f"dynamic_{algorithm}_results")
        save_results(dynamic_metadata, output_path, f"dynamic_{algorithm}_metadata")

        # Create GIF if requested
        if create_gif and graph_states and solutions:
            _create_evolution_gif(
                graph_states, solutions, config, algorithm, graph_type, output_path
            )

        click.echo(f"Dynamic analysis complete! Results saved to {output_path}")
        click.echo(f"Duration: {dynamic_metadata['duration_seconds']:.2f} seconds")

    except Exception as e:
        click.echo(f"Dynamic analysis failed: {e}")


def _print_evolution_summary(results: list, initial_size: int, initial_graph) -> None:
    """Print summary of graph evolution."""
    click.echo("\nGraph Evolution:")
    click.echo("Iter | Nodes | Edges | Cost | Node Δ | Edge Δ")
    click.echo("-" * 45)
    
    prev_nodes, prev_edges = initial_size, initial_graph.number_of_edges()
    
    for r in results[:min(10, len(results))]:  # Show first 10 iterations
        node_delta = r["n_nodes"] - prev_nodes
        edge_delta = r["n_edges"] - prev_edges
        click.echo(
            f"{r['iteration']:4} | {r['n_nodes']:5} | {r['n_edges']:5} | "
            f"{r['total_cost']:4.0f} | {node_delta:+3} | {edge_delta:+3}"
        )
        prev_nodes, prev_edges = r["n_nodes"], r["n_edges"]
    
    if len(results) > 10:
        click.echo("... (truncated)")

    # Calculate modification statistics
    node_changes = [abs(results[i]["n_nodes"] - results[i - 1]["n_nodes"]) for i in range(1, len(results))]
    edge_changes = [abs(results[i]["n_edges"] - results[i - 1]["n_edges"]) for i in range(1, len(results))]
    
    if node_changes and edge_changes:
        click.echo("\nModification Summary:")
        click.echo(f"Avg node changes per iteration: {sum(node_changes) / len(node_changes):.1f}")
        click.echo(f"Avg edge changes per iteration: {sum(edge_changes) / len(edge_changes):.1f}")
        click.echo(f"Total node variance: {max(r['n_nodes'] for r in results) - min(r['n_nodes'] for r in results)}")
        click.echo(f"Total edge variance: {max(r['n_edges'] for r in results) - min(r['n_edges'] for r in results)}")


def _create_evolution_gif(graph_states, solutions, config, algorithm, graph_type, output_path) -> None:
    """Create animated GIF of graph evolution."""
    click.echo("Creating animated GIF...")
    try:
        gif_path = output_path / f"dynamic_{algorithm}_evolution.gif"

        # Filter valid solutions
        valid_indices = [i for i, sol in enumerate(solutions) if sol is not None]
        if len(valid_indices) < 2:
            click.echo("Warning: Not enough valid solutions to create GIF")
            return

        valid_graph_states = [graph_states[i] for i in valid_indices]
        valid_solutions = [solutions[i] for i in valid_indices]

        graph_visualizer.create_dynamic_gif(
            graph_states=valid_graph_states,
            solutions=valid_solutions,
            config=config,
            algorithm_name=algorithm.title(),
            title=f"Dynamic {graph_type.title()} Graph Evolution",
            save_path=str(gif_path),
            duration=1.5,  # 1.5 seconds per frame
            show_changes=True,
        )
        click.echo(f"GIF saved to: {gif_path}")
    except Exception as e:
        click.echo(f"Failed to create GIF: {e}")


@click.command()
@click.option("--algorithm", required=True)
@click.option("--graph-type", default="random")
@click.option("--initial-size", default=20, type=int)
@click.option("--iterations", default=10, type=int)
@click.option(
    "--modification-prob", 
    default=1.0, 
    type=float, 
    help="Modification intensity (1.0 = ~3 changes per iteration)"
)
@click.option("--create-gif", is_flag=True, help="Create animated GIF showing graph evolution")
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def dynamic(algorithm, graph_type, initial_size, iterations, modification_prob, create_gif, log_level):
    """Run dynamic analysis on evolving graphs."""
    setup_logging(log_level)
    run_dynamic_analysis(
        algorithm=algorithm,
        graph_type=graph_type,
        initial_size=initial_size,
        iterations=iterations,
        modification_prob=modification_prob,
        create_gif=create_gif,
    )


if __name__ == "__main__":
    dynamic()
