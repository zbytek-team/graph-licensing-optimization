"""Single algorithm execution script."""

import click
from datetime import datetime

from graph_licensing.visualizers import graph_visualizer
from scripts.utils import (
    setup_logging,
    get_algorithms,
    create_license_config,
    create_test_graph,
    create_timestamped_path,
    calculate_solution_stats,
    create_metadata,
    save_results,
)


def run_single_algorithm(
    algorithm: str,
    graph_type: str = "random",
    graph_size: int = 20,
    solo_cost: float = 1.0,
    group_cost: float = 2.08,
    group_size: int = 6,
    seed: int = 42,
) -> None:
    """Run a single algorithm and save results."""
    config = create_license_config(solo_cost, group_cost, group_size)
    graph = create_test_graph(graph_type, graph_size, seed)
    algorithms = get_algorithms()

    if algorithm not in algorithms:
        click.echo(f"Unknown algorithm: {algorithm}")
        return

    algo = algorithms[algorithm]
    click.echo(f"Running {algorithm} on {graph_type} graph (size: {graph_size})")

    start_time = datetime.now()

    try:
        solution = algo.solve(graph, config)
        stats = calculate_solution_stats(solution, config, graph)

        click.echo("Solution found!")
        click.echo(f"Total cost: {stats['total_cost']}")
        click.echo(f"Solo licenses: {stats['solo_licenses']}")
        click.echo(f"Group licenses: {stats['group_licenses']}")
        click.echo(f"Valid solution: {stats['valid']}")

        # Create output directory
        output_dir = create_timestamped_path("results", "single")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create visualization
        viz_path = output_dir / f"{algorithm}_{graph_type}_{graph_size}_visualization.png"
        graph_visualizer.visualize_solution(
            graph,
            solution,
            config,
            title=f"{algorithm.title()} Algorithm - {graph_type.title()} Graph",
            save_path=str(viz_path),
        )
        click.echo(f"Visualization saved to: {viz_path}")

        # Save results
        results = create_metadata(
            start_time,
            algorithm=algorithm,
            graph_type=graph_type,
            graph_size=graph_size,
            config={"solo_cost": solo_cost, "group_cost": group_cost, "group_size": group_size},
            solution=stats,
        )

        results_path = save_results(results, output_dir, f"{algorithm}_{graph_type}_{graph_size}_results")
        click.echo(f"Results saved to: {results_path}")

    except Exception as e:
        click.echo(f"Algorithm failed: {e}")


@click.command()
@click.option("--algorithm", required=True, type=click.Choice(list(get_algorithms().keys())))
@click.option("--graph-type", default="random")
@click.option("--graph-size", default=20, type=int)
@click.option("--solo-cost", default=1.0, type=float)
@click.option("--group-cost", default=2.08, type=float)
@click.option("--group-size", default=6, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def single(algorithm, graph_type, graph_size, solo_cost, group_cost, group_size, seed, log_level):
    """Run a single algorithm on a graph."""
    setup_logging(log_level)
    run_single_algorithm(
        algorithm=algorithm,
        graph_type=graph_type,
        graph_size=graph_size,
        solo_cost=solo_cost,
        group_cost=group_cost,
        group_size=group_size,
        seed=seed,
    )


if __name__ == "__main__":
    single()
