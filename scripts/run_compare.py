"""Algorithm comparison script."""

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
    print_solution_summary,
    print_comparison_table,
)


def run_comparison(
    algorithms: tuple,
    graph_type: str = "scale_free",
    graph_size: int = 15,
    solo_cost: float = 1.0,
    group_cost: float = 2.08,
    group_size: int = 6,
    seed: int = 42,
) -> None:
    """Compare multiple algorithms on the same graph."""
    config = create_license_config(solo_cost, group_cost, group_size)
    graph = create_test_graph(graph_type, graph_size, seed)
    all_algorithms = get_algorithms()

    selected_algorithms = {name: algo for name, algo in all_algorithms.items() if name in algorithms}

    if not selected_algorithms:
        click.echo("No valid algorithms selected")
        return

    click.echo(f"Comparing {len(selected_algorithms)} algorithms on {graph_type} graph (size: {graph_size})")

    start_time = datetime.now()
    solutions = {}
    results_summary = []

    # Run all algorithms
    for algo_name, algorithm in selected_algorithms.items():
        try:
            click.echo(f"Running {algo_name}...")
            solution = algorithm.solve(graph, config)

            stats = calculate_solution_stats(solution, config, graph)
            solutions[algo_name] = solution

            result = stats.copy()
            result["algorithm"] = algo_name
            results_summary.append(result)

            print_solution_summary(algo_name, stats)

        except Exception as e:
            click.echo(f"  ✗ {algo_name} failed: {e}")

    if not solutions:
        click.echo("No algorithms completed successfully")
        return

    # Print comparison table
    print_comparison_table(results_summary)

    # Create output directory
    output_dir = create_timestamped_path("results", "compare")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualization
    algorithms_str = "_vs_".join(sorted(solutions.keys()))
    save_path = output_dir / f"{algorithms_str}_{graph_type}_{graph_size}_comparison.png"

    click.echo("\nGenerating comparison visualization...")
    graph_visualizer.compare_solutions(
        graph=graph,
        solutions=solutions,
        config=config,
        title=f"Algorithm Comparison - {graph_type.title()} Graph (size: {graph_size})",
        save_path=str(save_path),
    )

    # Save results
    comparison_data = create_metadata(
        start_time,
        graph_info={
            "type": graph_type,
            "size": graph_size,
            "seed": seed,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
        },
        config={"solo_cost": solo_cost, "group_cost": group_cost, "group_size": group_size},
        results=results_summary,
    )

    json_path = save_results(comparison_data, output_dir, f"{algorithms_str}_{graph_type}_{graph_size}_results")
    click.echo(f"Comparison results saved to: {json_path}")
    click.echo("Comparison complete!")


@click.command()
@click.option(
    "--algorithms", multiple=True, default=["ilp", "greedy", "randomized", "genetic", "dominating_set", "tabu_search"]
)
@click.option("--graph-type", default="scale_free")
@click.option("--graph-size", default=15, type=int)
@click.option("--solo-cost", default=1.0, type=float)
@click.option("--group-cost", default=2.08, type=float)
@click.option("--group-size", default=6, type=int)
@click.option("--seed", default=42, type=int)
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def compare(algorithms, graph_type, graph_size, solo_cost, group_cost, group_size, seed, log_level):
    """Compare multiple algorithms on the same graph."""
    setup_logging(log_level)
    run_comparison(
        algorithms=algorithms,
        graph_type=graph_type,
        graph_size=graph_size,
        solo_cost=solo_cost,
        group_cost=group_cost,
        group_size=group_size,
        seed=seed,
    )


if __name__ == "__main__":
    compare()
