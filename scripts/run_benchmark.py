"""Benchmarking script."""

import click
from datetime import datetime

from graph_licensing.utils import Benchmark, CSVLogger
from scripts.utils import (
    setup_logging,
    get_algorithms,
    create_license_config,
    create_test_graph,
    create_timestamped_path,
    create_metadata,
    save_results,
)


def run_benchmark(
    algorithms: tuple,
    graph_types: tuple = ("random", "scale_free", "small_world"),
    graph_sizes: tuple = tuple(range(10, 201, 10)),
    iterations: int = 1,
) -> None:
    """Run comprehensive benchmark of algorithms."""
    output_path = create_timestamped_path("results", "benchmark")
    output_path.mkdir(parents=True, exist_ok=True)
    click.echo(f"Results will be saved to: {output_path}")

    benchmark_runner = Benchmark()
    all_algorithms = get_algorithms()

    selected_algorithms = {name: algo for name, algo in all_algorithms.items() if name in algorithms}

    if not selected_algorithms:
        click.echo("No valid algorithms selected")
        return

    click.echo(f"Running benchmark with {len(selected_algorithms)} algorithms")
    click.echo(f"Graph types: {list(graph_types)}")
    click.echo(f"Graph sizes: {list(graph_sizes)}")

    results = []
    start_time = datetime.now()

    # Run benchmark
    for graph_type in graph_types:
        for size in graph_sizes:
            for i in range(iterations):
                graph = create_test_graph(graph_type, size, seed=42 + i)
                config = create_license_config()

                for algo_name, algorithm in selected_algorithms.items():
                    test_name = f"{graph_type}_{size}_{i}_{algo_name}"

                    try:
                        result = benchmark_runner.run_single_test(algorithm, graph, config, test_name)
                        result["timestamp"] = datetime.now().isoformat()
                        results.append(result)
                        click.echo(f"✓ {test_name}: {result['total_cost']:.2f}")

                    except Exception as e:
                        click.echo(f"✗ {test_name}: {e}")

    # Create metadata
    end_time = datetime.now()
    benchmark_metadata = create_metadata(
        start_time,
        total_tests=len(results),
        algorithms=list(algorithms),
        graph_types=list(graph_types),
        graph_sizes=list(graph_sizes),
        iterations=iterations,
    )

    # Save results
    save_results(results, output_path, "benchmark_results")
    save_results(benchmark_metadata, output_path, "benchmark_metadata")

    # Save CSV format
    csv_logger = CSVLogger(output_path / "benchmark_results.csv")
    csv_logger.log_results(results)

    click.echo(f"Benchmark complete! Results saved to {output_path}")
    click.echo(f"Duration: {benchmark_metadata['duration_seconds']:.2f} seconds")


@click.command()
@click.option(
    "--algorithms", 
    multiple=True, 
    default=["ilp", "greedy", "randomized", "genetic", "dominating_set", "tabu_search"]
)
@click.option("--graph-types", multiple=True, default=["random", "scale_free", "small_world"])
@click.option("--graph-sizes", multiple=True, type=int, default=[i for i in range(10, 201, 10)])
@click.option("--iterations", default=1, type=int)
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def benchmark(algorithms, graph_types, graph_sizes, iterations, log_level):
    """Run comprehensive benchmark of algorithms."""
    setup_logging(log_level)
    run_benchmark(
        algorithms=algorithms,
        graph_types=graph_types,
        graph_sizes=graph_sizes,
        iterations=iterations,
    )


if __name__ == "__main__":
    benchmark()
