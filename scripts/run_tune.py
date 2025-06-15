"""Parameter tuning script."""

import click
from datetime import datetime

from graph_licensing.optimization import (
    tune_genetic_algorithm,
    tune_simulated_annealing,
    tune_tabu_search,
)
from scripts.utils import (
    setup_logging,
    create_license_config,
    create_test_graph,
    create_timestamped_path,
    create_metadata,
    save_results,
)


def run_tuning(
    algorithm: str,
    n_trials: int = 50,
    graph_type: str = "random",
    graph_size: int = 30,
    metric: str = "cost",
) -> None:
    """Run parameter tuning for an algorithm."""
    click.echo(f"Tuning {algorithm} parameters using {n_trials} trials")

    # Create test graphs and configs
    graphs = [create_test_graph(graph_type, graph_size, seed=i) for i in range(5)]
    configs = [create_license_config() for _ in range(5)]

    start_time = datetime.now()

    # Run tuning based on algorithm
    if algorithm == "genetic":
        results = tune_genetic_algorithm(
            graphs=graphs,
            configs=configs,
            metric=metric,
            n_trials=n_trials,
        )
    elif algorithm == "simulated_annealing":
        results = tune_simulated_annealing(
            graphs=graphs,
            configs=configs,
            metric=metric,
            n_trials=n_trials,
        )
    elif algorithm == "tabu_search":
        results = tune_tabu_search(
            graphs=graphs,
            configs=configs,
            metric=metric,
            n_trials=n_trials,
        )
    else:
        click.echo(f"Tuning not supported for {algorithm}")
        return

    end_time = datetime.now()

    # Display results
    click.echo(f"Best parameters: {results['best_params']}")
    click.echo(f"Best {metric}: {results['best_value']:.4f}")
    click.echo(f"Trials completed: {results['n_trials']}")
    click.echo(f"Tuning duration: {(end_time - start_time).total_seconds():.2f} seconds")

    # Create output directory
    output_dir = create_timestamped_path("results", "tune")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    full_results = create_metadata(
        start_time,
        algorithm=algorithm,
        metric=metric,
        graph_type=graph_type,
        graph_size=graph_size,
        n_trials=n_trials,
        best_params=results["best_params"],
        best_value=results["best_value"],
        trials_completed=results["n_trials"],
    )

    # Save full results and best parameters
    save_results(full_results, output_dir, f"{algorithm}_tuning_results")
    save_results(results["best_params"], output_dir, f"{algorithm}_best_params")

    click.echo(f"Tuning results saved to {output_dir}")


@click.command()
@click.option("--algorithm", required=True, type=click.Choice(["genetic", "simulated_annealing", "tabu_search"]))
@click.option("--n-trials", default=50, type=int)
@click.option("--graph-type", default="random")
@click.option("--graph-size", default=30, type=int)
@click.option("--metric", default="cost", type=click.Choice(["cost", "runtime", "quality"]))
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def tune(algorithm, n_trials, graph_type, graph_size, metric, log_level):
    """Tune algorithm parameters using Optuna."""
    setup_logging(log_level)
    run_tuning(
        algorithm=algorithm,
        n_trials=n_trials,
        graph_type=graph_type,
        graph_size=graph_size,
        metric=metric,
    )


if __name__ == "__main__":
    tune()
