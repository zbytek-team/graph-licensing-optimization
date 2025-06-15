import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import click
import networkx as nx

from graph_licensing.algorithms import (
    AntColonyAlgorithm,
    DominatingSetAlgorithm,
    GeneticAlgorithm,
    GreedyAlgorithm,
    ILPAlgorithm,
    NaiveAlgorithm,
    RandomizedAlgorithm,
    SimulatedAnnealingAlgorithm,
    TabuSearchAlgorithm,
)
from graph_licensing.generators.graph_generator import GraphGenerator
from graph_licensing.models.license import LicenseConfig
from graph_licensing.optimization import OptunaTuner
from graph_licensing.utils import Benchmark, FileIO
from graph_licensing.visualizers.graph_visualizer import GraphVisualizer


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("graph_licensing.log"),
        ],
    )


def get_timestamp_suffix() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_timestamped_path(base_path: str, command_name: str) -> Path:
    timestamp = get_timestamp_suffix()
    return Path(base_path) / f"{command_name}_{timestamp}"


def get_algorithms() -> Dict[str, Any]:
    return {
        "ant_colony": AntColonyAlgorithm(),
        "greedy": GreedyAlgorithm(),
        "genetic": GeneticAlgorithm(),
        "simulated_annealing": SimulatedAnnealingAlgorithm(),
        "tabu_search": TabuSearchAlgorithm(),
        "ilp": ILPAlgorithm(),
        "naive": NaiveAlgorithm(),
        "dominating_set": DominatingSetAlgorithm(),
        "randomized": RandomizedAlgorithm(),
    }


def create_license_config(solo_cost: float = 1.0, group_cost: float = 2.08, group_size: int = 6) -> LicenseConfig:
    return LicenseConfig.create_flexible(
        {
            "solo": {"price": solo_cost, "min_size": 1, "max_size": 1},
            "duo": {"price": solo_cost * 1.6, "min_size": 2, "max_size": 2},
            "family": {"price": group_cost, "min_size": 2, "max_size": group_size},
        }
    )


def create_test_graph(graph_type: str, size: int, seed: int = None, **kwargs) -> nx.Graph:
    return GraphGenerator.generate_graph(graph_type=graph_type, size=size, seed=seed, **kwargs)


@click.group()
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
@click.pass_context
def cli(ctx, log_level):
    ctx.ensure_object(dict)
    setup_logging(log_level)
    ctx.obj["log_level"] = log_level


@cli.command()
@click.option("--algorithm", required=True, type=click.Choice(list(get_algorithms().keys())))
@click.option("--graph-type", default="random")
@click.option("--graph-size", default=20, type=int)
@click.option("--solo-cost", default=1.0, type=float)
@click.option("--group-cost", default=2.08, type=float)
@click.option("--group-size", default=6, type=int)
@click.option("--seed", default=42, type=int)
def single(algorithm, graph_type, graph_size, solo_cost, group_cost, group_size, seed):
    config = create_license_config(solo_cost, group_cost, group_size)
    graph = create_test_graph(graph_type, graph_size, seed)
    algorithms = get_algorithms()

    if algorithm not in algorithms:
        click.echo(f"Unknown algorithm: {algorithm}")
        return

    algo = algorithms[algorithm]
    click.echo(f"Running {algorithm} on {graph_type} graph (size: {graph_size})")

    try:
        solution = algo.solve(graph, config)

        click.echo("Solution found!")
        click.echo(f"Total cost: {solution.calculate_cost(config)}")

        solo_count = sum(
            1 for license_type, groups in solution.licenses.items() for members in groups.values() if len(members) == 1
        )
        group_count = sum(
            1 for license_type, groups in solution.licenses.items() for members in groups.values() if len(members) > 1
        )

        click.echo(f"Solo licenses: {solo_count}")
        click.echo(f"Group licenses: {group_count}")
        click.echo(f"Valid solution: {solution.is_valid(graph, config)}")

        output_dir = create_timestamped_path("results", "single")
        output_dir.mkdir(parents=True, exist_ok=True)

        visualizer = GraphVisualizer()
        viz_path = output_dir / f"{algorithm}_{graph_type}_{graph_size}_visualization.png"
        visualizer.visualize_solution(
            graph,
            solution,
            config,
            title=f"{algorithm.title()} Algorithm - {graph_type.title()} Graph",
            save_path=str(viz_path),
        )
        click.echo(f"Visualization saved to: {viz_path}")

        results = {
            "timestamp": datetime.now().isoformat(),
            "algorithm": algorithm,
            "graph_type": graph_type,
            "graph_size": graph_size,
            "config": {"solo_cost": solo_cost, "group_cost": group_cost, "group_size": group_size},
            "solution": {
                "total_cost": solution.calculate_cost(config),
                "solo_licenses": sum(
                    1
                    for license_type, groups in solution.licenses.items()
                    for members in groups.values()
                    if len(members) == 1
                ),
                "group_licenses": sum(
                    1
                    for license_type, groups in solution.licenses.items()
                    for members in groups.values()
                    if len(members) > 1
                ),
                "valid": solution.is_valid(graph, config),
            },
        }
        results_path = output_dir / f"{algorithm}_{graph_type}_{graph_size}_results.json"
        FileIO.save_json(results, results_path)
        click.echo(f"Results saved to: {results_path}")

    except Exception as e:
        click.echo(f"Algorithm failed: {e}")


@cli.command()
@click.option(
    "--algorithms", multiple=True, default=["ilp", "greedy", "randomized", "genetic", "dominating_set", "tabu_search"]
)
@click.option("--graph-types", multiple=True, default=["random", "scale_free", "small_world"])
@click.option("--graph-sizes", multiple=True, type=int, default=[i for i in range(10, 201, 10)])
@click.option("--iterations", default=1, type=int)
def benchmark(algorithms, graph_types, graph_sizes, iterations):
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

    end_time = datetime.now()
    benchmark_metadata = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "total_tests": len(results),
        "algorithms": list(algorithms),
        "graph_types": list(graph_types),
        "graph_sizes": list(graph_sizes),
        "iterations": iterations,
    }

    FileIO.save_json(results, output_path / "benchmark_results.json")
    FileIO.save_json(benchmark_metadata, output_path / "benchmark_metadata.json")

    from src.graph_licensing.utils import CSVLogger

    csv_logger = CSVLogger(output_path / "benchmark_results.csv")
    csv_logger.log_results(results)

    click.echo(f"Benchmark complete! Results saved to {output_path}")
    click.echo(f"Duration: {benchmark_metadata['duration_seconds']:.2f} seconds")


@cli.command()
@click.option("--algorithm", required=True, type=click.Choice(["genetic", "simulated_annealing", "tabu_search"]))
@click.option("--n-trials", default=50, type=int)
@click.option("--graph-type", default="random")
@click.option("--graph-size", default=30, type=int)
@click.option("--metric", default="cost", type=click.Choice(["cost", "runtime", "quality"]))
def tune(algorithm, n_trials, graph_type, graph_size, metric):
    click.echo(f"Tuning {algorithm} parameters using {n_trials} trials")

    graphs = [create_test_graph(graph_type, graph_size, seed=i) for i in range(5)]
    configs = [create_license_config() for _ in range(5)]

    tuner = OptunaTuner(n_trials=n_trials)
    start_time = datetime.now()

    if algorithm == "genetic":
        results = tuner.tune_genetic_algorithm(graphs, configs, metric)
    elif algorithm == "simulated_annealing":
        results = tuner.tune_simulated_annealing(graphs, configs, metric)
    elif algorithm == "tabu_search":
        results = tuner.tune_tabu_search(graphs, configs, metric)
    else:
        click.echo(f"Tuning not supported for {algorithm}")
        return

    end_time = datetime.now()

    click.echo(f"Best parameters: {results['best_params']}")
    click.echo(f"Best {metric}: {results['best_value']:.4f}")
    click.echo(f"Trials completed: {results['n_trials']}")
    click.echo(f"Tuning duration: {(end_time - start_time).total_seconds():.2f} seconds")

    output_dir = create_timestamped_path("results", "tune")
    output_dir.mkdir(parents=True, exist_ok=True)

    full_results = {
        "timestamp": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "algorithm": algorithm,
        "metric": metric,
        "graph_type": graph_type,
        "graph_size": graph_size,
        "n_trials": n_trials,
        "best_params": results["best_params"],
        "best_value": results["best_value"],
        "trials_completed": results["n_trials"],
    }

    FileIO.save_json(full_results, output_dir / f"{algorithm}_tuning_results.json")
    FileIO.save_json(results["best_params"], output_dir / f"{algorithm}_best_params.json")
    click.echo(f"Tuning results saved to {output_dir}")


@cli.command()
@click.option("--algorithm", required=True)
@click.option("--graph-type", default="random")
@click.option("--initial-size", default=20, type=int)
@click.option("--iterations", default=10, type=int)
@click.option(
    "--modification-prob", default=1.0, type=float, help="Modification intensity (1.0 = ~3 changes per iteration)"
)
@click.option("--create-gif", is_flag=True, help="Create animated GIF showing graph evolution")
def dynamic(algorithm, graph_type, initial_size, iterations, modification_prob, create_gif):
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
        if create_gif:
            results, graph_states, solutions = benchmark_runner.run_dynamic_test_with_states(
                algo, initial_graph, config, iterations, modification_prob
            )
        else:
            results = benchmark_runner.run_dynamic_test(algo, initial_graph, config, iterations, modification_prob)
            graph_states, solutions = None, None

        for result in results:
            result["timestamp"] = datetime.now().isoformat()

        costs = [r["total_cost"] for r in results if r["success"]]
        if costs:
            click.echo(f"Average cost: {sum(costs) / len(costs):.2f}")
            click.echo(f"Cost range: {min(costs):.2f} - {max(costs):.2f}")

        if results:
            click.echo("\nGraph Evolution:")
            click.echo("Iter | Nodes | Edges | Cost | Node Δ | Edge Δ")
            click.echo("-" * 45)
            prev_nodes, prev_edges = initial_size, initial_graph.number_of_edges()
            for r in results[: min(10, len(results))]:  # Show first 10 iterations
                node_delta = r["n_nodes"] - prev_nodes
                edge_delta = r["n_edges"] - prev_edges
                click.echo(
                    f"{r['iteration']:4} | {r['n_nodes']:5} | {r['n_edges']:5} | {r['total_cost']:4.0f} | {node_delta:+3} | {edge_delta:+3}"
                )
                prev_nodes, prev_edges = r["n_nodes"], r["n_edges"]
            if len(results) > 10:
                click.echo("... (truncated)")

            node_changes = [abs(results[i]["n_nodes"] - results[i - 1]["n_nodes"]) for i in range(1, len(results))]
            edge_changes = [abs(results[i]["n_edges"] - results[i - 1]["n_edges"]) for i in range(1, len(results))]
            if node_changes and edge_changes:
                click.echo("\nModification Summary:")
                click.echo(f"Avg node changes per iteration: {sum(node_changes) / len(node_changes):.1f}")
                click.echo(f"Avg edge changes per iteration: {sum(edge_changes) / len(edge_changes):.1f}")
                click.echo(
                    f"Total node variance: {max(r['n_nodes'] for r in results) - min(r['n_nodes'] for r in results)}"
                )
                click.echo(
                    f"Total edge variance: {max(r['n_edges'] for r in results) - min(r['n_edges'] for r in results)}"
                )

        end_time = datetime.now()
        dynamic_metadata = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "algorithm": algorithm,
            "graph_type": graph_type,
            "initial_size": initial_size,
            "iterations": iterations,
            "modification_prob": modification_prob,
            "successful_runs": len([r for r in results if r["success"]]),
            "total_runs": len(results),
        }

        FileIO.save_json(results, output_path / f"dynamic_{algorithm}_results.json")
        FileIO.save_json(dynamic_metadata, output_path / f"dynamic_{algorithm}_metadata.json")

        if create_gif and graph_states and solutions:
            click.echo("Creating animated GIF...")
            try:
                visualizer = GraphVisualizer()
                gif_path = output_path / f"dynamic_{algorithm}_evolution.gif"

                valid_indices = [i for i, sol in enumerate(solutions) if sol is not None]
                if len(valid_indices) < 2:
                    click.echo("Warning: Not enough valid solutions to create GIF")
                else:
                    valid_graph_states = [graph_states[i] for i in valid_indices]
                    valid_solutions = [solutions[i] for i in valid_indices]

                    visualizer.create_dynamic_gif(
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

        click.echo(f"Dynamic analysis complete! Results saved to {output_path}")
        click.echo(f"Duration: {dynamic_metadata['duration_seconds']:.2f} seconds")

    except Exception as e:
        click.echo(f"Dynamic analysis failed: {e}")


@cli.command()
@click.option(
    "--algorithms", multiple=True, default=["ilp", "greedy", "randomized", "genetic", "dominating_set", "tabu_search"]
)
@click.option("--graph-type", default="scale_free")
@click.option("--graph-size", default=15, type=int)
@click.option("--solo-cost", default=1.0, type=float)
@click.option("--group-cost", default=2.08, type=float)
@click.option("--group-size", default=6, type=int)
@click.option("--seed", default=42, type=int)
def compare(algorithms, graph_type, graph_size, solo_cost, group_cost, group_size, seed):
    config = create_license_config(solo_cost, group_cost, group_size)
    graph = create_test_graph(graph_type, graph_size, seed)
    all_algorithms = get_algorithms()

    selected_algorithms = {name: algo for name, algo in all_algorithms.items() if name in algorithms}

    if not selected_algorithms:
        click.echo("No valid algorithms selected")
        return

    click.echo(f"Comparing {len(selected_algorithms)} algorithms on {graph_type} graph (size: {graph_size})")

    solutions = {}
    results_summary = []

    for algo_name, algorithm in selected_algorithms.items():
        try:
            click.echo(f"Running {algo_name}...")
            solution = algorithm.solve(graph, config)

            total_cost = solution.calculate_cost(config)
            solo_count = sum(
                1
                for license_type, groups in solution.licenses.items()
                for members in groups.values()
                if len(members) == 1
            )
            group_count = sum(
                1
                for license_type, groups in solution.licenses.items()
                for members in groups.values()
                if len(members) > 1
            )
            is_valid = solution.is_valid(graph, config)

            solutions[algo_name] = solution
            results_summary.append(
                {
                    "algorithm": algo_name,
                    "total_cost": total_cost,
                    "solo_licenses": solo_count,
                    "group_licenses": group_count,
                    "valid": is_valid,
                }
            )

            click.echo(f"  ✓ Cost: {total_cost:.2f}, Solo: {solo_count}, Groups: {group_count}")

        except Exception as e:
            click.echo(f"  ✗ {algo_name} failed: {e}")

    if not solutions:
        click.echo("No algorithms completed successfully")
        return

    click.echo("\n" + "=" * 60)
    click.echo("COMPARISON SUMMARY")
    click.echo("=" * 60)

    results_summary.sort(key=lambda x: x["total_cost"])

    for i, result in enumerate(results_summary, 1):
        status = "✓" if result["valid"] else "✗"
        click.echo(
            f"{i}. {result['algorithm']}: ${result['total_cost']:.2f} "
            f"(Solo: {result['solo_licenses']}, Groups: {result['group_licenses']}) {status}"
        )

    visualizer = GraphVisualizer()

    output_dir = create_timestamped_path("results", "compare")
    output_dir.mkdir(parents=True, exist_ok=True)
    algorithms_str = "_vs_".join(sorted(solutions.keys()))
    save_path = output_dir / f"{algorithms_str}_{graph_type}_{graph_size}_comparison.png"

    click.echo("\nGenerating comparison visualization...")
    visualizer.compare_solutions(
        graph=graph,
        solutions=solutions,
        config=config,
        title=f"Algorithm Comparison - {graph_type.title()} Graph (size: {graph_size})",
        save_path=str(save_path),
    )

    comparison_data = {
        "timestamp": datetime.now().isoformat(),
        "graph_info": {
            "type": graph_type,
            "size": graph_size,
            "seed": seed,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
        },
        "config": {"solo_cost": solo_cost, "group_cost": group_cost, "group_size": group_size},
        "results": results_summary,
    }

    json_path = output_dir / f"{algorithms_str}_{graph_type}_{graph_size}_results.json"
    FileIO.save_json(comparison_data, json_path)
    click.echo(f"Comparison results saved to: {json_path}")

    click.echo("Comparison complete!")


@cli.command()
@click.option("--input-dir", type=click.Path(), help="Input directory with benchmark results")
def analyze(input_dir):
    if not input_dir:
        input_dir = "results/benchmark"

    output_path = create_timestamped_path("results", "analysis")
    output_path.mkdir(parents=True, exist_ok=True)
    click.echo(f"Analysis results will be saved to: {output_path}")

    click.echo("Running analysis on benchmark results...")

    try:
        start_time = datetime.now()

        from src.graph_licensing.analysis import AnalysisRunner

        analyzer = AnalysisRunner(results_path=str(input_dir))
        analyzer.output_dir = output_path
        analyzer.generate_comprehensive_report()

        end_time = datetime.now()

        analysis_metadata = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "input_directory": str(input_dir),
            "output_directory": str(output_path),
        }

        FileIO.save_json(analysis_metadata, output_path / "analysis_metadata.json")

        click.echo(f"Analysis complete! Results saved to {output_path}")
        click.echo(f"Duration: {analysis_metadata['duration_seconds']:.2f} seconds")

    except Exception as e:
        click.echo(f"Analysis failed: {e}")


if __name__ == "__main__":
    cli()
