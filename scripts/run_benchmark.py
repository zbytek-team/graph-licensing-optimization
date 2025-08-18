import argparse
import signal
import time

from src.algorithms.greedy import GreedyAlgorithm
from src.algorithms.ilp import ILPSolver
from src.algorithms.tabu_search import TabuSearch
from src.core import LicenseConfigFactory
from src.graphs.generator import GraphGeneratorFactory
from src.utils import BenchmarkCSVWriter

GRAPH_K_VALUES = [2, 4, 6, 8]
GRAPH_P_VALUES = [0.1, 0.2, 0.3, 0.4]
GRAPH_M_VALUES = [1, 2, 3, 4]
MAX_EXECUTION_TIME = 60

ALGORITHM_MAP = {
    "greedy": (GreedyAlgorithm, "Greedy Algorithm"),
    "tabu": (TabuSearch, "Tabu Search"),
    "ilp": (ILPSolver, "ILP Solver"),
}


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Algorithm execution timeout")


def run_algorithm_with_timeout(algorithm, graph, license_types, timeout_seconds):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        start_time = time.time()
        solution = algorithm.solve(graph, license_types)
        end_time = time.time()
        execution_time = end_time - start_time
        signal.alarm(0)
        return solution, execution_time
    except TimeoutException:
        signal.alarm(0)
        return None, timeout_seconds
    except Exception as e:
        signal.alarm(0)
        print(f"Algorithm failed with error: {e}")
        return None, -1


def generate_graph_configs(graph_types):
    configs = []
    for graph_type in graph_types:
        if graph_type == "random":
            for p in GRAPH_P_VALUES:
                configs.append({"type": graph_type, "p": p, "k": None, "m": None})
        elif graph_type == "scale_free":
            for m in GRAPH_M_VALUES:
                configs.append({"type": graph_type, "p": None, "k": None, "m": m})
        elif graph_type == "small_world":
            for k in GRAPH_K_VALUES:
                for p in GRAPH_P_VALUES:
                    configs.append({"type": graph_type, "p": p, "k": k, "m": None})
    return configs


def parse_args():
    parser = argparse.ArgumentParser(description="Run graph licensing benchmark")
    parser.add_argument(
        "--graph-type",
        nargs="+",
        choices=["random", "scale_free", "small_world"],
        default=["random", "scale_free", "small_world"],
        help="Types of graphs to benchmark",
    )
    parser.add_argument(
        "--nodes",
        nargs="+",
        type=int,
        default=[10, 15, 20, 25, 30],
        help="Number of nodes for generated graphs",
    )
    parser.add_argument(
        "--license-config",
        nargs="+",
        default=["roman_domination", "duolingo_super", "spotify"],
        help="License configurations to use",
    )
    parser.add_argument(
        "--algorithm",
        nargs="+",
        choices=list(ALGORITHM_MAP.keys()),
        default=list(ALGORITHM_MAP.keys()),
        help="Algorithms to run",
    )
    parser.add_argument(
        "--output-dir",
        default="results/stats",
        help="Directory to store CSV results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for graph generation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_writer = BenchmarkCSVWriter(args.output_dir)
    print(f"Starting benchmark - results will be saved to: {csv_writer.get_csv_path()}")
    graph_configs = generate_graph_configs(args.graph_type)
    algorithms = [
        (ALGORITHM_MAP[name][0](), ALGORITHM_MAP[name][1]) for name in args.algorithm
    ]
    total_experiments = (
        len(args.license_config)
        * len(algorithms)
        * len(args.nodes)
        * len(graph_configs)
    )
    current_experiment = 0
    algorithm_skip_flags = {}
    for license_config in args.license_config:
        for algorithm, algorithm_name in algorithms:
            for nodes in args.nodes:
                for graph_config in graph_configs:
                    graph_type = graph_config["type"]
                    skip_key = f"{algorithm_name}_{graph_type}_{graph_config}_{nodes}"
                    if skip_key not in algorithm_skip_flags:
                        algorithm_skip_flags[skip_key] = False
                    if algorithm_skip_flags[skip_key]:
                        continue
                    current_experiment += 1
                    print(
                        f"Experiment {current_experiment}/{total_experiments}: {algorithm_name} on {graph_type} graph ({nodes} nodes) with {license_config}"
                    )
                    try:
                        generator = GraphGeneratorFactory.get_generator(graph_type)
                        kwargs = {"seed": args.seed}
                        if graph_config["k"] is not None:
                            kwargs["k"] = graph_config["k"]
                        if graph_config["p"] is not None:
                            kwargs["p"] = graph_config["p"]
                        if graph_config["m"] is not None:
                            kwargs["m"] = graph_config["m"]
                        graph = generator(n_nodes=nodes, **kwargs)
                        license_types = LicenseConfigFactory.get_config(license_config)
                        solution, execution_time = run_algorithm_with_timeout(
                            algorithm, graph, license_types, MAX_EXECUTION_TIME
                        )
                        if solution is None:
                            if execution_time == MAX_EXECUTION_TIME:
                                print(
                                    f"Timeout for {algorithm_name} - skipping larger graphs for this algorithm+graph_type combination"
                                )
                                for larger_nodes in args.nodes:
                                    if larger_nodes >= nodes:
                                        larger_skip_key = (
                                            f"{algorithm_name}_{graph_type}_{graph_config}_{larger_nodes}"
                                        )
                                        algorithm_skip_flags[larger_skip_key] = True
                            continue
                        avg_degree = sum(dict(graph.degree()).values()) / len(
                            graph.nodes()
                        )
                        result = {
                            "algorithm": algorithm_name,
                            "graph_type": graph_type,
                            "nodes": nodes,
                            "edges": len(graph.edges()),
                            "graph_k": graph_config["k"],
                            "graph_p": graph_config["p"],
                            "graph_m": graph_config["m"],
                            "license_config": license_config,
                            "cost": round(solution.total_cost, 2),
                            "execution_time": execution_time,
                            "groups_count": len(solution.groups),
                            "avg_degree": round(avg_degree, 3),
                            "seed": args.seed,
                        }
                        csv_writer.write_result(result)
                        print(
                            f"  Result: cost={round(solution.total_cost, 2)}, time={execution_time:.3f}s, groups={len(solution.groups)}"
                        )
                    except Exception as e:
                        print(f"  Error: {e}")
    print(
        f"Benchmark completed! Results saved to: {csv_writer.get_csv_path()}"
    )


if __name__ == "__main__":
    main()
