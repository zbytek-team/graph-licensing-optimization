from src.algorithms.ilp import ILPSolver
from src.algorithms.greedy import GreedyAlgorithm
from src.algorithms.tabu_search import TabuSearch
from src.utils.graphs.generator import GraphGeneratorFactory
from src.utils.licenses import LicenseConfigFactory
from src.utils.csv_writer import BenchmarkCSVWriter
import time
import signal

GRAPH_TYPES = ["random", "scale_free", "small_world"]
# GRAPH_NODES = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500]
GRAPH_NODES = [10, 15, 20, 25, 30]
GRAPH_K_VALUES = [2, 4, 6, 8]
GRAPH_P_VALUES = [0.1, 0.2, 0.3, 0.4]
GRAPH_M_VALUES = [1, 2, 3, 4]
GRAPH_SEED = 42
LICENSE_CONFIGS = ["roman_domination", "duolingo_super", "spotify"]
MAX_EXECUTION_TIME = 60
ALGORITHMS = [
    (GreedyAlgorithm(), "Greedy Algorithm"),
    (TabuSearch(), "Tabu Search"),
    (ILPSolver(), "ILP Solver"),
]


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


def generate_graph_configs():
    configs = []

    for graph_type in GRAPH_TYPES:
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


def main():
    csv_writer = BenchmarkCSVWriter()
    print(f"Starting benchmark - results will be saved to: {csv_writer.get_csv_path()}")

    graph_configs = generate_graph_configs()
    total_experiments = len(LICENSE_CONFIGS) * len(ALGORITHMS) * len(GRAPH_NODES) * len(graph_configs)
    current_experiment = 0

    algorithm_skip_flags = {}

    for license_config in LICENSE_CONFIGS:
        for algorithm, algorithm_name in ALGORITHMS:
            for nodes in GRAPH_NODES:
                for graph_config in graph_configs:
                    graph_type = graph_config["type"]
                    skip_key = f"{algorithm_name}_{graph_type}_{graph_config}_{nodes}"

                    if skip_key not in algorithm_skip_flags:
                        algorithm_skip_flags[skip_key] = False

                    if algorithm_skip_flags[skip_key]:
                        continue

                    current_experiment += 1
                    print(f"Experiment {current_experiment}/{total_experiments}: {algorithm_name} on {graph_type} graph ({nodes} nodes) with {license_config}")

                    try:
                        generator = GraphGeneratorFactory.get_generator(graph_type)
                        kwargs = {"seed": GRAPH_SEED}

                        if graph_config["k"] is not None:
                            kwargs["k"] = graph_config["k"]
                        if graph_config["p"] is not None:
                            kwargs["p"] = graph_config["p"]
                        if graph_config["m"] is not None:
                            kwargs["m"] = graph_config["m"]

                        graph = generator(n_nodes=nodes, **kwargs)
                        license_types = LicenseConfigFactory.get_config(license_config)

                        solution, execution_time = run_algorithm_with_timeout(algorithm, graph, license_types, MAX_EXECUTION_TIME)

                        if solution is None:
                            if execution_time == MAX_EXECUTION_TIME:
                                print(f"Timeout for {algorithm_name} - skipping larger graphs for this algorithm+graph_type combination")
                                for larger_nodes in GRAPH_NODES:
                                    if larger_nodes >= nodes:
                                        larger_skip_key = f"{algorithm_name}_{graph_type}_{graph_config}_{larger_nodes}"
                                        algorithm_skip_flags[larger_skip_key] = True
                            continue

                        avg_degree = sum(dict(graph.degree()).values()) / len(graph.nodes())

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
                            "seed": GRAPH_SEED,
                        }

                        csv_writer.write_result(result)
                        print(f"  Result: cost={round(solution.total_cost, 2)}, time={execution_time:.3f}s, groups={len(solution.groups)}")

                    except Exception as e:
                        print(f"  Error: {e}")

    print(f"Benchmark completed! Results saved to: {csv_writer.get_csv_path()}")


if __name__ == "__main__":
    main()
