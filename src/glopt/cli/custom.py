from __future__ import annotations

from datetime import datetime
from typing import Any

from glopt.core import RunResult, generate_graph, instantiate_algorithms, run_once
from glopt.io import build_paths, write_csv
from glopt.license_config import LicenseConfigFactory

# Configuration
RUN_ID: str | None = None
GRAPH_NAME: str = "small_world"
GRAPH_PARAMS: dict[str, Any] = {"k": 4, "p": 0.1, "seed": 42}
N_NODES: int = 100
LICENSE_CONFIG_NAME: str = "spotify"
ALGORITHMS: list[str] = [
    "ILPSolver",
    # "NaiveAlgorithm",
    "RandomizedAlgorithm",
    "GreedyAlgorithm",
    "DominatingSetAlgorithm",
    "AntColonyOptimization",
    "SimulatedAnnealing",
    "TabuSearch",
    "GeneticAlgorithm",
    # "TreeDynamicProgramming",
]


def main() -> None:
    run_id = RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")
    _, graphs_dir, csv_dir = build_paths(run_id)

    # Print header config
    print("== glopt custom run ==")
    print(f"run_id: {run_id}")
    print(f"graph: {GRAPH_NAME} n={N_NODES} params={GRAPH_PARAMS}")
    print(f"license: {LICENSE_CONFIG_NAME}")
    print(f"algorithms: {', '.join(ALGORITHMS)}")

    graph = generate_graph(GRAPH_NAME, N_NODES, GRAPH_PARAMS)
    license_types = LicenseConfigFactory.get_config(LICENSE_CONFIG_NAME)
    algos = instantiate_algorithms(ALGORITHMS)

    results: list[RunResult] = []
    for algo in algos:
        print(f"-> running {algo.name}...")
        r = run_once(
            algo=algo,
            graph=graph,
            license_types=license_types,
            run_id=run_id,
            graphs_dir=graphs_dir,
            print_issue_limit=10,
        )
        print(f"   done: cost={r.total_cost:.2f} time_ms={r.time_ms:.2f} valid={r.valid} issues={r.issues}")
        r = RunResult(
            **{
                **r.__dict__,
                "graph": GRAPH_NAME,
                "graph_params": str(GRAPH_PARAMS),
                "license_config": LICENSE_CONFIG_NAME,
            },
        )
        results.append(r)

    csv_path = write_csv(csv_dir, run_id, results)

    # Summary
    print("== summary ==")
    for r in results:
        print(f"{r.algorithm}: cost={r.total_cost:.2f} time_ms={r.time_ms:.2f} valid={r.valid}")
        if r.image_path:
            print(f" image: {r.image_path}")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
