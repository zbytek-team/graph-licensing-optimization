import sys
import traceback
from datetime import datetime
from typing import Any

from glopt.core import RunResult, generate_graph, instantiate_algorithms, run_once
from glopt.io import build_paths, write_csv
from glopt.license_config import LicenseConfigFactory

RUN_ID: str | None = None

GRAPH_NAME = [
    "small_world",
][0]

GRAPH_PARAMS: dict[str, Any] = {
    "p": 0.05,
    "seed": 42,
}
N_NODES = 100

LICENSE_CONFIG_NAME = [
    "spotify",
][0]

ALGORITHMS: list[str] = [
    "ILPSolver",
    "GreedyAlgorithm",
    "TabuSearch",
    "SimulatedAnnealing",
    "AntColonyOptimization",
    "NaiveAlgorithm",
]

PRINT_ISSUE_LIMIT: int | None = 20


def main() -> None:
    run_id = RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")
    _, graphs_dir, csv_dir = build_paths(run_id)

    try:
        graph = generate_graph(GRAPH_NAME, N_NODES, GRAPH_PARAMS)
    except Exception as e:
        print(f"[ERROR] graph generation failed: {GRAPH_NAME}: {e}", file=sys.stderr)
        traceback.print_exc(limit=10, file=sys.stderr)
        sys.exit(2)

    try:
        license_types = LicenseConfigFactory.get_config(LICENSE_CONFIG_NAME)
    except Exception as e:
        print(f"[ERROR] license config failed: {LICENSE_CONFIG_NAME}: {e}", file=sys.stderr)
        traceback.print_exc(limit=10, file=sys.stderr)
        sys.exit(2)

    try:
        algos = instantiate_algorithms(ALGORITHMS)
    except Exception as e:
        print(f"[ERROR] algorithm setup failed: {e}", file=sys.stderr)
        traceback.print_exc(limit=10, file=sys.stderr)
        sys.exit(2)

    results: list[RunResult] = []
    for algo in algos:
        r = run_once(
            algo=algo,
            graph=graph,
            license_types=license_types,
            run_id=run_id,
            graphs_dir=graphs_dir,
            print_issue_limit=PRINT_ISSUE_LIMIT,
        )
        r = RunResult(
            **{
                **r.__dict__,
                "graph": GRAPH_NAME,
                "graph_params": str(GRAPH_PARAMS),
                "license_config": LICENSE_CONFIG_NAME,
            },
        )
        results.append(r)

    write_csv(csv_dir, run_id, results)


if __name__ == "__main__":
    main()
