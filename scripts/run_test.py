import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List

from src.license_config import LicenseConfigFactory
from scripts._common import (
    RunResult,
    build_paths,
    generate_graph,
    instantiate_algorithms,
    run_once,
    write_csv,
)

# ===== CONFIG =====

RUN_ID: str | None = None

GRAPH_NAME = [
    # "random",
    # "scale_free",
    "small_world",
    # "complete",
    # "star",
    # "path",
    # "cycle",
    # "tree",
][0]

GRAPH_PARAMS: Dict[str, Any] = {
    "p": 0.05,
    "seed": 42,
}
N_NODES = 100

LICENSE_CONFIG_NAME = [
    # "duolingo_super",
    "spotify",
    # "roman_domination",
][0]

ALGORITHMS: List[str] = [
    "ILPSolver",
    "GreedyAlgorithm",
    "TabuSearch",
    "SimulatedAnnealing",
    # "GeneticAlgorithm",
    "AntColonyOptimization",
    # "TreeDynamicProgramming",
    "NaiveAlgorithm",
    # "DominatingSetAlgorithm",
    # "RandomizedAlgorithm",
]

PRINT_ISSUE_LIMIT: int | None = 20

# ===== END CONFIG =====


def main() -> None:
    run_id = RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")
    _, graphs_dir, csv_dir = build_paths(run_id)

    # graph
    try:
        graph = generate_graph(GRAPH_NAME, N_NODES, GRAPH_PARAMS)
    except Exception as e:
        print(f"[ERROR] graph generation failed: {GRAPH_NAME}: {e}", file=sys.stderr)
        traceback.print_exc(limit=10, file=sys.stderr)
        sys.exit(2)

    # licenses
    try:
        license_types = LicenseConfigFactory.get_config(LICENSE_CONFIG_NAME)
    except Exception as e:
        print(f"[ERROR] license config failed: {LICENSE_CONFIG_NAME}: {e}", file=sys.stderr)
        traceback.print_exc(limit=10, file=sys.stderr)
        sys.exit(2)

    # algorithms
    try:
        algos = instantiate_algorithms(ALGORITHMS)
    except Exception as e:
        print(f"[ERROR] algorithm setup failed: {e}", file=sys.stderr)
        traceback.print_exc(limit=10, file=sys.stderr)
        sys.exit(2)

    # run
    results: List[RunResult] = []
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
            }
        )
        results.append(r)

    # persist + summary with ranking
    write_csv(csv_dir, run_id, results)


if __name__ == "__main__":
    main()
