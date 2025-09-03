from datetime import datetime
from typing import Any

from glopt import algorithms
from glopt.core import RunResult, generate_graph, instantiate_algorithms, run_once
from glopt.io import build_paths, ensure_dir, write_csv
from glopt.license_config import LicenseConfigFactory

# Configuration
N_NODES: int = 100
GRAPH_NAMES: list[str] = ["random", "scale_free", "small_world"]
DEFAULT_GRAPH_PARAMS: dict[str, dict[str, Any]] = {
    "random": {"p": 0.1, "seed": 42},
    "scale_free": {"m": 2, "seed": 42},
    "small_world": {"k": 4, "p": 0.1, "seed": 42},
}
LICENSE_CONFIGS: list[str] = ["spotify", "duolingo_super", "roman_domination"]
# Use only generally applicable algorithms here (avoid Naive/TreeDP on general graphs)
ALGORITHMS: list[str] = [
    "ILPSolver",
    "GreedyAlgorithm",
    "RandomizedAlgorithm",
    "DominatingSetAlgorithm",
    "AntColonyOptimization",
    "SimulatedAnnealing",
    "TabuSearch",
    "GeneticAlgorithm",
]


def main() -> int:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_all"
    _, graphs_dir_root, csv_dir = build_paths(run_id)

    print("== glopt all ==")
    print(f"run_id: {run_id}")
    print(f"graphs: {', '.join(GRAPH_NAMES)} n={N_NODES}")
    print(f"licenses: {', '.join(LICENSE_CONFIGS)}")
    print(f"algorithms: {', '.join(ALGORITHMS)}")

    results: list[RunResult] = []
    for graph_name in GRAPH_NAMES:
        params = DEFAULT_GRAPH_PARAMS.get(graph_name, {})
        graph = generate_graph(graph_name, N_NODES, params)

        for lic_name in LICENSE_CONFIGS:
            license_types = LicenseConfigFactory.get_config(lic_name)
            g_dir = f"{graphs_dir_root}/{graph_name}/{lic_name}"
            ensure_dir(g_dir)
            print(f"-> {graph_name} {lic_name}")

            for algo_name in ALGORITHMS:
                try:
                    algo = instantiate_algorithms([algo_name])[0]
                    print(f"   running {algo.name}...")
                    r = run_once(
                        algo=algo,
                        graph=graph,
                        license_types=license_types,
                        run_id=run_id,
                        graphs_dir=g_dir,
                        print_issue_limit=10,
                    )
                    print(f"     cost={r.total_cost:.2f} time_ms={r.time_ms:.2f} valid={r.valid} issues={r.issues}")
                    r = RunResult(
                        **{
                            **r.__dict__,
                            "graph": graph_name,
                            "graph_params": str(params),
                            "license_config": lic_name,
                        },
                    )
                    results.append(r)
                except Exception as e:
                    print(f"     skipped {algo_name}: {e}")

    csv_path = write_csv(csv_dir, run_id, results)
    print("== summary ==")
    print(f"runs: {len(results)}")
    print(f"csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
