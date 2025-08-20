import os
import sys
from datetime import datetime
from typing import Any, Dict, List

from scripts._common import (
    RunResult,
    build_paths,
    ensure_dir,
    generate_graph,
    instantiate_algorithms,
    print_summary,
    run_once,
    write_csv,
)
from src import algorithms
from src.factories.graph_generator_factory import GraphGeneratorFactory
from src.factories.license_config_factory import LicenseConfigFactory

# ===== CONFIG =====

N_NODES = 50
DEFAULT_GRAPH_PARAMS: Dict[str, Dict[str, Any]] = {
    "random": {"p": 0.1, "seed": 42},
    "scale_free": {"m": 2, "seed": 42},
    "small_world": {"k": 4, "p": 0.1, "seed": 42},
    "complete": {},
    "star": {},
    "path": {},
    "cycle": {},
    "tree": {"seed": 42},
}

# ===== END CONFIG =====


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _, graphs_dir_root, csv_dir = build_paths(run_id)

    graph_names = list(GraphGeneratorFactory._GENERATORS.keys())
    license_configs = list(LicenseConfigFactory._CONFIGS.keys())
    algorithm_names = list(algorithms.__all__)

    results: List[RunResult] = []
    for graph_name in graph_names:
        params = DEFAULT_GRAPH_PARAMS.get(graph_name, {})
        try:
            graph = generate_graph(graph_name, N_NODES, params)
        except Exception as e:
            print(f"[ERROR] graph generation failed: {graph_name}: {e}", file=sys.stderr)
            continue
        for lic_name in license_configs:
            try:
                license_types = LicenseConfigFactory.get_config(lic_name)
            except Exception as e:
                print(f"[ERROR] license config failed: {lic_name}: {e}", file=sys.stderr)
                continue
            g_dir = os.path.join(graphs_dir_root, graph_name, lic_name)
            ensure_dir(g_dir)
            for algo_name in algorithm_names:
                try:
                    algo = instantiate_algorithms([algo_name])[0]
                except Exception as e:
                    print(f"[ERROR] algorithm setup failed: {algo_name}: {e}", file=sys.stderr)
                    continue
                r = run_once(
                    algo=algo,
                    graph=graph,
                    license_types=license_types,
                    run_id=run_id,
                    graphs_dir=g_dir,
                )
                r = RunResult(
                    **{
                        **r.__dict__,
                        "graph": graph_name,
                        "graph_params": str(params),
                        "license_config": lic_name,
                    }
                )
                results.append(r)

    csv_path = write_csv(csv_dir, run_id, results)
    print_summary(run_id, csv_path, results)


if __name__ == "__main__":
    main()
