from datetime import datetime
from time import perf_counter
from typing import Any

from glopt.core import RunResult, generate_graph, instantiate_algorithms, run_once
from glopt.io import build_paths, ensure_dir, write_csv
from glopt.license_config import LicenseConfigFactory

# Configuration
N_NODES: int = 30
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


def _format_license_types(license_types: list) -> str:
    try:
        parts = [f"{lt.name}(cost={lt.cost}, cap=[{lt.min_capacity}-{lt.max_capacity}], color={lt.color})" for lt in license_types]
        return ", ".join(parts)
    except Exception:
        return ", ".join(getattr(lt, "name", str(lt)) for lt in license_types)


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
        print("\n== graph ==")
        params = DEFAULT_GRAPH_PARAMS.get(graph_name, {})
        print(f"generating: {graph_name} n={N_NODES} params={params}")
        t0 = perf_counter()
        graph = generate_graph(graph_name, N_NODES, params)
        gen_ms = (perf_counter() - t0) * 1000.0
        n_nodes, n_edges = graph.number_of_nodes(), graph.number_of_edges()
        try:
            import networkx as _nx

            n_comp = _nx.number_connected_components(graph)
            avg_deg = (2.0 * n_edges / max(1, n_nodes)) if n_nodes else 0.0
            print(f"generated: {n_nodes} nodes, {n_edges} edges, comps={n_comp}, avg_deg={avg_deg:.2f} in {gen_ms:.1f} ms")
        except Exception:
            print(f"generated: {n_nodes} nodes, {n_edges} edges in {gen_ms:.1f} ms")

        for lic_name in LICENSE_CONFIGS:
            license_types = LicenseConfigFactory.get_config(lic_name)
            g_dir = f"{graphs_dir_root}/{graph_name}/{lic_name}"
            ensure_dir(g_dir)
            print(f"-> license: {lic_name} [{_format_license_types(license_types)}]")

            for algo_name in ALGORITHMS:
                try:
                    algo = instantiate_algorithms([algo_name])[0]
                    print(f"   running: {algo.name}")
                    t0 = perf_counter()
                    r = run_once(
                        algo=algo,
                        graph=graph,
                        license_types=license_types,
                        run_id=run_id,
                        graphs_dir=g_dir,
                        print_issue_limit=10,
                    )
                    wall_ms = (perf_counter() - t0) * 1000.0
                    print(f"     result: cost={r.total_cost:.2f} time_ms={r.time_ms:.2f} valid={r.valid} issues={r.issues} img={bool(r.image_path)}")
                    if abs(r.time_ms - wall_ms) > 5.0:
                        print(f"     note: wall={wall_ms:.2f} ms vs reported={r.time_ms:.2f} ms")
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
                    print(f"     skipped: {algo_name} -> {e}")

    csv_path = write_csv(csv_dir, run_id, results)
    print("== summary ==")
    print(f"runs: {len(results)}")
    print(f"csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
