from time import perf_counter
from typing import Any

from glopt.cli.common import build_run_id, fmt_ms, print_banner, print_footer, print_stage, print_step
from glopt.core import RunResult, generate_graph, instantiate_algorithms, run_once
from glopt.io import build_paths, ensure_dir, write_csv
from glopt.license_config import LicenseConfigFactory

N_NODES: int = 30
GRAPH_NAMES: list[str] = ["random", "scale_free", "small_world"]
DEFAULT_GRAPH_PARAMS: dict[str, dict[str, Any]] = {
    "random": {"p": 0.1, "seed": 42},
    "scale_free": {"m": 2, "seed": 42},
    "small_world": {"k": 4, "p": 0.1, "seed": 42},
}
LICENSE_CONFIGS: list[str] = ["spotify", "duolingo_super", "roman_domination"]
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


def main() -> None:
    run_id = build_run_id("all")
    _, graphs_dir_root, csv_dir = build_paths(run_id)
    print_banner(
        "glopt all",
        {
            "run_id": run_id,
            "graphs": ", ".join(GRAPH_NAMES),
            "n_nodes": N_NODES,
            "licenses": ", ".join(LICENSE_CONFIGS),
            "algorithms": ", ".join(ALGORITHMS),
        },
    )

    results: list[RunResult] = []
    for lic_name in LICENSE_CONFIGS:
        print_stage(f"license {lic_name}")
        license_types = LicenseConfigFactory.get_config(lic_name)
        for algo_name in ALGORITHMS:
            print_stage(f"algorithm {algo_name}")
            for graph_name in GRAPH_NAMES:
                params = DEFAULT_GRAPH_PARAMS.get(graph_name, {})
                print_step("graph", name=graph_name, n=N_NODES, params=str(params))
                t0 = perf_counter()
                graph = generate_graph(graph_name, N_NODES, params)
                gen_ms = (perf_counter() - t0) * 1000.0
                n_nodes, n_edges = graph.number_of_nodes(), graph.number_of_edges()
                try:
                    import networkx as _nx

                    n_comp = _nx.number_connected_components(graph)
                    avg_deg = (2.0 * n_edges / max(1, n_nodes)) if n_nodes else 0.0
                    print_step("generated", nodes=n_nodes, edges=n_edges, comps=n_comp, avg_deg=f"{avg_deg:.2f}", time=fmt_ms(gen_ms))
                except Exception:
                    print_step("generated", nodes=n_nodes, edges=n_edges, time=fmt_ms(gen_ms))

                g_dir = f"{graphs_dir_root}/{graph_name}/{lic_name}"
                ensure_dir(g_dir)
                try:
                    algo = instantiate_algorithms([algo_name])[0]
                    print_step("run", algo=algo.name)
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
                    print_step(
                        "result",
                        cost=f"{r.total_cost:.2f}",
                        algo_ms=fmt_ms(r.time_ms),
                        wall_ms=fmt_ms(wall_ms),
                        valid=r.valid,
                        issues=r.issues,
                        image=bool(r.image_path),
                    )
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
                    print_step("skipped", algo=algo_name, reason=str(e))

    csv_path = write_csv(csv_dir, run_id, results)
    print_footer({"runs": len(results), "csv": csv_path})
    return None


if __name__ == "__main__":
    main()
