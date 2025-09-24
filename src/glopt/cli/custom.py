from __future__ import annotations

from typing import Any

from glopt.cli.common import build_run_id, fmt_ms, print_banner, print_footer, print_stage, print_step
from glopt.core import RunResult, generate_graph, instantiate_algorithms, run_once
from glopt.io import build_paths, write_csv
from glopt.license_config import LicenseConfigFactory

RUN_ID: str | None = None
GRAPH_NAME: str = "small_world"
GRAPH_PARAMS: dict[str, Any] = {"seed": 1}
N_NODES: int = 30
LICENSE_CONFIG_NAME: str = "duolingo_super"
ALGORITHMS: list[str] = [
    "ILPSolver",
    # "RandomizedAlgorithm",
    # "GreedyAlgorithm",
    # "DominatingSetAlgorithm",
    # "AntColonyOptimization",
    # "SimulatedAnnealing",
    # "TabuSearch",
    # "GeneticAlgorithm",
]


def main() -> None:
    run_id = build_run_id("custom", RUN_ID)
    _, graphs_dir, csv_dir = build_paths(run_id)
    print_banner(
        "glopt custom run",
        {
            "run_id": run_id,
            "graph": f"{GRAPH_NAME}",
            "n_nodes": N_NODES,
            "graph_params": GRAPH_PARAMS,
            "license": LICENSE_CONFIG_NAME,
            "algorithms": ", ".join(ALGORITHMS),
        },
    )

    graph = generate_graph(GRAPH_NAME, N_NODES, GRAPH_PARAMS)
    license_types = LicenseConfigFactory.get_config(LICENSE_CONFIG_NAME)
    algos = instantiate_algorithms(ALGORITHMS)

    results: list[RunResult] = []
    for algo in algos:
        print_stage(f"algorithm {algo.name}")
        r = run_once(
            algo=algo,
            graph=graph,
            license_types=license_types,
            run_id=run_id,
            graphs_dir=graphs_dir,
            print_issue_limit=10,
        )
        print_step(
            "result",
            cost=f"{r.total_cost:.2f}",
            algo_ms=fmt_ms(r.time_ms),
            valid=r.valid,
            issues=r.issues,
            image=bool(r.image_path),
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

    csv_path = write_csv(csv_dir, run_id, results)

    print_footer({"runs": len(results), "csv": csv_path})
    for r in results:
        print(f"{r.algorithm}: cost={r.total_cost:.2f} time_ms={r.time_ms:.2f} valid={r.valid}")
        if r.image_path:
            print(f" image: {r.image_path}")


if __name__ == "__main__":
    main()
