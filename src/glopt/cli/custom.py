from __future__ import annotations

from datetime import datetime
from typing import Any

from glopt.core import RunResult, generate_graph, instantiate_algorithms, run_once
from glopt.io import build_paths, write_csv
from glopt.license_config import LicenseConfigFactory
from glopt.logging_config import get_logger, log_run_banner, log_run_footer, setup_logging

# ==============================================
# Configuration (no CLI/env)
# ==============================================
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
    run_id = (RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")) + "_custom"
    _, graphs_dir, csv_dir = build_paths(run_id)
    setup_logging(run_id=run_id)
    logger = get_logger(__name__)
    log_run_banner(
        logger,
        title="glopt custom run",
        params={
            "run_id": run_id,
            "graph": f"{GRAPH_NAME} n={N_NODES} params={GRAPH_PARAMS}",
            "license": LICENSE_CONFIG_NAME,
            "algorithms": ", ".join(ALGORITHMS),
        },
    )

    graph = generate_graph(GRAPH_NAME, N_NODES, GRAPH_PARAMS)
    license_types = LicenseConfigFactory.get_config(LICENSE_CONFIG_NAME)
    algos = instantiate_algorithms(ALGORITHMS)

    results: list[RunResult] = []
    for algo in algos:
        logger.info("-> running %s...", algo.name)
        r = run_once(
            algo=algo,
            graph=graph,
            license_types=license_types,
            run_id=run_id,
            graphs_dir=graphs_dir,
            print_issue_limit=10,
        )
        logger.info(
            "   done: cost=%.2f time_ms=%.2f valid=%s issues=%d",
            r.total_cost,
            r.time_ms,
            r.valid,
            r.issues,
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

    # Summary
    from time import perf_counter as _pc
    # Note: we don't track a start time here; keeping footer consistent
    log_run_footer(
        logger,
        summary={
            "runs": len(results),
            "csv": csv_path,
        },
    )
    for r in results:
        logger.info("%s: cost=%.2f time_ms=%.2f valid=%s", r.algorithm, r.total_cost, r.time_ms, r.valid)
        if r.image_path:
            logger.info(" image: %s", r.image_path)


if __name__ == "__main__":
    main()
