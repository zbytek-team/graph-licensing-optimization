import contextlib
import multiprocessing as mp
import os
import sys
import traceback
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime
from multiprocessing.connection import Connection
from typing import Any

import networkx as nx

from glopt import algorithms
from glopt.core import Algorithm, LicenseType, RunResult, Solution, generate_graph
from glopt.core.solution_validator import SolutionValidator
from glopt.io import build_paths, ensure_dir
from glopt.license_config import LicenseConfigFactory

RUN_ID: str | None = None


GRAPH_NAMES: list[str] = [
    "random",
    "scale_free",
    "small_world",
]


GRAPH_PARAMS_OVERRIDES: dict[str, dict[str, Any]] = {}


SIZES: list[int] = list(range(10, 1001, 20))


LICENSE_CONFIG_NAMES: list[str] = [
    "spotify",
    "duolingo_super",
    "roman_domination",
]

ALGORITHMS: list[str] = [
    "ILPSolver",
    "GreedyAlgorithm",
    "TabuSearch",
    "SimulatedAnnealing",
    "GeneticAlgorithm",
    "AntColonyOptimization",
    "DominatingSetAlgorithm",
    "RandomizedAlgorithm",
]

TIMEOUT_SECONDS = 90
PRINT_ISSUE_LIMIT: int | None = 5


_GRAPH_DEFAULTS: dict[str, dict[str, Any]] = {
    "random": {"p": 0.10, "seed": 42},
    "scale_free": {"m": 2, "seed": 42},
    "small_world": {"k": 6, "p": 0.05, "seed": 42},
    "complete": {},
    "star": {},
    "path": {},
    "cycle": {},
    "tree": {"seed": 42},
}


def resolve_graph_params(name: str, n_nodes: int, overrides: dict[str, Any] | None) -> dict[str, Any]:
    p = dict(_GRAPH_DEFAULTS.get(name, {}))
    if overrides:
        p.update(overrides)

    if name == "random":
        p["p"] = max(0.0, min(1.0, float(p.get("p", 0.10))))

    elif name == "scale_free":
        m = int(p.get("m", 2))
        m = max(1, min(m, max(1, n_nodes - 1)))
        p["m"] = m

    elif name == "small_world":
        k = int(p.get("k", 6))
        if n_nodes > 2:
            k = max(2, min(k, n_nodes - 1))
            if k % 2 == 1:
                k = k + 1 if k + 1 < n_nodes else k - 1
        else:
            k = 2
        p["k"] = k
        p["p"] = max(0.0, min(1.0, float(p.get("p", 0.05))))

    elif name in {"complete", "star", "path", "cycle"}:
        p = {k: v for k, v in p.items() if k == "seed"}

    elif name == "tree":
        pass

    return p


def _solve_child(
    algo_class_name: str,
    graph: nx.Graph,
    license_types: list[LicenseType],
    conn: Connection,
) -> None:
    from time import perf_counter

    try:
        cls = getattr(algorithms, algo_class_name)
        algo: Algorithm = cls()
        t0 = perf_counter()
        solution: Solution = algo.solve(graph=graph, license_types=license_types)
        elapsed_ms = (perf_counter() - t0) * 1000.0
        conn.send((True, (algo.name, elapsed_ms, solution)))
    except Exception as e:
        conn.send((False, repr(e)))
    finally:
        conn.close()


def solve_with_timeout(
    algo_class_name: str,
    graph: nx.Graph,
    license_types: list[LicenseType],
    timeout_s: int,
) -> tuple[str, float, Solution] | None:
    parent_conn, child_conn = mp.Pipe(duplex=False)
    p = mp.Process(target=_solve_child, args=(algo_class_name, graph, license_types, child_conn))
    p.start()
    child_conn.close()
    try:
        if parent_conn.poll(timeout_s):
            ok, payload = parent_conn.recv()
            if ok:
                algo_name, elapsed_ms, solution = payload
                return algo_name, float(elapsed_ms), solution
            msg = f"solver error: {payload}"
            raise RuntimeError(msg)
        p.terminate()
        p.join(1)
        return None
    finally:
        with contextlib.suppress(Exception):
            parent_conn.close()
        if p.is_alive():
            p.terminate()
        p.join()


def write_algo_csv(csv_dir: str, run_id: str, license_name: str, algo_name: str, rows: Iterable[RunResult]) -> str:
    safe_lic = "".join(c if c.isalnum() or c in "-_" else "_" for c in license_name)
    out_path = os.path.join(csv_dir, f"{run_id}_{safe_lic}_{algo_name}.csv")
    first = True
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        import csv

        writer = None
        for r in rows:
            d = asdict(r)
            if first:
                writer = csv.DictWriter(f, fieldnames=list(d.keys()))
                writer.writeheader()
                first = False
            writer.writerow(d)
    return out_path


# _fmt_ms was unused; removed


def main() -> None:
    mp.set_start_method("spawn", force=True)

    run_id = RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")
    _, _graphs_dir, csv_dir = build_paths(run_id)
    ensure_dir(csv_dir)

    missing = [n for n in ALGORITHMS if not hasattr(algorithms, n)]
    if missing:
        ", ".join(getattr(algorithms, "__all__", []))
        sys.exit(2)

    SIZES[1] - SIZES[0] if len(SIZES) >= 2 else 0

    for license_name in LICENSE_CONFIG_NAMES:
        try:
            license_types = LicenseConfigFactory.get_config(license_name)
        except Exception:
            traceback.print_exc(limit=10, file=sys.stderr)
            continue


        for algo_name in ALGORITHMS:
            rows_all_graphs: list[RunResult] = []

            for graph_name in GRAPH_NAMES:
                timeouts = failures = successes = 0

                for n in SIZES:
                    params = resolve_graph_params(
                        graph_name,
                        n,
                        GRAPH_PARAMS_OVERRIDES.get(graph_name),
                    )

                    try:
                        G = generate_graph(graph_name, n, params)
                    except Exception:
                        failures += 1
                        rows_all_graphs.append(
                            RunResult(
                                run_id=run_id,
                                algorithm=algo_name,
                                graph=graph_name,
                                n_nodes=n,
                                n_edges=0,
                                graph_params=str(params),
                                license_config=license_name,
                                total_cost=float("nan"),
                                time_ms=0.0,
                                valid=False,
                                issues=1,
                                image_path="",
                                notes="graph_gen_error",
                            ),
                        )
                        continue

                    try:
                        solved = solve_with_timeout(algo_name, G, license_types, TIMEOUT_SECONDS)
                    except Exception:
                        failures += 1
                        rows_all_graphs.append(
                            RunResult(
                                run_id=run_id,
                                algorithm=algo_name,
                                graph=graph_name,
                                n_nodes=n,
                                n_edges=G.number_of_edges(),
                                graph_params=str(params),
                                license_config=license_name,
                                total_cost=float("nan"),
                                time_ms=0.0,
                                valid=False,
                                issues=1,
                                image_path="",
                                notes="solver_error",
                            ),
                        )
                        continue

                    if solved is None:
                        timeouts += 1
                        rows_all_graphs.append(
                            RunResult(
                                run_id=run_id,
                                algorithm=algo_name,
                                graph=graph_name,
                                n_nodes=n,
                                n_edges=G.number_of_edges(),
                                graph_params=str(params),
                                license_config=license_name,
                                total_cost=float("nan"),
                                time_ms=float(TIMEOUT_SECONDS * 1000),
                                valid=False,
                                issues=0,
                                image_path="",
                                notes="timeout",
                            ),
                        )
                        break

                    solved_algo_name, elapsed_ms, solution = solved
                    ok, issues = SolutionValidator(debug=False).validate(solution, G)

                    if not ok and PRINT_ISSUE_LIMIT is not None:
                        for _i in issues[:PRINT_ISSUE_LIMIT]:
                            pass
                        if len(issues) > PRINT_ISSUE_LIMIT:
                            pass

                    rows_all_graphs.append(
                        RunResult(
                            run_id=run_id,
                            algorithm=solved_algo_name,
                            graph=graph_name,
                            n_nodes=n,
                            n_edges=G.number_of_edges(),
                            graph_params=str(params),
                            license_config=license_name,
                            total_cost=float(solution.total_cost),
                            time_ms=float(elapsed_ms),
                            valid=ok,
                            issues=len(issues),
                            image_path="",
                            notes="" if ok else "; ".join(f"{i.code}" for i in issues[:5]),
                        ),
                    )
                    successes += 1

                # summarize per-graph loop only; aggregate totals were unused

            write_algo_csv(csv_dir, run_id, license_name, algo_name, rows_all_graphs)



if __name__ == "__main__":
    main()
