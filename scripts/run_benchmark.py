# scripts/run_benchmark.py
# python 3.13+
# Benchmark: sizes 50..5000 step 50, timeout per (algo, graph, size). CSV per algorithm and license. No images.

import os
import sys
import traceback
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Iterable, Tuple

import multiprocessing as mp
from multiprocessing.connection import Connection
import networkx as nx

from src.license_config import LicenseConfigFactory
from src.solution_validator import SolutionValidator
from src import Solution, LicenseType, Algorithm
from scripts._common import (
    RunResult,
    build_paths,
    ensure_dir,
    generate_graph,
)

from src import algorithms  # class lookup in child process


# ===== CONFIG =====

RUN_ID: str | None = None

# wiele generatorów w jednej sesji
GRAPH_NAMES: List[str] = [
    "random",
    "scale_free",
    "small_world",
    # "complete",
    # "star",
    # "path",
    # "cycle",
    # "tree",
]

# per generator opcjonalne nadpisania
GRAPH_PARAMS_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # "random": {"p": 0.08, "seed": 42},
    # "scale_free": {"m": 3, "seed": 42},
    # "small_world": {"k": 8, "p": 0.1, "seed": 42},
    # "tree": {"seed": 123},
}

# sizes: 10 → 5000 step 20
SIZES: List[int] = list(range(10, 1001, 20))

# wiele konfiguracji licencji
LICENSE_CONFIG_NAMES: List[str] = [
    "spotify",
    "duolingo_super",
    "roman_domination",
]

ALGORITHMS: List[str] = [
    "ILPSolver",
    "GreedyAlgorithm",
    "TabuSearch",
    "SimulatedAnnealing",
    "GeneticAlgorithm",
    "AntColonyOptimization",
    # "TreeDynamicProgramming",
    # "NaiveAlgorithm",
    "DominatingSetAlgorithm",
    "RandomizedAlgorithm",
]

TIMEOUT_SECONDS = 90  # per-run timeout
PRINT_ISSUE_LIMIT: int | None = 5

# ===== END CONFIG =====


# ===== PARAM RESOLVER =====

_GRAPH_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "random": {"p": 0.10, "seed": 42},
    "scale_free": {"m": 2, "seed": 42},
    "small_world": {"k": 6, "p": 0.05, "seed": 42},
    "complete": {},
    "star": {},
    "path": {},
    "cycle": {},
    "tree": {"seed": 42},
}


def resolve_graph_params(name: str, n_nodes: int, overrides: Dict[str, Any] | None) -> Dict[str, Any]:
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
        p = {k: v for k, v in p.items() if k in {"seed"}}

    elif name == "tree":
        pass

    return p


# ===== SOLVER HELPERS =====


def _solve_child(
    algo_class_name: str,
    graph: nx.Graph,
    license_types: List[LicenseType],
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
    license_types: List[LicenseType],
    timeout_s: int,
) -> Tuple[str, float, Solution] | None:
    parent_conn, child_conn = mp.Pipe(duplex=False)
    p = mp.Process(target=_solve_child, args=(algo_class_name, graph, license_types, child_conn))
    p.start()
    child_conn.close()
    try:
        if parent_conn.poll(timeout_s):
            ok, payload = parent_conn.recv()
            if ok:
                algo_name, elapsed_ms, solution = payload  # type: ignore[misc]
                return algo_name, float(elapsed_ms), solution
            else:
                raise RuntimeError(f"solver error: {payload}")
        else:
            p.terminate()
            p.join(1)
            return None
    finally:
        try:
            parent_conn.close()
        except Exception:
            pass
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
            writer.writerow(d)  # type: ignore[union-attr]
    return out_path


def _fmt_ms(ms: float) -> str:
    return f"{ms:.2f} ms"


# ===== MAIN =====


def main() -> None:
    mp.set_start_method("spawn", force=True)

    run_id = RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")
    _, _graphs_dir, csv_dir = build_paths(run_id)
    ensure_dir(csv_dir)

    missing = [n for n in ALGORITHMS if not hasattr(algorithms, n)]
    if missing:
        avail = ", ".join(getattr(algorithms, "__all__", []))
        print(f"[ERROR] unknown algorithms: {', '.join(missing)}; available: {avail}", file=sys.stderr)
        sys.exit(2)

    step = SIZES[1] - SIZES[0] if len(SIZES) >= 2 else 0
    print(f"Benchmark run_id={run_id}")
    print(f"Graphs={GRAPH_NAMES}")
    print(f"Sizes={SIZES[0]}..{SIZES[-1]} step={step}")
    print(f"Param overrides={GRAPH_PARAMS_OVERRIDES or '{}'}")
    print(f"Algorithms={ALGORITHMS}")
    print(f"Timeout={TIMEOUT_SECONDS}s\n")

    # pętla po konfiguracjach licencji
    for license_name in LICENSE_CONFIG_NAMES:
        try:
            license_types = LicenseConfigFactory.get_config(license_name)
        except Exception as e:
            print(f"[ERROR] license config failed: {license_name}: {e}", file=sys.stderr)
            traceback.print_exc(limit=10, file=sys.stderr)
            continue

        print(f"=== LICENSE={license_name} ===")

        # CSV per algorithm, wiersze ze wszystkich grafów
        for algo_name in ALGORITHMS:
            print(f"-- algo={algo_name} --")
            rows_all_graphs: List[RunResult] = []
            total_timeouts = total_failures = total_successes = 0

            for graph_name in GRAPH_NAMES:
                print(f"[graph={graph_name}]")
                timeouts = failures = successes = 0

                for n in SIZES:
                    params = resolve_graph_params(
                        graph_name,
                        n,
                        GRAPH_PARAMS_OVERRIDES.get(graph_name),
                    )

                    try:
                        G = generate_graph(graph_name, n, params)
                    except Exception as e:
                        failures += 1
                        print(f"[{license_name}][{algo_name}][{graph_name}] n={n} GEN-ERROR: {e}", file=sys.stderr)
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
                            )
                        )
                        continue

                    try:
                        solved = solve_with_timeout(algo_name, G, license_types, TIMEOUT_SECONDS)
                    except Exception as e:
                        failures += 1
                        print(f"[{license_name}][{algo_name}][{graph_name}] n={n} SOLVER-ERROR: {e}", file=sys.stderr)
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
                            )
                        )
                        continue

                    if solved is None:
                        timeouts += 1
                        print(f"[{license_name}][{algo_name}][{graph_name}] n={n} TIMEOUT {TIMEOUT_SECONDS}s → stop sizes for this graph")
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
                            )
                        )
                        break

                    solved_algo_name, elapsed_ms, solution = solved
                    ok, issues = SolutionValidator(debug=False).validate(solution, G)

                    if not ok and PRINT_ISSUE_LIMIT is not None:
                        print(f"[{license_name}][{algo_name}][{graph_name}] n={n} VALIDATION {len(issues)} issue(s)", file=sys.stderr)
                        for i in issues[:PRINT_ISSUE_LIMIT]:
                            print(f"  - {i.code}: {i.msg}", file=sys.stderr)
                        if len(issues) > PRINT_ISSUE_LIMIT:
                            print(f"  ... {len(issues) - PRINT_ISSUE_LIMIT} more", file=sys.stderr)

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
                        )
                    )
                    successes += 1
                    print(
                        f"[{license_name}][{algo_name}][{graph_name}] n={n:4d} | edges={G.number_of_edges():6d} | "
                        f"time={_fmt_ms(elapsed_ms):>10} | "
                        f"cost={'nan' if solution.total_cost != solution.total_cost else f'{solution.total_cost:.2f}':>10} | "
                        f"valid={'yes' if ok else 'no'}"
                    )

                print(f"[SUMMARY graph] {license_name} {algo_name} on {graph_name}: ok={successes} timeout={timeouts} fail={failures}\n")
                total_successes += successes
                total_timeouts += timeouts
                total_failures += failures

            out_csv = write_algo_csv(csv_dir, run_id, license_name, algo_name, rows_all_graphs)
            print(f"[CSV] {out_csv}")
            print(f"[SUMMARY algo] {license_name} {algo_name}: ok={total_successes} timeout={total_timeouts} fail={total_failures}\n")

    print("Benchmark done.")


if __name__ == "__main__":
    main()
