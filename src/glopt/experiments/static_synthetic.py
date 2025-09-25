from __future__ import annotations

import json
import math
import multiprocessing as mp
import pickle
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import networkx as nx

from glopt.core import generate_graph, instantiate_algorithms
from glopt.core.algorithms.greedy import GreedyAlgorithm
from glopt.core.io import ensure_dir
from glopt.core.license_config import LicenseConfigFactory
from glopt.core.solution_validator import SolutionValidator
from glopt.experiments.common import build_run_id, print_footer

RUN_ID: str | None = None
GRAPH_NAMES: list[str] = ["scale_free"]
GRAPH_DEFAULTS: dict[str, dict[str, Any]] = {
    "random": {"p": 0.1, "seed": 42},
    "small_world": {"k": 6, "p": 0.05, "seed": 42},
    "scale_free": {"m": 2, "seed": 42},
}
SIZES_SMALL: list[int] = (
    list(range(10, 101, 10))
    + list(range(120, 201, 20))
    + list(range(250, 501, 50))
)
SIZES_LARGE: list[int] = []
SIZES: list[int] = SIZES_SMALL + SIZES_LARGE
SAMPLES_PER_SIZE: int = 1
REPEATS_PER_GRAPH: int = 1
TIMEOUT_SECONDS: float = 600.0
LICENSE_CONFIG_NAMES: list[str] = [
    "duolingo_super",
    "roman_domination",
]
DYNAMIC_ROMAN_PS: list[float] = [1.5, 2.5, 3.0]
LICENSE_CONFIG_NAMES.extend(
    [f"roman_p_{str(p).replace('.', '_')}" for p in DYNAMIC_ROMAN_PS]
)
DYNAMIC_DUO_PS: list[float] = [2.0, 3.0]
LICENSE_CONFIG_NAMES.extend(
    [f"duolingo_p_{str(p).replace('.', '_')}" for p in DYNAMIC_DUO_PS]
)
ALGORITHM_CLASSES: list[str] = [
    "ILPSolver",
    "RandomizedAlgorithm",
    "GreedyAlgorithm",
    "DominatingSetAlgorithm",
    "AntColonyOptimization",
    "SimulatedAnnealing",
    "TabuSearch",
    "GeneticAlgorithm",
]
GRAPH_CACHE_DIR: str = "data/graphs_cache"


def _adjust_params(name: str, n: int, base: dict[str, Any]) -> dict[str, Any]:
    p = dict(base)
    if name == "scale_free":
        m = int(p.get("m", 2))
        p["m"] = max(1, min(m, max(1, n - 1)))
    if name == "small_world":
        k = int(p.get("k", 6))
        if n <= 2:
            k = 0
        else:
            k = max(2, min(k, n - 1))
            if k % 2 == 1:
                if k + 1 < n:
                    k += 1
                else:
                    k -= 1
            k = max(0, min(k, n - 1))
        p["k"] = k
    return p


def _write_row(csv_path: Path, row: dict[str, object]) -> None:
    import csv as _csv

    is_new = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)


def _json_dumps(obj: Any) -> str:
    import json as _json

    try:
        return _json.dumps(obj, ensure_ascii=False)
    except Exception:
        return "{}"


def _cache_paths(
    cache_dir: str, gname: str, n: int, sample: int
) -> tuple[Path, Path]:
    base = Path(cache_dir) / gname / f"n{n:04d}"
    gpath = base / f"s{sample}.gpickle"
    mpath = gpath.with_suffix(".json")
    return (gpath, mpath)


def _ensure_cache_for_all() -> None:
    ensure_dir(GRAPH_CACHE_DIR)
    created = 0
    for gname in GRAPH_NAMES:
        for n in SIZES:
            base_params = GRAPH_DEFAULTS.get(gname, {})
            base_params = _adjust_params(gname, n, base_params)
            for s_idx in range(SAMPLES_PER_SIZE):
                gpath, mpath = _cache_paths(GRAPH_CACHE_DIR, gname, n, s_idx)
                if gpath.exists() and mpath.exists():
                    continue
                seed = int((base_params.get("seed", 42) or 42) + s_idx * 1009)
                params = dict(base_params)
                params["seed"] = seed
                G = generate_graph(gname, n, params)
                Path(gpath).parent.mkdir(parents=True, exist_ok=True)
                with gpath.open("wb") as f:
                    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
                meta = {
                    "type": gname,
                    "n": n,
                    "params": params,
                    "sample": s_idx,
                }
                with mpath.open("w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False)
                created += 1
    if created:
        print(
            f"Graph cache ready: generated {created} graph file(s) in " \
            f"{GRAPH_CACHE_DIR}"
        )
    else:
        print(f"Graph cache ready: using existing data in {GRAPH_CACHE_DIR}")


def _load_cached_graph(
    gname: str, n: int, sample_idx: int
) -> tuple[nx.Graph, dict[str, Any]]:
    gpath, mpath = _cache_paths(GRAPH_CACHE_DIR, gname, n, sample_idx)
    with gpath.open("rb") as f:
        G: nx.Graph = pickle.load(f)
    params: dict[str, Any] = {}
    if mpath.exists():
        try:
            with mpath.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            params = dict(meta.get("params", {}))
        except Exception:
            params = {}
    return (G, params)


def _worker_solve(
    algo_name: str,
    graph: nx.Graph,
    license_config: str,
    seed: int,
    conn,
) -> None:
    try:
        validator = SolutionValidator(debug=False)
        algo = instantiate_algorithms([algo_name])[0]
        lts = LicenseConfigFactory.get_config(license_config)
        kwargs: dict[str, Any] = {"seed": seed}
        warm_names = {
            "GeneticAlgorithm",
            "SimulatedAnnealing",
            "TabuSearch",
            "AntColonyOptimization",
        }
        if algo_name in warm_names:
            greedy_sol = GreedyAlgorithm().solve(graph, lts)
            kwargs["initial_solution"] = greedy_sol
        t0 = perf_counter()
        sol = algo.solve(graph, lts, **kwargs)
        elapsed_ms = (perf_counter() - t0) * 1000.0
        ok, issues = validator.validate(sol, graph)
        sizes = [g.size for g in sol.groups]
        sizes_sorted = sorted(sizes)
        groups = len(sizes)
        mean_sz = sum(sizes) / groups if groups else 0.0
        if groups:
            mid = groups // 2
            if groups % 2 == 1:
                median_sz = float(sizes_sorted[mid])
            else:
                median_sz = (sizes_sorted[mid - 1] + sizes_sorted[mid]) / 2.0
            p90 = float(sizes_sorted[min(groups - 1, int(0.9 * (groups - 1)))])
        else:
            median_sz = 0.0
            p90 = 0.0
        lic_counts = Counter(g.license_type.name for g in sol.groups)
        try:
            params_json = _json_dumps(
                {
                    k: v
                    for k, v in vars(algo).items()
                    if isinstance(v, (int, float, str, bool))
                }
            )
        except Exception:
            params_json = "{}"
        res = {
            "success": True,
            "total_cost": float(sol.total_cost),
            "time_ms": float(elapsed_ms),
            "valid": bool(ok),
            "issues": int(len(issues)),
            "groups": int(groups),
            "group_size_mean": float(mean_sz),
            "group_size_median": float(median_sz),
            "group_size_p90": float(p90),
            "license_counts_json": _json_dumps(lic_counts),
            "cost_per_node": float(sol.total_cost)
            / max(1, graph.number_of_nodes()),
            "algo_params_json": params_json,
            "warm_start": bool(algo_name in warm_names),
        }
    except Exception as e:
        res = {"success": False, "error": str(e)}
    try:
        conn.send(res)
    finally:
        conn.close()


def _run_one(
    algo_name: str, graph: nx.Graph, license_config: str, seed: int
) -> tuple[dict[str, object], bool]:
    parent_conn, child_conn = mp.Pipe(duplex=False)
    p = mp.Process(
        target=_worker_solve,
        args=(algo_name, graph, license_config, seed, child_conn),
    )
    p.start()
    timed_out = False
    res: dict[str, object]
    if parent_conn.poll(TIMEOUT_SECONDS):
        try:
            msg = parent_conn.recv()
        except EOFError:
            msg = {"success": False, "error": "no-data"}
        p.join()
        if msg.get("success"):
            res = {
                "total_cost": float(msg.get("total_cost", float("nan"))),
                "time_ms": float(msg.get("time_ms", 0.0)),
                "valid": bool(msg.get("valid", False)),
                "issues": int(msg.get("issues", 0)),
                "groups": int(msg.get("groups", 0)),
                "group_size_mean": float(msg.get("group_size_mean", 0.0)),
                "group_size_median": float(msg.get("group_size_median", 0.0)),
                "group_size_p90": float(msg.get("group_size_p90", 0.0)),
                "license_counts_json": str(
                    msg.get("license_counts_json", "{}")
                ),
                "algo_params_json": str(msg.get("algo_params_json", "{}")),
                "warm_start": bool(msg.get("warm_start", False)),
                "cost_per_node": float(msg.get("cost_per_node", 0.0)),
                "notes": "",
            }
        else:
            res = {
                "total_cost": float("nan"),
                "time_ms": 0.0,
                "valid": False,
                "issues": 0,
                "groups": 0,
                "group_size_mean": 0.0,
                "group_size_median": 0.0,
                "group_size_p90": 0.0,
                "license_counts_json": "{}",
                "notes": "error",
                "warm_start": False,
            }
    else:
        timed_out = True
        try:
            p.terminate()
        finally:
            p.join()
        res = {
            "total_cost": float("nan"),
            "time_ms": float(TIMEOUT_SECONDS * 1000.0),
            "valid": False,
            "issues": 0,
            "groups": 0,
            "group_size_mean": 0.0,
            "group_size_median": 0.0,
            "group_size_p90": 0.0,
            "license_counts_json": "{}",
            "notes": "timeout",
            "warm_start": False,
        }
    return (res, timed_out or res.get("notes") == "error")


def main() -> None:
    run_id = build_run_id("benchmark", RUN_ID)
    base = Path("runs") / run_id
    csv_dir = base / "csv"
    ensure_dir(str(csv_dir))
    out_path = csv_dir / f"{run_id}.csv"
    from time import perf_counter as _pc

    _t0 = _pc()
    graphs_summary = ", ".join(GRAPH_NAMES) if GRAPH_NAMES else "none"
    if SIZES:
        sizes_summary = f"{min(SIZES)}..{max(SIZES)} ({len(SIZES)} sizes)"
    else:
        sizes_summary = "no sizes configured"
    licenses_summary = (
        ", ".join(LICENSE_CONFIG_NAMES) if LICENSE_CONFIG_NAMES else "none"
    )
    algorithms_summary = (
        ", ".join(ALGORITHM_CLASSES) if ALGORITHM_CLASSES else "none"
    )
    print(f"Starting glopt benchmark run {run_id}")
    print(f"Run directory: {base}")
    print(f"Results file: {out_path}")
    print(f"Graphs: {graphs_summary}")
    print(f"Sizes: {sizes_summary}")
    print(f"Samples per size: {SAMPLES_PER_SIZE}")
    print(f"Repeats per graph: {REPEATS_PER_GRAPH}")
    print(f"License configurations: {licenses_summary}")
    print(f"Algorithms: {algorithms_summary}")
    print(f"Timeout per run: {TIMEOUT_SECONDS:.0f}s")
    print("Graph cache warm-up started")
    _ensure_cache_for_all()
    print("Graph cache warm-up finished")
    total_runs = 0
    timeout_runs = 0
    error_runs = 0
    valid_runs = 0
    warm_start_runs = 0
    combination_count = 0
    for lic_name in LICENSE_CONFIG_NAMES:
        for algo_name in ALGORITHM_CLASSES:
            combination_count += 1
            print(f"Processing license {lic_name} with algorithm {algo_name}")
            for gname in GRAPH_NAMES:
                stop_sizes = False
                for n in SIZES:
                    if stop_sizes:
                        break
                    for s_idx in range(SAMPLES_PER_SIZE):
                        G, params = _load_cached_graph(gname, n, s_idx)
                        graph_seed = int(params.get("seed", 0) or 0)
                        n_nodes = G.number_of_nodes()
                        n_edges = G.number_of_edges()
                        density = (
                            2.0 * n_edges / (n_nodes * (n_nodes - 1))
                            if n_nodes > 1
                            else 0.0
                        )
                        avg_deg = (
                            2.0 * n_edges / n_nodes if n_nodes > 0 else 0.0
                        )
                        clustering = (
                            nx.average_clustering(G)
                            if n_nodes > 1 and n_nodes <= 1500
                            else float("nan")
                        )
                        components = nx.number_connected_components(G)
                        over_here = False
                        for rep in range(REPEATS_PER_GRAPH):
                            algo_seed = 12345 + s_idx * 1000 + rep
                            result, is_over = _run_one(
                                algo_name, G, lic_name, algo_seed
                            )
                            row = {
                                "run_id": run_id,
                                "algorithm": algo_name,
                                "graph": gname,
                                "n_nodes": n_nodes,
                                "n_edges": n_edges,
                                "graph_params": str(params),
                                "license_config": lic_name,
                                "rep": rep,
                                "seed": algo_seed,
                                "sample": s_idx,
                                "graph_seed": int(graph_seed),
                                "density": float(density),
                                "avg_degree": float(avg_deg),
                                "clustering": float(clustering),
                                "components": int(components),
                                "image_path": "",
                                **result,
                            }
                            _write_row(out_path, row)
                            total_runs += 1
                            notes = row.get("notes") or ""
                            status = "OK"
                            if notes == "timeout":
                                status = "TIMEOUT"
                                timeout_runs += 1
                            elif notes == "error":
                                status = "ERROR"
                                error_runs += 1
                            elif not row.get("valid"):
                                status = "INVALID"
                            else:
                                valid_runs += 1
                            if row.get("warm_start"):
                                warm_start_runs += 1
                            cost_raw = row.get("total_cost", float("nan"))
                            try:
                                cost_value = float(cost_raw)
                            except (TypeError, ValueError):
                                cost_value = float("nan")
                            cost_display = (
                                f"{cost_value:.2f}"
                                if math.isfinite(cost_value)
                                else "N/A"
                            )
                            time_raw = row.get("time_ms", 0.0)
                            try:
                                time_value = float(time_raw)
                            except (TypeError, ValueError):
                                time_value = 0.0
                            time_display = f"{time_value:.2f}"
                            cpn_value = row.get("cost_per_node")
                            if isinstance(
                                cpn_value, (int, float)
                            ) and math.isfinite(float(cpn_value)):
                                cost_per_node = f"{float(cpn_value):.4f}"
                            else:
                                cost_per_node = "N/A"
                            issues_value = row.get("issues", 0)
                            groups_value = row.get("groups", 0)
                            step_details = {
                                "license_config": lic_name,
                                "algorithm": algo_name,
                                "graph": gname,
                                "nodes": n_nodes,
                                "sample": s_idx,
                                "repeat": rep,
                                "cost": cost_display,
                                "cost_per_node": cost_per_node,
                                "time_ms": time_display,
                                "valid": row.get("valid"),
                                "issues": issues_value,
                                "groups": groups_value,
                                "status": status,
                                "warm_start": row.get("warm_start", False),
                            }
                            print(
                                f"Step n={n:<4} s={s_idx:<3} r={rep:<2} " \
                                f"lic={lic_name:<10} " \
                                f"algo={algo_name:<15} " \
                                f"on {gname:<12} " \
                                f"cost={cost_display:<8} " \
                                f"cpn={cost_per_node:<8} " \
                                f"time={time_display:<8}ms"
                            )
                            if notes:
                                step_details["note"] = notes
                            if is_over:
                                over_here = True
                        if over_here:
                            reason = row.get("notes") or "issue"
                            print(
                                f"Stopping larger sizes " \
                                f"for license {lic_name} " \
                                f"with algorithm {algo_name} " \
                                f"on graph {gname} " \
                                f"after {reason} at n={n}"
                            )
                            stop_sizes = True
                            break
    if out_path.exists():
        duration_s = _pc() - _t0
        successful_runs = total_runs - timeout_runs - error_runs
        invalid_runs = max(successful_runs - valid_runs, 0)
        print_footer(
            {
                "csv": out_path,
                "elapsed_sec": f"{duration_s:.2f}",
                "runs_total": total_runs,
                "runs_success": successful_runs,
                "runs_timeout": timeout_runs,
                "runs_error": error_runs,
                "valid_solutions": valid_runs,
                "invalid_solutions": invalid_runs,
                "warm_starts": warm_start_runs,
                "license_algorithm_pairs": combination_count,
            }
        )


if __name__ == "__main__":
    main()
