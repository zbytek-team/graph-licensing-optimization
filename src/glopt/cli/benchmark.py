from __future__ import annotations

import json
import multiprocessing as mp
import pickle
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import networkx as nx


from glopt.algorithms.greedy import GreedyAlgorithm
from glopt.core import generate_graph, instantiate_algorithms
from glopt.core.solution_validator import SolutionValidator
from glopt.io import ensure_dir
from glopt.license_config import LicenseConfigFactory
from glopt.logging_config import get_logger, log_run_banner, log_run_footer, setup_logging

# ==============================================
# Configuration (no CLI/env)
# ==============================================

# Optional custom run id suffix. If None, timestamp is used.
RUN_ID: str | None = None

# Graph families and default params
GRAPH_NAMES: list[str] = ["random", "small_world", "scale_free"]
GRAPH_DEFAULTS: dict[str, dict[str, Any]] = {
    "random": {"p": 0.10, "seed": 42},
    "small_world": {"k": 6, "p": 0.05, "seed": 42},
    "scale_free": {"m": 2, "seed": 42},
}

# Sizes: dense grid for small n, coarser for larger n
SIZES_SMALL: list[int] = list(range(20, 201, 20))
SIZES_LARGE: list[int] = [300, 400, 600, 800, 1000, 1500, 2000, 2500, 3000]
SIZES: list[int] = SIZES_SMALL + SIZES_LARGE

# Experiments: number of independent graph samples per (graph, n)
SAMPLES_PER_SIZE: int = 3  # increase for more robust averages

# Repeated runs of stochastic solvers on the same graph
REPEATS_PER_GRAPH: int = 2  # e.g., different algorithm seeds

# Per-run time budget (hard cap)
TIMEOUT_SECONDS: float = 60.0

# License configurations and algorithms under test
LICENSE_CONFIG_NAMES: list[str] = [
    "duolingo_super",           # real-world Duolingo Super (1 + family up to 6)
    "roman_domination",         # normalized Roman domination (group unbounded)
]
# Param sweeps: Roman (unbounded capacity) and Duolingo-style (cap 6)
DYNAMIC_ROMAN_PS: list[float] = [1.5, 2.5, 3.0]
LICENSE_CONFIG_NAMES.extend([f"roman_p_{str(p).replace('.', '_')}" for p in DYNAMIC_ROMAN_PS])
DYNAMIC_DUO_PS: list[float] = [2.0, 3.0]
LICENSE_CONFIG_NAMES.extend([f"duolingo_p_{str(p).replace('.', '_')}" for p in DYNAMIC_DUO_PS])
ALGORITHM_CLASSES: list[str] = [
    "ILPSolver",  # exact (small n only)
    "GreedyAlgorithm",
    "RandomizedAlgorithm",
    "DominatingSetAlgorithm",
    "AntColonyOptimization",
    "SimulatedAnnealing",
    "TabuSearch",
    "GeneticAlgorithm",
    # "NaiveAlgorithm",          # enable for very small n only
    # "TreeDynamicProgramming",  # only for trees
]

# Optional cap for ILP to avoid extreme runs (set to None to disable)
ILP_MAX_N: int | None = None

# Graph cache directory
GRAPH_CACHE_DIR: str = "data/graphs_cache"


def _adjust_params(name: str, n: int, base: dict[str, Any]) -> dict[str, Any]:
    p = dict(base)
    # scale_free: keep m reasonable vs n
    if name == "scale_free":
        m = int(p.get("m", 2))
        p["m"] = max(1, min(m, max(1, n - 1)))
    # small_world: enforce even k within [2, n-1]
    if name == "small_world":
        k = int(p.get("k", 6))
        if n > 2:
            k = max(2, min(k, n - 1))
            if k % 2 == 1:
                k = k + 1 if k + 1 < n else k - 1
        else:
            k = 2
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


def _cache_paths(cache_dir: str, gname: str, n: int, sample: int) -> tuple[Path, Path]:
    base = Path(cache_dir) / gname / f"n{n:04d}"
    gpath = base / f"s{sample}.gpickle"
    mpath = gpath.with_suffix(".json")
    return gpath, mpath


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
                meta = {"type": gname, "n": n, "params": params, "sample": s_idx}
                with mpath.open("w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False)
                created += 1
    logger = get_logger(__name__)
    if created:
        logger.info("graph cache: generated %d new graphs under %s", created, GRAPH_CACHE_DIR)
    else:
        logger.info("graph cache: up-to-date at %s", GRAPH_CACHE_DIR)


def _load_cached_graph(gname: str, n: int, sample_idx: int) -> tuple[nx.Graph, dict[str, Any]]:
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
    return G, params


def _worker_solve(algo_name: str, graph: nx.Graph, license_config: str, seed: int, conn) -> None:  # type: ignore[no-redef]
    """Child process: run solve once and return metrics via pipe."""
    try:
        validator = SolutionValidator(debug=False)
        algo = instantiate_algorithms([algo_name])[0]
        lts = LicenseConfigFactory.get_config(license_config)
        # No soft deadline: rely solely on hard kill at TIMEOUT_SECONDS
        kwargs: dict[str, Any] = {"seed": seed}
        # Warm-start metaheuristics with greedy
        warm_names = {
            "GeneticAlgorithm",
            "SimulatedAnnealing",
            "TabuSearch",
            "AntColonyOptimization",
        }
        if algo_name in warm_names:
            greedy_sol = GreedyAlgorithm().solve(graph, lts)
            kwargs["initial_solution"] = greedy_sol
        # No internal ILP time limit: enforce only external hard timeout
        t0 = perf_counter()
        sol = algo.solve(graph, lts, **kwargs)
        elapsed_ms = (perf_counter() - t0) * 1000.0

        ok, issues = validator.validate(sol, graph)
        sizes = [g.size for g in sol.groups]
        sizes_sorted = sorted(sizes)
        groups = len(sizes)
        mean_sz = (sum(sizes) / groups) if groups else 0.0
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
        # Serialize algo params (including ILP diagnostics if exposed)
        try:
            params_json = _json_dumps({k: v for k, v in vars(algo).items() if isinstance(v | (int, float, str, bool))})
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
            "cost_per_node": float(sol.total_cost) / max(1, graph.number_of_nodes()),
            "algo_params_json": params_json,
            "warm_start": bool(algo_name in warm_names),
        }
    except Exception as e:  # defensive: return error to parent instead of crashing worker
        res = {"success": False, "error": str(e)}
    try:
        conn.send(res)
    finally:
        conn.close()


def _run_one(
    algo_name: str,
    graph: nx.Graph,
    license_config: str,
    seed: int,
) -> tuple[dict[str, object], bool]:
    parent_conn, child_conn = mp.Pipe(duplex=False)
    p = mp.Process(target=_worker_solve, args=(algo_name, graph, license_config, seed, child_conn))
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
                "license_counts_json": str(msg.get("license_counts_json", "{}")),
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
            }
    else:
        # Timeout: kill child and report
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
        }

    return res, timed_out or (res.get("notes") == "error")


def main() -> None:
    run_id = (RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")) + "_benchmark"
    base = Path("runs") / run_id
    csv_dir = base / "csv"
    ensure_dir(str(csv_dir))
    out_path = csv_dir / f"{run_id}.csv"

    from time import perf_counter as _pc

    setup_logging(run_id=run_id)
    logger = get_logger(__name__)
    _t0 = _pc()
    log_run_banner(
        logger,
        title="glopt benchmark",
        params={
            "run_id": run_id,
            "graphs": ", ".join(GRAPH_NAMES),
            "sizes": f"{SIZES[0]}..{SIZES[-1]} ({len(SIZES)} points)",
            "samples/size": SAMPLES_PER_SIZE,
            "repeats/graph": REPEATS_PER_GRAPH,
            "licenses": ", ".join(LICENSE_CONFIG_NAMES),
            "algorithms": ", ".join(ALGORITHM_CLASSES),
            "timeout limit": f"{TIMEOUT_SECONDS:.0f}s (kills run and stops larger sizes)",
        },
    )
    logger.info("warming up graph cache …")
    _ensure_cache_for_all()

    for lic_name in LICENSE_CONFIG_NAMES:
        for algo_name in ALGORITHM_CLASSES:
            for gname in GRAPH_NAMES:
                stop_sizes = False
                for n in SIZES:
                    if stop_sizes:
                        break

                    # Load graph instances from cache
                    for s_idx in range(SAMPLES_PER_SIZE):
                        G, params = _load_cached_graph(gname, n, s_idx)
                        graph_seed = int(params.get("seed", 0) or 0)

                        # Graph metrics
                        n_nodes = G.number_of_nodes()
                        n_edges = G.number_of_edges()
                        density = (2.0 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
                        avg_deg = (2.0 * n_edges) / n_nodes if n_nodes > 0 else 0.0
                        # Avoid expensive clustering for very large graphs
                        clustering = nx.average_clustering(G) if (n_nodes > 1 and n_nodes <= 1500) else float("nan")
                        components = nx.number_connected_components(G)

                        over_here = False
                        for rep in range(REPEATS_PER_GRAPH):
                            algo_seed = 12345 + s_idx * 1000 + rep
                            result, is_over = _run_one(algo_name, G, lic_name, algo_seed)
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
                            status = "OK"
                            if row.get("notes") == "timeout":
                                status = "TIMEOUT"
                            elif row.get("notes") == "error":
                                status = "ERROR"
                            logger.info(
                                "%-12s n=%4d s=%d rep=%d cost=%.2f time_ms=%.2f valid=%s %s",
                                gname,
                                n,
                                s_idx,
                                rep,
                                row["total_cost"],
                                row["time_ms"],
                                row["valid"],
                                status,
                            )
                            if is_over:
                                over_here = True
                        if over_here:
                            logger.warning(
                                "%-12s n=%4d TIMEOUT — stopping larger sizes for %s on this graph",
                                gname,
                                n,
                                algo_name,
                            )
                            stop_sizes = True
                            break

    if out_path.exists():
        duration_s = _pc() - _t0
        log_run_footer(logger, {"csv": out_path, "elapsed_sec": f"{duration_s:.2f}"})


if __name__ == "__main__":
    main()
