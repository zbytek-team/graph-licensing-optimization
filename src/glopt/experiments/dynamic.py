from __future__ import annotations

import json
import multiprocessing as mp
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx

from glopt.core import (
    LicenseGroup,
    Solution,
    SolutionBuilder,
    generate_graph,
    instantiate_algorithms,
)
from glopt.core.algorithms.greedy import GreedyAlgorithm
from glopt.core.dynamic_simulator import (
    DynamicNetworkSimulator,
    MutationParams,
)
from glopt.core.io import ensure_dir
from glopt.core.license_config import LicenseConfigFactory
from glopt.core.solution_validator import SolutionValidator
from glopt.experiments.common import (
    build_run_id,
    normalize_license_costs,
    print_footer,
)

RUN_ID: str | None = None
GRAPH_NAMES: list[str] = ["random", "small_world", "scale_free"]
GRAPH_DEFAULTS: dict[str, dict[str, Any]] = {
    "random": {"p": 0.1, "seed": 42},
    "small_world": {"k": 6, "p": 0.05, "seed": 42},
    "scale_free": {"m": 2, "seed": 42},
}
SIZES: list[int] = [40, 80, 160, 320, 640, 1280, 2560, 5120, 10000]
NUM_STEPS: int = 50
REPEATS_PER_GRAPH: int = 1
TIMEOUT_SECONDS: float = 60.0
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
    "AntColonyOptimization",
    "SimulatedAnnealing",
    "TabuSearch",
    "GeneticAlgorithm",
]
GRAPH_CACHE_DIR: str = "data/graphs_cache"
ADD_NODES_PROB: float = 0.06
REMOVE_NODES_PROB: float = 0.04
ADD_EDGES_PROB: float = 0.18
REMOVE_EDGES_PROB: float = 0.12
RANDOM_SEED: int = 123
MODE_NODES: str = "random"
MODE_EDGES: str = "random"


def _adjust_params(name: str, n: int, base: dict[str, Any]) -> dict[str, Any]:
    p = dict(base)
    if name == "scale_free":
        m = int(p.get("m", 2))
        p["m"] = max(1, min(m, max(1, n - 1)))
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


def _json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return "{}"


def _write_row(csv_path: Path, row: dict[str, object]) -> None:
    import csv as _csv

    ensure_dir(str(csv_path.parent))
    is_new = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)


def _cache_paths(cache_dir: str, gname: str, n: int) -> tuple[Path, Path]:
    base = Path(cache_dir) / gname / f"n{n:04d}"
    gpath = base / "s0.gpickle"
    mpath = gpath.with_suffix(".json")
    return (gpath, mpath)


def _ensure_cache(graphs: list[str], sizes: list[int]) -> None:
    ensure_dir(GRAPH_CACHE_DIR)
    created = 0
    for gname in graphs:
        for n in sizes:
            base_params = _adjust_params(
                gname, n, GRAPH_DEFAULTS.get(gname, {})
            )
            gpath, mpath = _cache_paths(GRAPH_CACHE_DIR, gname, n)
            if gpath.exists() and mpath.exists():
                continue
            seed = int(base_params.get("seed", 42) or 42)
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
                "sample": 0,
            }
            with mpath.open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)
            created += 1
    if created:
        print(
            f"graph cache: generated {created} new base graphs under " \
            f"{GRAPH_CACHE_DIR}"
        )
    else:
        print(f"graph cache: up-to-date at {GRAPH_CACHE_DIR}")


def _load_cached_graph(gname: str, n: int) -> tuple[nx.Graph, dict[str, Any]]:
    gpath, mpath = _cache_paths(GRAPH_CACHE_DIR, gname, n)
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
    use_warm: bool,
    conn,
    initial_solution_bytes: bytes | None = None,
    return_solution: bool = False,
) -> None:
    try:
        validator = SolutionValidator(debug=False)
        algo = instantiate_algorithms([algo_name])[0]
        lts = normalize_license_costs(
            LicenseConfigFactory.get_config(license_config)
        )
        kwargs: dict[str, Any] = {"seed": seed}
        warm_names = {
            "GeneticAlgorithm",
            "SimulatedAnnealing",
            "TabuSearch",
            "AntColonyOptimization",
        }
        do_warm = use_warm and algo_name in warm_names
        if do_warm:
            if initial_solution_bytes:
                try:
                    init = pickle.loads(initial_solution_bytes)
                    kwargs["initial_solution"] = init
                except Exception:
                    greedy_sol = GreedyAlgorithm().solve(graph, lts)
                    kwargs["initial_solution"] = greedy_sol
            else:
                greedy_sol = GreedyAlgorithm().solve(graph, lts)
                kwargs["initial_solution"] = greedy_sol
        from time import perf_counter as _pc

        t0 = _pc()
        sol = algo.solve(graph, lts, **kwargs)
        elapsed_ms = (_pc() - t0) * 1000.0
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
            "algo_params_json": params_json,
            "warm_start": bool(do_warm),
            "cost_per_node": float(sol.total_cost)
            / max(1, graph.number_of_nodes()),
        }
        if return_solution:
            try:
                res["solution_pickle"] = pickle.dumps(
                    sol, protocol=pickle.HIGHEST_PROTOCOL
                )
            except Exception:
                pass
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
        args=(
            algo_name,
            graph,
            license_config,
            seed,
            False,
            child_conn,
        ),
    )
    p.start()
    timed_out = False
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
        }
    return (res, timed_out or res.get("notes") == "error")


def main() -> None:
    run_id = build_run_id("dynamic", RUN_ID)
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
    print(f"Starting glopt dynamic run {run_id}")
    print(f"Run directory: {base}")
    print(f"Results file: {out_path}")
    print(f"Graphs: {graphs_summary}")
    print(f"Sizes: {sizes_summary}")
    print(f"Steps per size: {NUM_STEPS}")
    print(f"Repeats per graph: {REPEATS_PER_GRAPH}")
    print(f"License configurations: {licenses_summary}")
    print(f"Algorithms: {algorithms_summary}")
    print(
        "Mutation probabilities: "
        f"add_nodes={ADD_NODES_PROB}, remove_nodes={REMOVE_NODES_PROB}, "
        f"add_edges={ADD_EDGES_PROB}, remove_edges={REMOVE_EDGES_PROB}"
    )
    print(f"Mutation modes: nodes={MODE_NODES}, edges={MODE_EDGES}")
    print(f"Timeout per run: {TIMEOUT_SECONDS:.0f}s")
    print("Graph cache warm-up started")
    _ensure_cache(GRAPH_NAMES, SIZES)
    print("Graph cache warm-up finished")

    def _project_solution(
        prev: Solution, graph: nx.Graph, lic_name: str
    ) -> Solution:
        lts = normalize_license_costs(
            LicenseConfigFactory.get_config(lic_name)
        )
        nodes = set(graph.nodes())
        used: set = set()
        new_groups: list[LicenseGroup] = []
        from typing import cast as _cast

        degv = _cast(Any, graph.degree)
        for g in prev.groups:
            cand = list((g.all_members & nodes) - used)
            if not cand:
                continue
            owner = (
                g.owner
                if g.owner in cand
                else max(cand, key=lambda n: int(degv[n]))
            )
            allowed = (set(graph.neighbors(owner)) | {owner}) - used
            members = [n for n in cand if n in allowed]
            if not members:
                continue
            max_cap = max(lt.max_capacity for lt in lts)
            members_sorted = sorted(
                members, key=lambda n: int(degv[n]), reverse=True
            )
            if owner not in members_sorted:
                members_sorted = [owner] + [
                    n for n in members_sorted if n != owner
                ]
            size = min(len(members_sorted), max_cap)
            chosen_lt = None
            chosen_members = None
            while size >= 1 and chosen_lt is None:
                lt = SolutionBuilder.find_cheapest_license_for_size(size, lts)
                if lt is not None:
                    chosen_lt = lt
                    chosen_members = members_sorted[:size]
                    if owner not in chosen_members:
                        chosen_members[-1] = owner
                else:
                    size -= 1
            if chosen_lt is None or chosen_members is None:
                continue
            owner_final = (
                owner if owner in chosen_members else chosen_members[0]
            )
            additional = frozenset(set(chosen_members) - {owner_final})
            try:
                ng = LicenseGroup(chosen_lt, owner_final, additional)
            except Exception:
                continue
            new_groups.append(ng)
            used.update(ng.all_members)
        uncovered = nodes - used
        if uncovered:
            lt1 = SolutionBuilder.find_cheapest_single_license(lts)
            for n in uncovered:
                new_groups.append(LicenseGroup(lt1, n, frozenset()))
        sol = Solution(groups=tuple(new_groups))
        ok, _ = SolutionValidator(debug=False).validate(sol, graph)
        if not ok:
            sol = GreedyAlgorithm().solve(graph, lts)
        return sol

    warmable = {
        "GeneticAlgorithm",
        "SimulatedAnnealing",
        "TabuSearch",
        "AntColonyOptimization",
    }
    for lic_name in LICENSE_CONFIG_NAMES:
        for algo_name in ALGORITHM_CLASSES:
            print(f"-> {lic_name} / {algo_name}")
            warm_variants = [False, True] if algo_name in warmable else [False]
            for gname in GRAPH_NAMES:
                stop_sizes = False
                for n in SIZES:
                    if stop_sizes:
                        break
                    G0, params = _load_cached_graph(gname, n)
                    sim = DynamicNetworkSimulator(
                        mutation_params=MutationParams(
                            add_nodes_prob=ADD_NODES_PROB,
                            remove_nodes_prob=REMOVE_NODES_PROB,
                            add_edges_prob=ADD_EDGES_PROB,
                            remove_edges_prob=REMOVE_EDGES_PROB,
                            max_nodes_add=max(1, int(0.02 * max(1, n))),
                            max_nodes_remove=max(1, int(0.015 * max(1, n))),
                            max_edges_add=max(
                                1,
                                int(0.03 * max(1, G0.number_of_edges() or n)),
                            ),
                            max_edges_remove=max(
                                1,
                                int(0.02 * max(1, G0.number_of_edges() or n)),
                            ),
                            mode_nodes=MODE_NODES,
                            mode_edges=MODE_EDGES,
                        ),
                        seed=RANDOM_SEED,
                    )
                    sim.next_node_id = max(G0.nodes()) + 1 if G0.nodes() else 0
                    graphs: list[nx.Graph] = [G0.copy()]
                    mutations_per_step: list[list[str]] = [[]]
                    g_curr = G0.copy()
                    for _ in range(NUM_STEPS):
                        g_curr, muts = sim._apply_mutations(g_curr)
                        graphs.append(g_curr.copy())
                        mutations_per_step.append(muts)
                    over_here = False
                    prev_solution_warm: Solution | None = None
                    for step in range(NUM_STEPS + 1):
                        Gs = graphs[step]
                        muts = "; ".join(mutations_per_step[step])
                        for rep in range(REPEATS_PER_GRAPH):
                            for warm_flag in warm_variants:
                                parent_conn, child_conn = mp.Pipe(duplex=False)
                                init_bytes = None
                                if (
                                    warm_flag
                                    and prev_solution_warm is not None
                                ):
                                    projected = _project_solution(
                                        prev_solution_warm,
                                        Gs,
                                        lic_name,
                                    )
                                    try:
                                        init_bytes = pickle.dumps(
                                            projected,
                                            protocol=pickle.HIGHEST_PROTOCOL,
                                        )
                                    except Exception:
                                        init_bytes = None
                                p = mp.Process(
                                    target=_worker_solve,
                                    args=(
                                        algo_name,
                                        Gs,
                                        lic_name,
                                        12345 + step * 1000 + rep,
                                        warm_flag,
                                        child_conn,
                                        init_bytes,
                                        True,
                                    ),
                                )
                                p.start()
                                timed_out = False
                                if parent_conn.poll(TIMEOUT_SECONDS):
                                    try:
                                        msg = parent_conn.recv()
                                    except EOFError:
                                        msg = {
                                            "success": False,
                                            "error": "no-data",
                                        }
                                    p.join()
                                    if msg.get("success"):
                                        result = {
                                            "total_cost": float(
                                                msg.get(
                                                    "total_cost",
                                                    float("nan"),
                                                )
                                            ),
                                            "time_ms": float(
                                                msg.get("time_ms", 0.0)
                                            ),
                                            "valid": bool(
                                                msg.get("valid", False)
                                            ),
                                            "issues": int(
                                                msg.get("issues", 0)
                                            ),
                                            "groups": int(
                                                msg.get("groups", 0)
                                            ),
                                            "group_size_mean": float(
                                                msg.get(
                                                    "group_size_mean",
                                                    0.0,
                                                )
                                            ),
                                            "group_size_median": float(
                                                msg.get(
                                                    "group_size_median",
                                                    0.0,
                                                )
                                            ),
                                            "group_size_p90": float(
                                                msg.get(
                                                    "group_size_p90",
                                                    0.0,
                                                )
                                            ),
                                            "license_counts_json": str(
                                                msg.get(
                                                    "license_counts_json",
                                                    "{}",
                                                )
                                            ),
                                            "algo_params_json": str(
                                                msg.get(
                                                    "algo_params_json",
                                                    "{}",
                                                )
                                            ),
                                            "warm_start": bool(
                                                msg.get(
                                                    "warm_start",
                                                    False,
                                                )
                                            ),
                                            "cost_per_node": float(
                                                msg.get(
                                                    "cost_per_node",
                                                    0.0,
                                                )
                                            ),
                                            "notes": "",
                                        }
                                        if warm_flag:
                                            sol_bytes = msg.get(
                                                "solution_pickle"
                                            )
                                            if isinstance(
                                                sol_bytes,
                                                (bytes, bytearray),
                                            ):
                                                try:
                                                    prev_solution_warm = (
                                                        pickle.loads(sol_bytes)
                                                    )
                                                except Exception:
                                                    prev_solution_warm = None
                                    else:
                                        result = {
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
                                    timed_out = True
                                    try:
                                        p.terminate()
                                    finally:
                                        p.join()
                                    result = {
                                        "total_cost": float("nan"),
                                        "time_ms": float(
                                            TIMEOUT_SECONDS * 1000.0
                                        ),
                                        "valid": False,
                                        "issues": 0,
                                        "groups": 0,
                                        "group_size_mean": 0.0,
                                        "group_size_median": 0.0,
                                        "group_size_p90": 0.0,
                                        "license_counts_json": "{}",
                                        "notes": "timeout",
                                    }
                                is_over = (
                                    timed_out or result.get("notes") == "error"
                                )
                            n_nodes = Gs.number_of_nodes()
                            n_edges = Gs.number_of_edges()
                            density = (
                                2.0 * n_edges / (n_nodes * (n_nodes - 1))
                                if n_nodes > 1
                                else 0.0
                            )
                            avg_deg = (
                                2.0 * n_edges / n_nodes if n_nodes > 0 else 0.0
                            )
                            clustering = (
                                nx.average_clustering(Gs)
                                if n_nodes > 1 and n_nodes <= 1500
                                else float("nan")
                            )
                            components = nx.number_connected_components(Gs)
                            row = {
                                "run_id": run_id,
                                "algorithm": algo_name,
                                "graph": gname,
                                "n_nodes": n_nodes,
                                "n_edges": n_edges,
                                "graph_params": str(params),
                                "license_config": lic_name,
                                "rep": rep,
                                "seed": 12345 + step * 1000 + rep,
                                "sample": 0,
                                "graph_seed": int(params.get("seed", 0) or 0),
                                "density": float(density),
                                "avg_degree": float(avg_deg),
                                "clustering": float(clustering),
                                "components": int(components),
                                "image_path": "",
                                **result,
                                "step": int(step),
                                "mutation_params_json": _json_dumps(
                                    sim.mutation_params.__dict__
                                ),
                                "mutations": muts,
                            }
                            _write_row(out_path, row)
                            status = "OK"
                            if row.get("notes") == "timeout":
                                status = "TIMEOUT"
                            elif row.get("notes") == "error":
                                status = "ERROR"
                            warm_lbl = (
                                "warm" if row.get("warm_start") else "cold"
                            )
                            print(
                                f"{gname:12s} n={n:4d} " \
                                f"step={step:02d} rep={rep} " \
                                f"{warm_lbl:4s} " \
                                f"cost={row['total_cost']:.2f} " \
                                f"time_ms={row['time_ms']:.2f} " \
                                f"valid={row['valid']} {status}"
                            )
                            if is_over:
                                over_here = True
                        if over_here:
                            break
                    if over_here:
                        print(
                            f"{gname:12s} n={n:4d} TIMEOUT " \
                            f"stopping larger sizes " \
                            f"for {algo_name}"
                        )
                        stop_sizes = True
                        break
    if out_path.exists():
        duration_s = _pc() - _t0
        print_footer({"csv": out_path, "elapsed_sec": f"{duration_s:.2f}"})


if __name__ == "__main__":
    main()
