from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import networkx as nx

from glopt.algorithms.greedy import GreedyAlgorithm
from glopt.cli.common import build_run_id, print_banner, print_footer
from glopt.core import LicenseGroup, Solution, SolutionBuilder, instantiate_algorithms
from glopt.core.solution_validator import SolutionValidator
from glopt.dynamic_simulator import DynamicNetworkSimulator, MutationParams
from glopt.io import ensure_dir
from glopt.io.data_loader import RealWorldDataLoader
from glopt.license_config import LicenseConfigFactory

RUN_ID: str | None = None

ALGORITHM_CLASSES: list[str] = [
    "ILPSolver",
    "GreedyAlgorithm",
    "RandomizedAlgorithm",
    "DominatingSetAlgorithm",
    "AntColonyOptimization",
    "SimulatedAnnealing",
    "TabuSearch",
    "GeneticAlgorithm",
]

LICENSE_CONFIG_NAMES: list[str] = [
    "duolingo_super",
    "spotify",
    "roman_domination",
]


TIMEOUT_SECONDS: float = 60.0
REPEATS_PER_GRAPH: int = 1
NUM_STEPS: int = 100
RANDOM_SEED: int = 123

ADD_NODES_PROB: float = 0.05
REMOVE_NODES_PROB: float = 0.03
ADD_EDGES_PROB: float = 0.15
REMOVE_EDGES_PROB: float = 0.10


def _json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return "{}"


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
        lts = LicenseConfigFactory.get_config(license_config)
        kwargs: dict[str, Any] = {"seed": seed}
        warm_names = {
            "GeneticAlgorithm",
            "SimulatedAnnealing",
            "TabuSearch",
            "AntColonyOptimization",
        }
        do_warm = use_warm and (algo_name in warm_names)
        if do_warm:
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
        from collections import Counter as _Counter

        lic_counts = _Counter(g.license_type.name for g in sol.groups)
        try:
            params_json = _json_dumps({k: v for k, v in vars(algo).items() if isinstance(v, (int, float, str, bool))})
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
            "cost_per_node": float(sol.total_cost) / max(1, graph.number_of_nodes()),
            "license_counts_json": _json_dumps(lic_counts),
            "algo_params_json": params_json,
            "warm_start": bool(do_warm),
        }
        if return_solution:
            try:
                import pickle as _p

                res["solution_pickle"] = _p.dumps(sol, protocol=_p.HIGHEST_PROTOCOL)
            except Exception:
                pass
    except Exception as e:
        res = {"success": False, "error": str(e)}
    try:
        conn.send(res)
    finally:
        conn.close()


def _run_one(algo: str, graph: nx.Graph, lic: str, seed: int, warm: bool, init_bytes: bytes | None = None) -> tuple[dict[str, object], bool]:
    parent_conn, child_conn = mp.Pipe(duplex=False)
    p = mp.Process(target=_worker_solve, args=(algo, graph, lic, seed, warm, child_conn, init_bytes, True))
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
                "cost_per_node": float(msg.get("cost_per_node", 0.0)),
                "license_counts_json": str(msg.get("license_counts_json", "{}")),
                "algo_params_json": str(msg.get("algo_params_json", "{}")),
                "warm_start": bool(msg.get("warm_start", False)),
                "notes": "",
            }
            if "solution_pickle" in msg:
                res["solution_pickle"] = msg["solution_pickle"]
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
                "cost_per_node": 0.0,
                "license_counts_json": "{}",
                "algo_params_json": "{}",
                "warm_start": False,
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
            "cost_per_node": 0.0,
            "license_counts_json": "{}",
            "algo_params_json": "{}",
            "warm_start": False,
            "notes": "timeout",
        }
    return res, timed_out or (res.get("notes") == "error")


def main() -> None:
    run_id = build_run_id("dynamic_real", RUN_ID)
    base = Path("runs") / run_id
    csv_dir = base / "csv"
    ensure_dir(str(csv_dir))
    out_path = csv_dir / f"{run_id}.csv"

    from time import perf_counter as _pc

    _t0 = _pc()
    print_banner(
        "glopt dynamic",
        {
            "run_id": run_id,
            "graphs": "facebook_ego",
            "steps_per_ego": NUM_STEPS,
            "repeats_per_step": REPEATS_PER_GRAPH,
            "licenses": ", ".join(LICENSE_CONFIG_NAMES),
            "algorithms": ", ".join(ALGORITHM_CLASSES),
            "timeout": f"{TIMEOUT_SECONDS:.0f}s",
        },
    )

    loader = RealWorldDataLoader(data_dir="data")
    networks = loader.load_all_facebook_networks()
    ego_ids = sorted(networks.keys())

    import csv as _csv

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "algorithm",
                "graph",
                "n_nodes",
                "n_edges",
                "graph_params",
                "license_config",
                "rep",
                "seed",
                "ego_id",
                "density",
                "avg_degree",
                "clustering",
                "components",
                "total_cost",
                "cost_per_node",
                "time_ms",
                "license_counts_json",
                "algo_params_json",
                "warm_start",
                "valid",
                "issues",
                "groups",
                "group_size_mean",
                "group_size_median",
                "group_size_p90",
                "image_path",
                "notes",
                "step",
                "mutation_params_json",
                "mutations",
            ],
        )
        w.writeheader()

        warmable = {"GeneticAlgorithm", "SimulatedAnnealing", "TabuSearch", "AntColonyOptimization"}
        for lic in LICENSE_CONFIG_NAMES:
            for algo in ALGORITHM_CLASSES:
                print(f"-> {lic} / {algo}")
                warm_variants = [False, True] if algo in warmable else [False]
                for ego_id in ego_ids:
                    G0 = networks[ego_id]
                    N = G0.number_of_nodes() or 1
                    E0 = G0.number_of_edges() or N
                    mut_params = MutationParams(
                        add_nodes_prob=ADD_NODES_PROB,
                        remove_nodes_prob=REMOVE_NODES_PROB,
                        add_edges_prob=ADD_EDGES_PROB,
                        remove_edges_prob=REMOVE_EDGES_PROB,
                        max_nodes_add=max(1, int(0.02 * N)),
                        max_nodes_remove=max(1, int(0.015 * N)),
                        max_edges_add=max(1, int(0.03 * E0)),
                        max_edges_remove=max(1, int(0.02 * E0)),
                    )
                    sim = DynamicNetworkSimulator(mutation_params=mut_params, seed=RANDOM_SEED)
                    sim.next_node_id = max(G0.nodes()) + 1 if G0.nodes() else 0
                    graphs: list[nx.Graph] = [G0.copy()]
                    mutations_per_step: list[list[str]] = [[]]
                    g_curr = G0.copy()
                    for _ in range(NUM_STEPS):
                        g_curr, muts = sim._apply_mutations(g_curr)
                        graphs.append(g_curr.copy())
                        mutations_per_step.append(muts)

                    stop_steps = False
                    prev_solution_warm: Solution | None = None
                    for step in range(NUM_STEPS + 1):
                        print(f"== graph {str(ego_id)} step {step} / {NUM_STEPS} ==")
                        Gs = graphs[step]
                        muts = "; ".join(mutations_per_step[step])
                        n_nodes = Gs.number_of_nodes()
                        n_edges = Gs.number_of_edges()
                        density = (2.0 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
                        avg_deg = (2.0 * n_edges) / n_nodes if n_nodes > 0 else 0.0
                        clustering = nx.average_clustering(Gs) if (n_nodes > 1 and n_nodes <= 1500) else float("nan")
                        components = nx.number_connected_components(Gs)

                        for rep in range(REPEATS_PER_GRAPH):
                            print(f"-- license {lic} algo {algo} rep {rep} --")
                            for warm_flag in warm_variants:
                                algo_seed = 12345 + step * 1000 + rep
                                init_bytes = None
                                if warm_flag and prev_solution_warm is not None:

                                    lts = LicenseConfigFactory.get_config(lic)

                                    nodes = set(Gs.nodes())
                                    used: set = set()
                                    new_groups: list[LicenseGroup] = []
                                    from typing import cast as _cast

                                    degv = _cast(Any, Gs.degree)
                                    for g in prev_solution_warm.groups:
                                        cand = list((g.all_members & nodes) - used)
                                        if not cand:
                                            continue
                                        owner = g.owner if g.owner in cand else max(cand, key=lambda n: int(degv[n]))
                                        allowed = (set(Gs.neighbors(owner)) | {owner}) - used
                                        members = [n for n in cand if n in allowed]
                                        if not members:
                                            continue
                                        max_cap = max(lt.max_capacity for lt in lts)
                                        members_sorted = sorted(members, key=lambda n: int(degv[n]), reverse=True)
                                        if owner not in members_sorted:
                                            members_sorted = [owner] + [n for n in members_sorted if n != owner]
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
                                        owner_final = owner if owner in chosen_members else chosen_members[0]
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
                                        for n_ in uncovered:
                                            new_groups.append(LicenseGroup(lt1, n_, frozenset()))
                                    proj = Solution(groups=tuple(new_groups))
                                    ok, _ = SolutionValidator(debug=False).validate(proj, Gs)
                                    if not ok:
                                        proj = GreedyAlgorithm().solve(Gs, lts)
                                    try:
                                        import pickle as _p

                                        init_bytes = _p.dumps(proj, protocol=_p.HIGHEST_PROTOCOL)
                                    except Exception:
                                        init_bytes = None
                                res, over = _run_one(algo, Gs, lic, seed=algo_seed, warm=warm_flag, init_bytes=init_bytes)

                                sol_bytes = res.pop("solution_pickle", None)
                                if warm_flag and isinstance(sol_bytes, (bytes, bytearray)):
                                    try:
                                        import pickle as _p

                                        prev_solution_warm = _p.loads(sol_bytes)
                                    except Exception:
                                        prev_solution_warm = None
                            row = {
                                "run_id": run_id,
                                "algorithm": algo,
                                "graph": "facebook_ego",
                                "n_nodes": n_nodes,
                                "n_edges": n_edges,
                                "graph_params": _json_dumps({"ego_id": ego_id}),
                                "license_config": lic,
                                "rep": rep,
                                "seed": algo_seed,
                                "ego_id": ego_id,
                                "density": float(density),
                                "avg_degree": float(avg_deg),
                                "clustering": float(clustering),
                                "components": int(components),
                                "image_path": "",
                                **res,
                                "step": int(step),
                                "mutation_params_json": _json_dumps(sim.mutation_params.__dict__),
                                "mutations": muts,
                            }
                            w.writerow(row)

                            status = "OK"
                            if row.get("notes") == "timeout":
                                status = "TIMEOUT"
                            elif row.get("notes") == "error":
                                status = "ERROR"
                            warm_lbl = "warm" if row.get("warm_start") else "cold"
                            print(f"{'facebook_ego':12s} ego={str(ego_id):8s} step={step:02d} rep={rep} {warm_lbl:4s} cost={row['total_cost']:.2f} time_ms={row['time_ms']:.2f} valid={row['valid']} {status}")
                            if over:
                                stop_steps = True
                                break
                        if stop_steps:
                            break

    if out_path.exists():
        duration_s = _pc() - _t0
        print_footer({"csv": out_path, "elapsed_sec": f"{duration_s:.2f}"})


if __name__ == "__main__":
    main()
