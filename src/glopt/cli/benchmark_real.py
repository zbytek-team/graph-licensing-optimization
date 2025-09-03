from __future__ import annotations

import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import networkx as nx

from glopt.algorithms.greedy import GreedyAlgorithm
from glopt.core import instantiate_algorithms
from glopt.core.solution_validator import SolutionValidator
from glopt.io import ensure_dir
from glopt.io.data_loader import RealWorldDataLoader
from glopt.license_config import LicenseConfigFactory

# Configuration
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
LICENSE_CONFIG_NAMES: list[str] = ["spotify", "duolingo_super", "roman_domination"]
DYNAMIC_ROMAN_PS: list[float] = [1.5, 2.0, 2.5, 3.0]
LICENSE_CONFIG_NAMES.extend([f"roman_p_{str(p).replace('.', '_')}" for p in DYNAMIC_ROMAN_PS])

TIMEOUT_SECONDS: float = 60.0


def _json_dumps(obj: Any) -> str:
    import json as _json

    try:
        return _json.dumps(obj, ensure_ascii=False)
    except Exception:
        return "{}"


def _worker_solve(algo_name: str, graph: nx.Graph, license_config: str, seed: int, conn) -> None:  # type: ignore[no-redef]
    try:
        validator = SolutionValidator(debug=False)
        algo = instantiate_algorithms([algo_name])[0]
        lts = LicenseConfigFactory.get_config(license_config)
        deadline = perf_counter() + (TIMEOUT_SECONDS * 0.98)
        kwargs: dict[str, Any] = {"seed": seed, "deadline": deadline}
        # Warm-start for metaheuristics
        warm_names = {
            "GeneticAlgorithm",
            "SimulatedAnnealing",
            "TabuSearch",
            "AntColonyOptimization",
        }
        if algo_name in warm_names:
            greedy_sol = GreedyAlgorithm().solve(graph, lts)
            kwargs["initial_solution"] = greedy_sol
        if algo_name == "ILPSolver":
            kwargs["time_limit"] = int(max(1, TIMEOUT_SECONDS - 1))
        t0 = perf_counter()
        sol = algo.solve(graph, lts, **kwargs)
        elapsed_ms = (perf_counter() - t0) * 1000.0

        ok, issues = validator.validate(sol, graph)
        sizes = [g.size for g in sol.groups]
        groups = len(sizes)
        mean_sz = (sum(sizes) / groups) if groups else 0.0
        med_sz = float(sorted(sizes)[groups // 2]) if groups else 0.0
        p90 = float(sorted(sizes)[min(groups - 1, int(0.9 * (groups - 1)))]) if groups else 0.0
        # Algo params
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
            "group_size_median": float(med_sz),
            "group_size_p90": float(p90),
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


def _run_one(algo: str, graph: nx.Graph, lic: str, seed: int) -> tuple[dict[str, object], bool]:
    parent_conn, child_conn = mp.Pipe(duplex=False)
    p = mp.Process(target=_worker_solve, args=(algo, graph, lic, seed, child_conn))
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
                "cost_per_node": 0.0,
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
            "notes": "timeout",
        }
    return res, timed_out or (res.get("notes") == "error")


def main() -> None:
    run_id = (RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")) + "_benchmark_real"
    base = Path("runs") / run_id
    csv_dir = base / "csv"
    ensure_dir(str(csv_dir))
    out_path = csv_dir / f"{run_id}.csv"

    print("== glopt benchmark real ==")
    print(f"run_id: {run_id}")
    print(f"algorithms: {', '.join(ALGORITHM_CLASSES)}")
    print(f"licenses: {', '.join(LICENSE_CONFIG_NAMES)}")
    print(f"timeout: {TIMEOUT_SECONDS:.0f}s")

    loader = RealWorldDataLoader(data_dir="data")
    networks = loader.load_all_facebook_networks()
    ego_ids = sorted(networks.keys())
    print(f"facebook ego networks: {len(ego_ids)}")

    # Write CSV header
    import csv as _csv

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "graph",
                "graph_params",
                "license_config",
                "algorithm",
                "ego_id",
                "n_nodes",
                "n_edges",
                "density",
                "avg_degree",
                "clustering",
                "components",
                "total_cost",
                "cost_per_node",
                "time_ms",
                "algo_params_json",
                "warm_start",
                "valid",
                "issues",
                "groups",
                "group_size_mean",
                "group_size_median",
                "group_size_p90",
                "notes",
            ],
        )
        w.writeheader()

        for ego_id in ego_ids:
            G = networks[ego_id]
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            density = (2.0 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
            avg_deg = (2.0 * n_edges) / n_nodes if n_nodes > 0 else 0.0
            clustering = nx.average_clustering(G) if (n_nodes > 1 and n_nodes <= 1500) else float("nan")
            components = nx.number_connected_components(G)

            for lic in LICENSE_CONFIG_NAMES:
                for algo in ALGORITHM_CLASSES:
                    print(f"-> ego={ego_id} lic={lic} algo={algo}")
                    res, _ = _run_one(algo, G, lic, seed=12345)
                    row = {
                        "run_id": run_id,
                        "graph": "facebook_ego",
                        "graph_params": _json_dumps({"ego_id": ego_id}),
                        "license_config": lic,
                        "algorithm": algo,
                        "ego_id": ego_id,
                        "n_nodes": n_nodes,
                        "n_edges": n_edges,
                        "density": float(density),
                        "avg_degree": float(avg_deg),
                        "clustering": float(clustering),
                        "components": int(components),
                        **res,
                    }
                    w.writerow(row)

    print(f"csv: {out_path}")


if __name__ == "__main__":
    main()
