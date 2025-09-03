from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import networkx as nx

from glopt.algorithms.greedy import GreedyAlgorithm
from glopt.core import instantiate_algorithms
from glopt.core.solution_builder import SolutionBuilder
from glopt.core.solution_validator import SolutionValidator
from glopt.dynamic_simulator import DynamicNetworkSimulator, MutationParams
from glopt.io import build_paths, ensure_dir
from glopt.io.data_loader import RealWorldDataLoader
from glopt.license_config import LicenseConfigFactory

# ==============================================
# Configuration (edit here)
# ==============================================

RUN_ID: str | None = None

LICENSE_CONFIGS: list[str] = [
    "spotify",
    "duolingo_super",
    "roman_domination",
    # sweep p for roman domination
    "roman_p_1_5",
    "roman_p_2_0",
    "roman_p_2_5",
    "roman_p_3_0",
]

NUM_STEPS: int = 8
RANDOM_SEED: int = 123

WARM_ALGOS: list[str] = [
    "GeneticAlgorithm",
    "SimulatedAnnealing",
    "TabuSearch",
    "AntColonyOptimization",
]
BASELINE_ALGOS: list[str] = [
    "GreedyAlgorithm",
    "ILPSolver",
]


def _repair_solution_for_graph(prev_solution, graph: nx.Graph, license_types) -> Any:
    ok_groups = []
    nodes = set(graph.nodes())
    for g in prev_solution.groups:
        if g.owner not in nodes:
            continue
        allowed = set(graph.neighbors(g.owner)) | {g.owner}
        allm = set(g.all_members) & nodes
        if not allm:
            continue
        if not allm.issubset(allowed):
            continue
        try:
            new_g = type(g)(g.license_type, g.owner, frozenset(allm - {g.owner}))
        except Exception:
            continue
        ok_groups.append(new_g)
    covered = set().union(*(g.all_members for g in ok_groups)) if ok_groups else set()
    uncovered = set(graph.nodes()) - covered
    if uncovered:
        H = graph.subgraph(uncovered).copy()
        greedy = GreedyAlgorithm().solve(H, list(license_types))
        ok_groups.extend(greedy.groups)
    return SolutionBuilder.create_solution_from_groups(ok_groups)


def _write_csv_header(path: Path, header: list[str]) -> None:
    import csv

    ensure_dir(str(path.parent))
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)


def _append_csv_row(path: Path, row: list[Any]) -> None:
    import csv

    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def main() -> int:
    run_id = (RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")) + "_dynamic_real"
    _, _, csv_dir = build_paths(run_id)
    out_path = Path(csv_dir) / f"{run_id}.csv"

    print("== glopt dynamic benchmark (real graphs) ==")
    print(f"run_id: {run_id}")
    print(f"licenses: {', '.join(LICENSE_CONFIGS)}")
    print(f"warm algos: {', '.join(WARM_ALGOS)}")
    print(f"baselines: {', '.join(BASELINE_ALGOS)}")
    print(f"steps: {NUM_STEPS} seed={RANDOM_SEED}")

    header = [
        "run_id",
        "graph",
        "graph_params",
        "license_config",
        "algorithm",
        "warm_start",
        "step",
        "ego_id",
        "n_nodes",
        "n_edges",
        "total_cost",
        "time_ms",
        "delta_cost",
        "avg_time_ms_per_step",
        "delta_cost_abs",
        "delta_cost_std_so_far",
        "mutation_params_json",
        "valid",
        "issues",
        "groups",
        "mutations",
    ]
    _write_csv_header(out_path, header)

    validator = SolutionValidator(debug=False)
    loader = RealWorldDataLoader(data_dir="data")
    networks = loader.load_all_facebook_networks()

    for ego_id, graph in networks.items():
        print(f"ego_id={ego_id} nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")

        # Scale mutation params to graph size
        N = graph.number_of_nodes() or 1
        E0 = graph.number_of_edges() or N
        mut_params = MutationParams(
            add_nodes_prob=0.05,
            remove_nodes_prob=0.03,
            add_edges_prob=0.15,
            remove_edges_prob=0.10,
            max_nodes_add=max(1, int(0.02 * N)),
            max_nodes_remove=max(1, int(0.015 * N)),
            max_edges_add=max(1, int(0.03 * E0)),
            max_edges_remove=max(1, int(0.02 * E0)),
        )
        sim = DynamicNetworkSimulator(mutation_params=mut_params, seed=RANDOM_SEED)
        sim.next_node_id = max(graph.nodes()) + 1 if graph.nodes() else 0

        # Precompute mutation sequence
        graphs: list[nx.Graph] = [graph.copy()]
        mutations_per_step: list[list[str]] = [[]]
        g_curr = graph.copy()
        for _ in range(NUM_STEPS):
            g_curr, muts = sim._apply_mutations(g_curr)  # type: ignore[attr-defined]
            graphs.append(g_curr.copy())
            mutations_per_step.append(muts)

        for lic in LICENSE_CONFIGS:
            lts = LicenseConfigFactory.get_config(lic)
            warm_algos = instantiate_algorithms(WARM_ALGOS)
            base_algos = instantiate_algorithms(BASELINE_ALGOS)

            prev_solutions: dict[tuple[str, bool], Any] = {}
            time_accum: dict[tuple[str, bool], tuple[float, int]] = {}
            delta_stats: dict[tuple[str, bool], tuple[float, float, int]] = {}

            # Step 0 init
            G0 = graphs[0]
            for algo in warm_algos + base_algos:
                t0 = perf_counter()
                sol0 = algo.solve(G0, lts, seed=RANDOM_SEED)
                elapsed = (perf_counter() - t0) * 1000.0
                ok, issues = validator.validate(sol0, G0)
                key = (algo.name, False)
                prev_solutions[key] = sol0
                time_accum[key] = (elapsed, 1)
                delta_stats[key] = (0.0, 0.0, 0)
                print(f"init -> ego={ego_id} / {lic} / {algo.name:<24s} cold   cost={sol0.total_cost:.2f} time_ms={elapsed:.2f} valid={ok} groups={len(sol0.groups)}")
                import json as _json

                _append_csv_row(
                    out_path,
                    [
                        run_id,
                        "facebook_ego",
                        str({"ego_id": ego_id}),
                        lic,
                        algo.name,
                        False,
                        0,
                        ego_id,
                        G0.number_of_nodes(),
                        G0.number_of_edges(),
                        float(sol0.total_cost),
                        float(elapsed),
                        0.0,
                        float(elapsed),
                        0.0,
                        0.0,
                        _json.dumps(mut_params.__dict__),
                        bool(ok),
                        int(len(issues)),
                        int(len(sol0.groups)),
                        "; ".join(mutations_per_step[0]),
                    ],
                )

            # Steps 1..NUM_STEPS
            for step in range(1, NUM_STEPS + 1):
                Gs = graphs[step]
                muts = "; ".join(mutations_per_step[step])
                print(f"step {step:02d} -> ego={ego_id} lic={lic} nodes={Gs.number_of_nodes()} edges={Gs.number_of_edges()} mutations=[{muts}]")

                # Warm-start algos: warm and cold variants
                for algo in warm_algos:
                    prev_warm = prev_solutions.get((algo.name, True)) or prev_solutions.get((algo.name, False))
                    warm = _repair_solution_for_graph(prev_warm, Gs, lts) if prev_warm is not None else None

                    # warm run
                    t0 = perf_counter()
                    sol = algo.solve(Gs, lts, seed=RANDOM_SEED + step, initial_solution=warm)
                    elapsed = (perf_counter() - t0) * 1000.0
                    ok, issues = validator.validate(sol, Gs)
                    key_w = (algo.name, True)
                    prev_cost = (
                        float(prev_solutions.get(key_w, sol).total_cost)
                        if isinstance(prev_solutions.get(key_w), type(sol))
                        else (float(prev_solutions.get((algo.name, False), sol).total_cost) if isinstance(prev_solutions.get((algo.name, False)), type(sol)) else float("nan"))
                    )
                    delta = float(sol.total_cost) - (prev_cost if prev_cost == prev_cost else float("nan"))
                    tot, cnt = time_accum.get(key_w, (0.0, 0))
                    avg = (tot + elapsed) / (cnt + 1)
                    time_accum[key_w] = (tot + elapsed, cnt + 1)
                    mean_d, m2_d, k = delta_stats.get(key_w, (0.0, 0.0, 0))
                    x = float(delta) if delta == delta else 0.0
                    k1 = k + 1
                    dtmp = x - mean_d
                    mean_new = mean_d + dtmp / k1
                    m2_new = m2_d + dtmp * (x - mean_new)
                    delta_stats[key_w] = (mean_new, m2_new, k1)
                    prev_solutions[key_w] = sol
                    print(f"   warm  {algo.name:<24s} cost={sol.total_cost:.2f} time_ms={elapsed:.2f} dCost={delta if delta == delta else 0.0:+.2f} valid={ok} groups={len(sol.groups)}")
                    _append_csv_row(
                        out_path,
                        [
                            run_id,
                            "facebook_ego",
                            str({"ego_id": ego_id}),
                            lic,
                            algo.name,
                            True,
                            step,
                            ego_id,
                            Gs.number_of_nodes(),
                            Gs.number_of_edges(),
                            float(sol.total_cost),
                            float(elapsed),
                            float(delta if delta == delta else 0.0),
                            float(avg),
                            float(abs(delta) if delta == delta else 0.0),
                            float(((delta_stats[key_w][1] / max(1, delta_stats[key_w][2] - 1)) ** 0.5) if delta_stats[key_w][2] > 1 else 0.0),
                            _json.dumps(mut_params.__dict__),
                            bool(ok),
                            int(len(issues)),
                            int(len(sol.groups)),
                            muts,
                        ],
                    )

                    # cold run
                    t0 = perf_counter()
                    sol_cold = algo.solve(Gs, lts, seed=RANDOM_SEED + step)
                    elapsed_c = (perf_counter() - t0) * 1000.0
                    ok_c, issues_c = validator.validate(sol_cold, Gs)
                    key_c = (algo.name, False)
                    prev_cost_c = float(prev_solutions.get(key_c, sol_cold).total_cost) if isinstance(prev_solutions.get(key_c), type(sol_cold)) else float("nan")
                    delta_c = float(sol_cold.total_cost) - (prev_cost_c if prev_cost_c == prev_cost_c else float("nan"))
                    tot_c, cnt_c = time_accum.get(key_c, (0.0, 0))
                    avg_c = (tot_c + elapsed_c) / (cnt_c + 1)
                    time_accum[key_c] = (tot_c + elapsed_c, cnt_c + 1)
                    mean_d, m2_d, k = delta_stats.get(key_c, (0.0, 0.0, 0))
                    x = float(delta_c) if delta_c == delta_c else 0.0
                    k1 = k + 1
                    dtmp = x - mean_d
                    mean_new = mean_d + dtmp / k1
                    m2_new = m2_d + dtmp * (x - mean_new)
                    delta_stats[key_c] = (mean_new, m2_new, k1)
                    prev_solutions[key_c] = sol_cold
                    print(f"   cold  {algo.name:<24s} cost={sol_cold.total_cost:.2f} time_ms={elapsed_c:.2f} dCost={delta_c if delta_c == delta_c else 0.0:+.2f} valid={ok_c} groups={len(sol_cold.groups)}")
                    _append_csv_row(
                        out_path,
                        [
                            run_id,
                            "facebook_ego",
                            str({"ego_id": ego_id}),
                            lic,
                            algo.name,
                            False,
                            step,
                            ego_id,
                            Gs.number_of_nodes(),
                            Gs.number_of_edges(),
                            float(sol_cold.total_cost),
                            float(elapsed_c),
                            float(delta_c if delta_c == delta_c else 0.0),
                            float(avg_c),
                            float(abs(delta_c) if delta_c == delta_c else 0.0),
                            float(((delta_stats[key_c][1] / max(1, delta_stats[key_c][2] - 1)) ** 0.5) if delta_stats[key_c][2] > 1 else 0.0),
                            _json.dumps(mut_params.__dict__),
                            bool(ok_c),
                            int(len(issues_c)),
                            int(len(sol_cold.groups)),
                            muts,
                        ],
                    )

                # Baseline algos (cold only)
                for algo in base_algos:
                    t0 = perf_counter()
                    sol = algo.solve(Gs, lts, seed=RANDOM_SEED + step)
                    elapsed = (perf_counter() - t0) * 1000.0
                    ok, issues = validator.validate(sol, Gs)
                    key_b = (algo.name, False)
                    prev_cost_b = float(prev_solutions.get(key_b, sol).total_cost) if isinstance(prev_solutions.get(key_b), type(sol)) else float("nan")
                    delta_b = float(sol.total_cost) - (prev_cost_b if prev_cost_b == prev_cost_b else float("nan"))
                    tot_b, cnt_b = time_accum.get(key_b, (0.0, 0))
                    avg_b = (tot_b + elapsed) / (cnt_b + 1)
                    time_accum[key_b] = (tot_b + elapsed, cnt_b + 1)
                    mean_d, m2_d, k = delta_stats.get(key_b, (0.0, 0.0, 0))
                    x = float(delta_b) if delta_b == delta_b else 0.0
                    k1 = k + 1
                    dtmp = x - mean_d
                    mean_new = mean_d + dtmp / k1
                    m2_new = m2_d + dtmp * (x - mean_new)
                    delta_stats[key_b] = (mean_new, m2_new, k1)
                    prev_solutions[key_b] = sol
                    print(f"   base  {algo.name:<24s} cost={sol.total_cost:.2f} time_ms={elapsed:.2f} dCost={delta_b if delta_b == delta_b else 0.0:+.2f} valid={ok} groups={len(sol.groups)}")
                    _append_csv_row(
                        out_path,
                        [
                            run_id,
                            "facebook_ego",
                            str({"ego_id": ego_id}),
                            lic,
                            algo.name,
                            False,
                            step,
                            ego_id,
                            Gs.number_of_nodes(),
                            Gs.number_of_edges(),
                            float(sol.total_cost),
                            float(elapsed),
                            float(delta_b if delta_b == delta_b else 0.0),
                            float(avg_b),
                            float(abs(delta_b) if delta_b == delta_b else 0.0),
                            float(((delta_stats[key_b][1] / max(1, delta_stats[key_b][2] - 1)) ** 0.5) if delta_stats[key_b][2] > 1 else 0.0),
                            _json.dumps(mut_params.__dict__),
                            bool(ok),
                            int(len(issues)),
                            int(len(sol.groups)),
                            muts,
                        ],
                    )

    print(f"csv: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
