from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import networkx as nx

from glopt.core import SolutionValidator, generate_graph
from glopt.license_config import LicenseConfigFactory
from glopt.algorithms.genetic import GeneticAlgorithm
from glopt.algorithms.simulated_annealing import SimulatedAnnealing
from glopt.algorithms.tabu_search import TabuSearch
from glopt.algorithms.ant_colony import AntColonyOptimization


@dataclass
class Trial:
    algo: str
    params: dict[str, Any]


def run_one(G: nx.Graph, license_config: str, trial: Trial, seed: int) -> dict[str, Any]:
    validator = SolutionValidator()
    L = LicenseConfigFactory.get_config(license_config)

    if trial.algo == "genetic":
        algo = GeneticAlgorithm(**trial.params, seed=seed)
    elif trial.algo == "sa":
        algo = SimulatedAnnealing(**trial.params)
    elif trial.algo == "tabu":
        algo = TabuSearch(**trial.params)
    elif trial.algo == "aco":
        algo = AntColonyOptimization(**trial.params)
    else:
        raise ValueError(f"unknown algo: {trial.algo}")

    t0 = perf_counter()
    sol = algo.solve(G, L, seed=seed)
    dt = (perf_counter() - t0) * 1000.0
    ok, issues = validator.validate(sol, G)
    return {
        "algo": trial.algo,
        "params": {k: v for k, v in trial.params.items()},
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "total_cost": float(sol.total_cost),
        "cost_per_node": float(sol.total_cost) / max(1, G.number_of_nodes()),
        "time_ms": dt,
        "valid": bool(ok),
        "issues": int(len(issues)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Coarse parameter sweep for metaheuristics")
    ap.add_argument("--out", default="results/meta_params_sweep.csv")
    ap.add_argument("--graphs", nargs="*", default=["random", "small_world"])
    ap.add_argument("--sizes", nargs="*", type=int, default=[150, 300])
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--license", default="duolingo_super")
    args = ap.parse_args()

    # Define coarse grids
    trials: list[Trial] = []
    for P in [20, 30, 40]:
        for Gens in [20, 40, 60]:
            trials.append(Trial("genetic", {"population_size": P, "generations": Gens}))
    for T0 in [50, 100, 200]:
        for alpha in [0.99, 0.995]:
            trials.append(Trial("sa", {"T0": T0, "cooling": alpha, "T_min": 1e-3}))
    for it in [500, 1000]:
        for tenure in [10, 20, 40]:
            trials.append(Trial("tabu", {"max_iterations": it, "tabu_tenure": tenure, "neighbors_per_iter": 10}))
    for a in [1.0]:
        for b in [2.0, 3.0]:
            trials.append(Trial("aco", {"alpha": a, "beta": b, "evaporation": 0.5, "q0": 0.9, "num_ants": 20, "max_iterations": 100}))

    out_path = args.out
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "graph",
                "n",
                "m",
                "algo",
                "params",
                "total_cost",
                "cost_per_node",
                "time_ms",
                "valid",
                "issues",
            ],
        )
        w.writeheader()

        for gname in args.graphs:
            for n in args.sizes:
                # Basic default params per generator
                if gname == "random":
                    gparams = {"p": 0.10, "seed": args.seed}
                elif gname == "small_world":
                    gparams = {"k": 6, "p": 0.05, "seed": args.seed}
                elif gname == "scale_free":
                    gparams = {"m": 2, "seed": args.seed}
                else:
                    gparams = {"seed": args.seed}
                for s in range(args.samples):
                    gparams["seed"] = int(args.seed + s * 1009)
                    G = generate_graph(gname, n, gparams)
                    for trial in trials:
                        res = run_one(G, args.license, trial, seed=int(args.seed + s))
                        res_row = {"graph": gname, **res}
                        res_row["params"] = str(res_row["params"])  # make CSV-friendly
                        w.writerow(res_row)

    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()

