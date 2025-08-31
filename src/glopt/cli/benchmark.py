from __future__ import annotations

from datetime import datetime
from time import perf_counter
from typing import Any

import networkx as nx

from glopt import algorithms
from glopt.core import RunResult, generate_graph, instantiate_algorithms
from glopt.core.solution_validator import SolutionValidator
from glopt.io import build_paths, ensure_dir
from glopt.license_config import LicenseConfigFactory

# Configuration
RUN_ID: str | None = None
GRAPH_NAMES: list[str] = ["random", "scale_free", "small_world"]
GRAPH_DEFAULTS: dict[str, dict[str, Any]] = {
    "random": {"p": 0.10, "seed": 42},
    "scale_free": {"m": 2, "seed": 42},
    "small_world": {"k": 6, "p": 0.05, "seed": 42},
}
# Keep the benchmark short by default during development.
# Ends before or at 61 (10, 30, 50 with step=20).
SIZES: list[int] = list(range(10, 1001, 20))
LICENSE_CONFIG_NAMES: list[str] = ["spotify", "duolingo_super", "roman_domination"]
ALGO_NAMES: list[str] = list(algorithms.__all__)
TIMEOUT_SECONDS: float = 90.0


def _params(name: str, n: int) -> dict[str, Any]:
    p = dict(GRAPH_DEFAULTS.get(name, {}))
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


def main() -> None:
    run_id = RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")
    _, graphs_dir, csv_dir = build_paths(run_id)
    ensure_dir(csv_dir)

    print("== glopt benchmark ==")
    print(f"run_id: {run_id}")
    print(f"sizes: {SIZES[0]}..{SIZES[-1]} step {SIZES[1]-SIZES[0]}")
    print(f"graphs: {', '.join(GRAPH_NAMES)}")
    print(f"licenses: {', '.join(LICENSE_CONFIG_NAMES)}")
    print(f"algorithms: {', '.join(ALGO_NAMES)}")

    for lic in LICENSE_CONFIG_NAMES:
        license_types = LicenseConfigFactory.get_config(lic)
        for algo_name in ALGO_NAMES:
            algo = instantiate_algorithms([algo_name])[0]
            rows: list[RunResult] = []
            print(f"-> {lic} / {algo.name}")
            for gname in GRAPH_NAMES:
                for n in SIZES:
                    params = _params(gname, n)
                    G = generate_graph(gname, n, params)
                    t0 = perf_counter()
                    sol = algo.solve(G, license_types)
                    elapsed_ms = (perf_counter() - t0) * 1000.0
                    ok, issues = SolutionValidator(debug=False).validate(sol, G)
                    density = nx.density(G)
                    avg_deg = sum(dict(G.degree()).values()) / max(1, G.number_of_nodes())
                    notes = f"density={density:.6f};avg_deg={avg_deg:.3f}"
                    rows.append(
                        RunResult(
                            run_id=run_id,
                            algorithm=algo.name,
                            graph=gname,
                            n_nodes=G.number_of_nodes(),
                            n_edges=G.number_of_edges(),
                            graph_params=str(params),
                            license_config=lic,
                            total_cost=float(sol.total_cost),
                            time_ms=float(elapsed_ms),
                            valid=ok,
                            issues=len(issues),
                            image_path="",
                            notes=notes,
                        ),
                    )
                    print(f"   {gname:12s} n={n:4d} cost={sol.total_cost:.2f} time_ms={elapsed_ms:.2f} valid={ok}")
                    if elapsed_ms / 1000.0 > TIMEOUT_SECONDS:
                        print("   timeout reached -> skipping larger sizes for this graph")
                        break
            # write per license+algo CSV
            from pathlib import Path
            import csv

            safe_lic = "".join(c if c.isalnum() or c in "-_" else "_" for c in lic)
            out_path = Path(csv_dir) / f"{run_id}_{safe_lic}_{algo.name}.csv"
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].__dict__.keys()) if rows else [])
                if rows:
                    writer.writeheader()
                    for r in rows:
                        writer.writerow(r.__dict__)
            print(f"csv: {out_path}")


if __name__ == "__main__":
    main()
