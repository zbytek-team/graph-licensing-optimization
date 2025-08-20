import os
import sys
from datetime import datetime
from math import isnan
from typing import Any, Dict, List

from scripts._common import (
    RunResult,
    build_paths,
    ensure_dir,
    generate_graph,
    instantiate_algorithms,
    print_summary,
    run_once,
    write_csv,
)
from src import algorithms
from src.factories.graph_generator_factory import GraphGeneratorFactory
from src.factories.license_config_factory import LicenseConfigFactory

# ===== CONFIG =====

N_NODES = 50
DEFAULT_GRAPH_PARAMS: Dict[str, Dict[str, Any]] = {
    "random": {"p": 0.1, "seed": 42},
    "scale_free": {"m": 2, "seed": 42},
    "small_world": {"k": 4, "p": 0.1, "seed": 42},
    "complete": {},
    "star": {},
    "path": {},
    "cycle": {},
    "tree": {"seed": 42},
}

# ===== END CONFIG =====


def _fmt_status(valid: bool) -> str:
    return "ok" if valid else "invalid"


def _print_table(title: str, headers: List[str], rows: List[List[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(sep_left: str = "+", sep_mid: str = "+", sep_right: str = "+", fill: str = "-") -> str:
        return sep_left + sep_mid.join(fill * (w + 2) for w in widths) + sep_right

    def fmt_row(cols: List[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"

    print(f"\n[LICENSE] {title}")
    print(line())
    print(fmt_row(headers))
    print(line(sep_mid="+"))
    for r in rows:
        print(fmt_row(r))
    print(line())


def main() -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _, graphs_dir_root, csv_dir = build_paths(run_id)

    graph_names = list(GraphGeneratorFactory._GENERATORS.keys())
    license_configs = list(LicenseConfigFactory._CONFIGS.keys())
    algorithm_names = list(algorithms.__all__)

    results: List[RunResult] = []
    for graph_name in graph_names:
        params = DEFAULT_GRAPH_PARAMS.get(graph_name, {})
        print(f"\n[GRAPH] {graph_name} params={params}")
        try:
            graph = generate_graph(graph_name, N_NODES, params)
        except Exception as e:
            print(f"[ERROR] graph generation failed: {graph_name}: {e}", file=sys.stderr)
            continue

        for lic_name in license_configs:
            try:
                license_types = LicenseConfigFactory.get_config(lic_name)
            except Exception as e:
                print(f"[ERROR] license config failed: {lic_name}: {e}", file=sys.stderr)
                continue

            g_dir = os.path.join(graphs_dir_root, graph_name, lic_name)
            ensure_dir(g_dir)

            table_rows: List[List[str]] = []

            for algo_name in algorithm_names:
                try:
                    algo = instantiate_algorithms([algo_name])[0]
                except Exception as e:
                    print(f"[ERROR] algorithm setup failed: {algo_name}: {e}", file=sys.stderr)
                    continue

                r = run_once(
                    algo=algo,
                    graph=graph,
                    license_types=license_types,
                    run_id=run_id,
                    graphs_dir=g_dir,
                )
                r = RunResult(
                    **{
                        **r.__dict__,
                        "graph": graph_name,
                        "graph_params": str(params),
                        "license_config": lic_name,
                    }
                )
                results.append(r)

                cost_str = f"{r.total_cost:.2f}" if not isnan(r.total_cost) else "NaN"
                time_str = f"{r.time_ms:.2f}"
                table_rows.append([algo_name, cost_str, time_str, _fmt_status(r.valid)])

            def _key(row: List[str]) -> tuple[int, float]:
                try:
                    v = float(row[1])
                    return (0, v)
                except ValueError:
                    return (1, float("inf"))

            table_rows.sort(key=_key)
            _print_table(
                title=f"{lic_name} on {graph_name}",
                headers=["algorithm", "cost", "time_ms", "status"],
                rows=table_rows,
            )

    csv_path = write_csv(csv_dir, run_id, results)
    print_summary(run_id, csv_path, results)


if __name__ == "__main__":
    main()
