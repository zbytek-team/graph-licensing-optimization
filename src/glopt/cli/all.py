# test_all.py
import os
import sys
import io
import traceback
from contextlib import contextmanager
from datetime import datetime
from math import isnan
from typing import Any, Dict, List

from glopt.core import RunResult, generate_graph, instantiate_algorithms, run_once
from glopt.io import build_paths, ensure_dir, write_csv
from glopt import algorithms
from glopt.io.graph_generator import GraphGeneratorFactory
from glopt.license_config import LicenseConfigFactory

# ===== CONFIG =====

N_NODES = 100
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


@contextmanager
def suppress_trace_output():
    """mute traceback.print_exc and stderr prints inside the block"""
    orig_print_exc = traceback.print_exc
    orig_stderr = sys.stderr
    try:
        traceback.print_exc = lambda *a, **k: None  # type: ignore[assignment]
        sys.stderr = io.StringIO()
        yield
    finally:
        traceback.print_exc = orig_print_exc  # type: ignore[assignment]
        sys.stderr = orig_stderr


def _err(msg: str, e: Exception) -> None:
    brief = "".join(traceback.format_exception_only(type(e), e)).strip()
    print(f"[ERROR] {msg}: {brief}", file=sys.stderr)


def _fmt_status(valid: bool) -> str:
    return "ok" if valid else "invalid"


def _print_table(title: str, headers: List[str], rows: List[List[str]]) -> None:
    if not rows:
        print(f"\n[LICENSE] {title} (no runs)")
        return

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    def line(sep_left: str = "+", sep_mid: str = "+", sep_right: str = "+", fill: str = "-") -> str:
        return sep_left + sep_mid.join(fill * (w + 2) for w in widths) + sep_right

    def fmt_row(cols: List[str]) -> str:
        padded = [c.ljust(w) for c, w in zip(cols, widths)]
        return "| " + " | ".join(padded) + " |"

    print(f"\n[LICENSE] {title}")
    print(line())
    print(fmt_row(headers))
    print(line(sep_mid="+"))
    for r in rows:
        print(fmt_row(r))
    print(line())


def rank_results(results: List[RunResult]) -> List[tuple[int, RunResult, float]]:
    valid = [r for r in results if r.valid]
    if not valid:
        return [(i + 1, r, float("nan")) for i, r in enumerate(results)]
    ilp = next((r for r in valid if r.algorithm.lower() in {"ilp", "ilpsolver"}), None)
    best_cost = ilp.total_cost if ilp else min(r.total_cost for r in valid)
    ranked = sorted(results, key=lambda r: (not r.valid, r.total_cost))
    out: List[tuple[int, RunResult, float]] = []
    for idx, r in enumerate(ranked, start=1):
        gap = float("nan") if not r.valid or best_cost == 0 else (r.total_cost - best_cost) / best_cost * 100.0
        out.append((idx, r, gap))
    return out


def main() -> int:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _, graphs_dir_root, csv_dir = build_paths(run_id)

    try:
        graph_names = list(GraphGeneratorFactory._GENERATORS.keys())
    except Exception as e:
        _err("loading graph generators", e)
        return 2

    try:
        license_configs = list(LicenseConfigFactory._CONFIGS.keys())
    except Exception as e:
        _err("loading license configs", e)
        return 2

    try:
        algorithm_names = list(algorithms.__all__)
    except Exception as e:
        _err("loading algorithms list", e)
        return 2

    results: List[RunResult] = []
    had_errors = False

    for graph_name in graph_names:
        if graph_name == "complete" or graph_name == "star":  # graphs that take too much time to compute
            continue

        params = DEFAULT_GRAPH_PARAMS.get(graph_name, {})
        print(f"\n[GRAPH] {graph_name} params={params}")

        try:
            graph = generate_graph(graph_name, N_NODES, params)
        except Exception as e:
            _err(f"graph generation failed: {graph_name}", e)
            had_errors = True
            continue

        for lic_name in license_configs:
            try:
                license_types = LicenseConfigFactory.get_config(lic_name)
            except Exception as e:
                _err(f"license config failed: {lic_name}", e)
                had_errors = True
                continue

            g_dir = os.path.join(graphs_dir_root, graph_name, lic_name)
            try:
                ensure_dir(g_dir)
            except Exception as e:
                _err(f"ensure_dir failed: {g_dir}", e)
                had_errors = True
                continue

            table_rows: List[List[str]] = []

            for algo_name in algorithm_names:
                try:
                    algo = instantiate_algorithms([algo_name])[0]
                except Exception as e:
                    _err(f"algorithm setup failed: {algo_name}", e)
                    had_errors = True
                    continue

                try:
                    # zagłuszamy długie tracebacks generowane w run_once lub w solverze
                    with suppress_trace_output():
                        r = run_once(
                            algo=algo,
                            graph=graph,
                            license_types=license_types,
                            run_id=run_id,
                            graphs_dir=g_dir,
                        )
                except Exception as e:
                    _err(f"run failed: algo={algo_name} graph={graph_name} lic={lic_name}", e)
                    had_errors = True
                    continue

                # enrich result
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

            # sort by cost asc, NaN last
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

    try:
        write_csv(csv_dir, run_id, results)
    except Exception as e:
        _err("writing CSV failed", e)
        had_errors = True

    return 1 if had_errors else 0


if __name__ == "__main__":
    sys.exit(main())
