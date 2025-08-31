import argparse
import sys
import traceback
from datetime import datetime
from typing import Any

from glopt.core import RunResult, generate_graph, instantiate_algorithms, run_once
from glopt.io import build_paths, write_csv
from glopt.license_config import LicenseConfigFactory


def _auto(value: str) -> object:
    vl = value.lower()
    if vl in {"true", "false"}:
        return vl == "true"
    if vl in {"none", "null"}:
        return None
    try:
        return int(value)
    except Exception:
        try:
            return float(value)
        except Exception:
            return value


def _parse_params(pairs: list[str]) -> dict[str, object]:
    params: dict[str, object] = {}
    for p in pairs:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        params[k.strip()] = _auto(v.strip())
    return params


def _parse_algos(arg: list[str] | None, default: list[str]) -> list[str]:
    if not arg:
        return default
    out: list[str] = []
    for item in arg:
        parts = [x.strip() for x in item.split(",") if x.strip()]
        out.extend(parts)
    return out or default


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run selected algorithms on a generated graph with a chosen license config.")
    p.add_argument("--graph-name", "-g", default="small_world", help="Graph generator name (e.g., small_world, random, scale_free).")
    p.add_argument("--n-nodes", "-n", type=int, default=100, help="Number of nodes for the generated graph.")
    p.add_argument(
        "--param",
        dest="params",
        action="append",
        default=["p=0.05", "seed=42"],
        help="Graph parameter key=value. Repeatable. Example: --param p=0.1 --param k=6",
    )
    p.add_argument("--license", "-l", default="spotify", help="License config name.")
    p.add_argument(
        "--algo",
        dest="algos",
        action="append",
        default=["GreedyAlgorithm,SimulatedAnnealing,AntColonyOptimization"],
        help="Algorithm class names (comma-separated). Repeatable.",
    )
    p.add_argument("--run-id", default=None, help="Custom run id (defaults to current timestamp).")
    p.add_argument("--print-issue-limit", type=int, default=20, help="Max issues to print per run (use 0 to suppress).")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    graph_name: str = args.graph_name
    n_nodes: int = args.n_nodes
    graph_params: dict[str, Any] = _parse_params(args.params or [])
    license_config_name: str = args.license
    algorithms_list: list[str] = _parse_algos(args.algos, ["GreedyAlgorithm"])  # default safety
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    print_issue_limit: int | None = None if args.print_issue_limit <= 0 else args.print_issue_limit

    _, graphs_dir, csv_dir = build_paths(run_id)

    try:
        graph = generate_graph(graph_name, n_nodes, graph_params)
    except Exception:
        traceback.print_exc(limit=10, file=sys.stderr)
        sys.exit(2)

    try:
        license_types = LicenseConfigFactory.get_config(license_config_name)
    except Exception:
        traceback.print_exc(limit=10, file=sys.stderr)
        sys.exit(2)

    try:
        algos = instantiate_algorithms(algorithms_list)
    except Exception:
        traceback.print_exc(limit=10, file=sys.stderr)
        sys.exit(2)

    results: list[RunResult] = []
    for algo in algos:
        r = run_once(
            algo=algo,
            graph=graph,
            license_types=license_types,
            run_id=run_id,
            graphs_dir=graphs_dir,
            print_issue_limit=print_issue_limit,
        )
        r = RunResult(
            **{
                **r.__dict__,
                "graph": graph_name,
                "graph_params": str(graph_params),
                "license_config": license_config_name,
            },
        )
        results.append(r)

    write_csv(csv_dir, run_id, results)


if __name__ == "__main__":
    main()

