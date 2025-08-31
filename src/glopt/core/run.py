from __future__ import annotations

import os
import sys
import traceback
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List

import networkx as nx

from .models import Algorithm, LicenseType, Solution
from .solution_validator import SolutionValidator
from ..io.graph_visualizer import GraphVisualizer
from ..io.graph_generator import GraphGeneratorFactory
from .. import algorithms


@dataclass(frozen=True)
class RunResult:
    run_id: str
    algorithm: str
    graph: str
    n_nodes: int
    n_edges: int
    graph_params: str
    license_config: str
    total_cost: float
    time_ms: float
    valid: bool
    issues: int
    image_path: str
    notes: str = ""


def generate_graph(name: str, n_nodes: int, params: Dict[str, Any]) -> nx.Graph:
    gen = GraphGeneratorFactory.get(name)
    G = gen(n_nodes=n_nodes, **params)
    if not all(isinstance(v, int) for v in G.nodes()):
        mapping = {v: i for i, v in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping, copy=True)
    return G


def instantiate_algorithms(names: List[str]) -> List[Algorithm]:
    loaded: List[Algorithm] = []
    missing: List[str] = []
    for name in names:
        cls = getattr(algorithms, name, None)
        if cls is None:
            missing.append(name)
        else:
            loaded.append(cls())
    if missing:
        avail = ", ".join(getattr(algorithms, "__all__", []))
        raise ValueError(f"unknown algorithms: {', '.join(missing)}; available: {avail}")
    if not loaded:
        raise ValueError("no algorithms selected")
    return loaded


def run_once(
    algo: Algorithm,
    graph: nx.Graph,
    license_types: List[LicenseType],
    run_id: str,
    graphs_dir: str,
    print_issue_limit: int | None = 20,
) -> RunResult:
    validator = SolutionValidator(debug=False)
    visualizer = GraphVisualizer(figsize=(12, 8))

    try:
        t0 = perf_counter()
        solution: Solution = algo.solve(graph=graph, license_types=license_types)
        elapsed_ms = (perf_counter() - t0) * 1000.0
    except Exception as e:
        algo_name = getattr(algo, "name", algo.__class__.__name__)
        print(f"[ERROR] solver crashed: {algo_name}: {e}", file=sys.stderr)
        traceback.print_exc(limit=20, file=sys.stderr)
        return RunResult(
            run_id=run_id,
            algorithm=algo_name,
            graph="?",
            n_nodes=graph.number_of_nodes(),
            n_edges=graph.number_of_edges(),
            graph_params="{}",
            license_config="?",
            total_cost=float("nan"),
            time_ms=0.0,
            valid=False,
            issues=1,
            image_path="",
            notes=f"solver_error: {e}",
        )

    ok, issues = validator.validate(solution, graph)
    if not ok:
        print(f"[VALIDATION] {algo.name}: {len(issues)} issue(s):", file=sys.stderr)
        to_show = issues if print_issue_limit is None else issues[:print_issue_limit]
        for i in to_show:
            print(f"  - {i.code}: {i.msg}", file=sys.stderr)
        if print_issue_limit is not None and len(issues) > print_issue_limit:
            print(f"  ... {len(issues) - print_issue_limit} more", file=sys.stderr)

    img_name = f"{algo.name}_{graph.number_of_nodes()}n_{graph.number_of_edges()}e.png"
    img_path = os.path.join(graphs_dir, img_name)
    try:
        visualizer.visualize_solution(
            graph=graph,
            solution=solution,
            solver_name=algo.name,
            timestamp_folder=run_id,
            save_path=img_path,
        )
    except Exception as e:
        print(f"[WARN] failed to save image for {algo.name}: {e}", file=sys.stderr)
        traceback.print_exc(limit=10, file=sys.stderr)
        img_path = ""

    return RunResult(
        run_id=run_id,
        algorithm=algo.name,
        graph="?",
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        graph_params="{}",
        license_config="?",
        total_cost=float(solution.total_cost),
        time_ms=elapsed_ms,
        valid=ok,
        issues=len(issues),
        image_path=img_path,
        notes="" if ok else "; ".join(f"{i.code}" for i in issues[:5]),
    )
