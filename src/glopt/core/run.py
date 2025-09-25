from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import networkx as nx

from . import algorithms
from .io.graph_generator import GraphGeneratorFactory
from .io.graph_visualizer import GraphVisualizer
from .solution_validator import SolutionValidator

if TYPE_CHECKING:
    from glopt.core.models import Algorithm, LicenseType, Solution


@dataclass(frozen=True)
class RunResult:
    run_id: str
    algorithm: str
    graph: str
    n_nodes: int
    n_edges: int
    license_config: str
    total_cost: float
    time_ms: float
    valid: bool
    issues: int
    image_path: str
    notes: str = ""


def generate_graph(
    name: str, n_nodes: int, params: dict[str, Any]
) -> nx.Graph:
    gen = GraphGeneratorFactory.get(name)
    g = gen(n_nodes=n_nodes, **params)
    if not all(isinstance(v, int) for v in g.nodes()):
        mapping = {v: i for i, v in enumerate(g.nodes())}
        g = nx.relabel_nodes(g, mapping, copy=True)
    return g


def instantiate_algorithms(names: list[str]) -> list[Algorithm]:
    loaded: list[Algorithm] = []
    missing: list[str] = []
    for name in names:
        cls = getattr(algorithms, name, None)
        if cls is None:
            missing.append(name)
        else:
            loaded.append(cls())
    if missing:
        avail = ", ".join(getattr(algorithms, "__all__", []))
        msg = f"unknown algorithms: {', '.join(missing)}; available: {avail}"
        raise ValueError(msg)
    if not loaded:
        msg = "no algorithms selected"
        raise ValueError(msg)
    return loaded


def run_once(
    algo: Algorithm,
    graph: nx.Graph,
    license_types: list[LicenseType],
    run_id: str,
    graphs_dir: str,
) -> RunResult:
    validator = SolutionValidator(debug=False)
    visualizer = GraphVisualizer(figsize=(12, 8))
    t0 = perf_counter()
    solution: Solution = algo.solve(graph=graph, license_types=license_types)
    elapsed_ms = (perf_counter() - t0) * 1000.0
    ok, issues = validator.validate(solution, graph)
    img_name = f"{algo.name}_{graph.number_of_nodes()}n_" \
        f"{graph.number_of_edges()}e.png"
    img_path = str(Path(graphs_dir) / img_name)
    try:
        visualizer.visualize_solution(
            graph=graph,
            solution=solution,
            solver_name=algo.name,
            timestamp_folder=run_id,
            save_path=img_path,
        )
    except Exception:
        img_path = ""
    return RunResult(
        run_id=run_id,
        algorithm=algo.name,
        graph="?",
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        license_config="?",
        total_cost=float(solution.total_cost),
        time_ms=elapsed_ms,
        valid=ok,
        issues=len(issues),
        image_path=img_path,
        notes="",
    )
