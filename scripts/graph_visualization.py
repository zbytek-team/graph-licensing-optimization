from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import networkx as nx

from glopt.core.run import generate_graph, instantiate_algorithms
from glopt.io.graph_visualizer import GraphVisualizer
from glopt.license_config import LicenseConfigFactory


def _int_env(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    if v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def _cap_graph_for_drawing(G: nx.Graph, cap: int, seed: int = 42) -> nx.Graph:
    n = G.number_of_nodes()
    if cap <= 0 or n <= cap:
        return G
    import random as _r

    _r.seed(seed)
    nodes = list(G.nodes())
    ego_candidates = [u for u, d in G.nodes(data=True) if d.get("is_ego")]
    keep = set(ego_candidates[:1])
    by_deg = sorted((u for u in nodes if u not in keep), key=lambda u: G.degree(u), reverse=True)
    keep.update(by_deg[: max(0, cap - len(keep) - 50)])
    remaining = [u for u in nodes if u not in keep]
    _r.shuffle(remaining)
    keep.update(remaining[: max(0, cap - len(keep))])
    return G.subgraph(keep).copy()


def solve_and_visualize_graph(
    graph: nx.Graph,
    *,
    license_config: str = "duolingo_super",
    algorithm: str = "auto",
    out_path: Path | None = None,
    ilp_threshold: int = 100,
    time_limit: int | None = None,
    layout_seed: int = 42,
    fig_max_draw_nodes: int | None = None,
) -> Path:
    if fig_max_draw_nodes is None:
        fig_max_draw_nodes = _int_env("FIG_MAX_DRAW_NODES", 0)

    if fig_max_draw_nodes and graph.number_of_nodes() > fig_max_draw_nodes:
        graph = _cap_graph_for_drawing(graph, fig_max_draw_nodes, seed=layout_seed)

    n = graph.number_of_nodes()
    algo_name = algorithm
    if algorithm == "auto":
        algo_name = "ILPSolver" if n <= ilp_threshold else "GreedyAlgorithm"

    algo = instantiate_algorithms([algo_name])[0]
    license_types = LicenseConfigFactory.get_config(license_config)

    kwargs: dict[str, Any] = {}
    if isinstance(algo_name, str) and algo_name.lower().startswith("ilp"):
        if time_limit is not None:
            kwargs["time_limit"] = int(time_limit)

    solution = algo.solve(graph=graph, license_types=license_types, **kwargs)

    viz = GraphVisualizer(layout_seed=layout_seed)
    if out_path is None:
        out_dir = Path("results/graph_visualization")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"viz_{algo.name}_{n}n_{graph.number_of_edges()}e.png"
    saved = viz.visualize_solution(
        graph=graph,
        solution=solution,
        solver_name=algo.name,
        timestamp_folder=None,
        save_path=str(out_path),
    )
    return Path(saved)


def parse_kv_params(param_list: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in param_list:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if v.lower() in {"true", "false"}:
            out[k] = v.lower() == "true"
            continue
        try:
            if "." in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
            continue
        except Exception:
            pass
        out[k] = v
    return out


def build_graph(name: str, n_nodes: int, params: dict[str, Any]) -> nx.Graph:
    if name in {"random", "small_world", "scale_free", "tree"}:
        return generate_graph(name, n_nodes, params)
    if name == "facebook_ego":
        from glopt.io.data_loader import RealWorldDataLoader

        ego = str(params.get("ego", params.get("id", "107")))
        return RealWorldDataLoader(data_dir=str(params.get("data_dir", "data"))).load_facebook_ego_network(ego)
    raise ValueError(f"Unsupported graph name: {name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize license assignments on a graph")
    ap.add_argument("--graph", default="random", choices=["random", "small_world", "scale_free", "tree", "facebook_ego"], help="graph type")
    ap.add_argument("--n", type=int, default=50, help="number of nodes (synthetic)")
    ap.add_argument("--param", action="append", default=[], help="extra generator params key=val (repeat)")
    ap.add_argument("--license-config", default="duolingo_super", help="license configuration name")
    ap.add_argument("--algorithm", default="auto", help="algorithm name or 'auto'")
    ap.add_argument("--ilp-threshold", type=int, default=100, help="use ILP if n <= threshold")
    ap.add_argument("--time-limit", type=int, default=None, help="ILP time limit (seconds)")
    ap.add_argument("--seed", type=int, default=42, help="layout/generator seed")
    ap.add_argument("--out", default=None, help="output image path")
    args = ap.parse_args()

    params = parse_kv_params(list(args.param))
    G = build_graph(args.graph, args.n, params)

    out_path = Path(args.out) if args.out else None
    save_to = solve_and_visualize_graph(
        G,
        license_config=args.license_config,
        algorithm=args.algorithm,
        out_path=out_path,
        ilp_threshold=args.ilp_threshold,
        time_limit=args.time_limit,
        layout_seed=args.seed,
    )
    print(save_to)


if __name__ == "__main__":
    main()

