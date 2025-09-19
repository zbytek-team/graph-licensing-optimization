from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx

from .commons import ensure_dir, load_rows
from .ilp_boundary import plot_ilp_boundary
from .plots_compare_configs import plot_compare_configs
from .plots_cost_time import plot_cost_vs_n, plot_time_vs_n
from .plots_density import plot_density_vs_time
from .plots_heatmap import plot_cost_heatmap
from .plots_pareto import plot_pareto


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


FIG_MAX_DRAW_NODES = _int_env("FIG_MAX_DRAW_NODES", 400)


def _collect_rows(results_dir: Path, runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Preferred combined CSV if present
    for csv_path in [results_dir / "benchmark_all.csv"]:
        if csv_path.exists():
            before = len(rows)
            rows.extend(load_rows(csv_path))
            print(f"[load] {csv_path} rows=+{len(rows) - before} total={len(rows)}", flush=True)
    # Fallback/augment from any run CSVs
    for csv in sorted((runs_dir).glob("*/csv/*.csv")):
        try:
            before = len(rows)
            rows.extend(load_rows(csv))
            print(f"[load] {csv} rows=+{len(rows) - before} total={len(rows)}", flush=True)
        except Exception:
            pass
    return rows


def _filter(rows: list[dict[str, Any]], **conds: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        ok = True
        for k, v in conds.items():
            if str(r.get(k, "")) != str(v):
                ok = False
                break
        if ok:
            out.append(r)
    return out


def _graph_defaults(name: str) -> dict[str, Any]:
    if name == "random":
        return {"p": 0.10, "seed": 42}
    if name == "small_world":
        return {"k": 6, "p": 0.05, "seed": 42}
    if name == "scale_free":
        return {"m": 2, "seed": 42}
    if name == "tree":
        return {"seed": 42}
    return {}


def _draw_plain_graph(G: nx.Graph, out_path: Path, seed: int = 42) -> None:
    n = G.number_of_nodes()
    # For larger graphs, limit work by drawing a capped induced subgraph
    if n > FIG_MAX_DRAW_NODES:
        import random as _r

        _r.seed(seed)
        # Keep ego node if present, then pick high-degree nodes, then random fill
        nodes = list(G.nodes())
        ego_candidates = [u for u, d in G.nodes(data=True) if d.get("is_ego")]
        keep = set(ego_candidates[:1])
        by_deg = sorted((u for u in nodes if u not in keep), key=lambda u: G.degree(u), reverse=True)
        keep.update(by_deg[: max(0, FIG_MAX_DRAW_NODES - len(keep) - 50)])
        remaining = [u for u in nodes if u not in keep]
        _r.shuffle(remaining)
        keep.update(remaining[: max(0, FIG_MAX_DRAW_NODES - len(keep))])
        G = G.subgraph(keep).copy()
        n = G.number_of_nodes()
        print(f"[draw] capped large graph to {n} nodes for {out_path.name}", flush=True)

    # Use fewer iterations to avoid long FR layout on larger graphs
    pos = nx.spring_layout(G, seed=seed, iterations=30)
    plt.figure(figsize=(6.4, 4.5))
    nx.draw_networkx_edges(G, pos, alpha=0.6, width=1.0, edge_color="#7f7f7f")
    nx.draw_networkx_nodes(G, pos, node_color="#4F46E5", alpha=0.9, node_size=50)
    plt.axis("off")
    ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _make_example_graphs(out_dir: Path) -> None:
    # Use the dedicated visualization to show license partitions; ILP for small graphs
    from glopt.core import generate_graph
    from scripts.graph_visualization import solve_and_visualize_graph

    t0 = perf_counter()
    print("[step] example graphs -- synthetic (with licenses)", flush=True)
    for name in ["random", "small_world", "scale_free"]:
        params = _graph_defaults(name)
        G = generate_graph(name, 20, params)
        solve_and_visualize_graph(
            G,
            license_config="duolingo_super",
            algorithm="auto",
            ilp_threshold=100,
            layout_seed=42,
            fig_max_draw_nodes=FIG_MAX_DRAW_NODES,
            out_path=out_dir / f"graph_{name}_20.png",
        )
    # Tree without SciPy: derive an MST over a weighted complete graph
    import random as _r

    _r.seed(42)
    K = nx.complete_graph(20)
    for u, v in K.edges():
        K[u][v]["weight"] = _r.random()
    Gtree = nx.minimum_spanning_tree(K)
    solve_and_visualize_graph(
        Gtree,
        license_config="duolingo_super",
        algorithm="auto",
        ilp_threshold=100,
        layout_seed=42,
        fig_max_draw_nodes=FIG_MAX_DRAW_NODES,
        out_path=out_dir / "graph_tree_20.png",
    )

    # Facebook ego example (id 107)
    from glopt.io.data_loader import RealWorldDataLoader

    try:
        print("[step] example graph -- facebook_ego_107 (with licenses)", flush=True)
        t1 = perf_counter()
        Gfb = RealWorldDataLoader(data_dir="data").load_facebook_ego_network("107")
        print(
            f"[load] facebook_ego_107 n={Gfb.number_of_nodes()} e={Gfb.number_of_edges()}",
            flush=True,
        )
        solve_and_visualize_graph(
            Gfb,
            license_config="duolingo_super",
            algorithm="auto",
            ilp_threshold=100,
            layout_seed=7,
            fig_max_draw_nodes=FIG_MAX_DRAW_NODES,
            out_path=out_dir / "graph_facebook_ego_107.png",
        )
        print(f"[done] facebook_ego_107 in {perf_counter() - t1:.2f}s", flush=True)
    except Exception:
        # Skip gracefully if data missing
        pass


def _plot_density_vs_cost(rows: list[dict[str, Any]], title: str, out_path: Path) -> None:
    # Group points by algorithm and scatter once per algo for speed
    by_alg: dict[str, tuple[list[float], list[float]]] = {}
    for r in rows:
        try:
            alg = str(r.get("algorithm", ""))
            d = float(r.get("density", 0.0))
            c = float(r.get("cost_per_node", r.get("total_cost", 0.0)))
        except Exception:
            continue
        if not alg:
            continue
        xs, ys = by_alg.setdefault(alg, ([], []))
        xs.append(d)
        ys.append(c)
    if not by_alg:
        return
    plt.figure(figsize=(6.5, 5))
    for i, (alg, (xs, ys)) in enumerate(sorted(by_alg.items())):
        plt.scatter(xs, ys, s=18, alpha=0.8, label=alg)
    plt.xlabel("density")
    plt.ylabel("cost per node")
    plt.title(title)
    ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_ilp_success_rate(rows: list[dict[str, Any]], out_path: Path) -> None:
    ilp = [r for r in rows if str(r.get("algorithm", "")) in {"ILPSolver", "ilp"}]
    if not ilp:
        return
    # group by n_nodes bins; compute success (non-timeout) fraction
    from statistics import mean

    by_n: dict[int, list[int]] = {}
    for r in ilp:
        try:
            n = int(float(r.get("n_nodes", 0)))
        except Exception:
            continue
        note = str(r.get("notes", ""))
        succ = 0 if note == "timeout" else 1
        by_n.setdefault(n, []).append(succ)
    xs = sorted(by_n.keys())
    ys = [mean(by_n[n]) for n in xs]
    plt.figure(figsize=(7, 4.6))
    if xs:
        plt.plot(xs, ys, marker="o")
    plt.ylim(0, 1)
    plt.xlabel("n_nodes")
    plt.ylabel("ILP success rate")
    plt.title("ILP success vs problem size")
    ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def build_new_figures(results_dir: Path, runs_dir: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    print("[step] collecting CSV rows", flush=True)
    t0 = perf_counter()
    rows_all = _collect_rows(results_dir, runs_dir)
    print(f"[done] collected {len(rows_all)} rows in {perf_counter() - t0:.2f}s", flush=True)
    if not rows_all:
        print("No CSV rows found under results/ or runs/*/csv; nothing to do.")
        return

    # Focused subsets
    rows_duo = _filter(rows_all, license_config="duolingo_super")
    rows_roman = _filter(rows_all, license_config="roman_domination")
    # Prefer synthetic graphs for general figures
    synth_graphs = {"random", "small_world", "scale_free", "tree"}
    rows_synth_duo = [r for r in rows_duo if str(r.get("graph", "")) in synth_graphs]
    rows_synth_all = [r for r in rows_all if str(r.get("graph", "")) in synth_graphs]

    # 1) Example graphs
    _make_example_graphs(out_dir)

    # 2) Cost/time vs n for random / scale_free / small_world (duolingo_super)
    print("[step] plots: cost/time vs n (duolingo_super)", flush=True)
    t1 = perf_counter()
    for g in ["random", "scale_free", "small_world"]:
        sub = [r for r in rows_synth_duo if str(r.get("graph", "")) == g]
        if sub:
            plot_cost_vs_n(sub, title=f"duolingo_super -- {g}", out_path=out_dir / f"{g}_cost_vs_n")
            plot_time_vs_n(sub, title=f"duolingo_super -- {g}", out_path=out_dir / f"{g}_time_vs_n")
    print(f"[done] cost/time vs n in {perf_counter() - t1:.2f}s", flush=True)

    # 3) Compare duolingo_super vs roman_domination (overall)
    if rows_duo and rows_roman:
        tmp_dir = out_dir / "_tmp_compare"
        ensure_dir(tmp_dir)
        print("[step] compare duolingo_super vs roman_domination", flush=True)
        plot_compare_configs(rows_all, "duolingo_super", "roman_domination", "new_figs", tmp_dir)
        # Copy to requested simpler names
        src_cost = tmp_dir / "compare_cost_per_node_duolingo_super_vs_roman_domination.png"
        src_time = tmp_dir / "compare_time_ms_duolingo_super_vs_roman_domination.png"
        if src_cost.exists():
            shutil.copy2(src_cost, out_dir / "compare_cost_duo_vs_roman.png")
        if src_time.exists():
            shutil.copy2(src_time, out_dir / "compare_time_duo_vs_roman.png")

    # 4) Heatmap of cost (overall synthetic)
    if rows_synth_all:
        print("[step] heatmap (synthetic)", flush=True)
        plot_cost_heatmap(rows_synth_all, title="Cost heatmap (synthetic)", out_path=out_dir / "heatmap_cost")

    # 5) Density vs cost/time (duo, all synthetic graphs)
    if rows_synth_duo:
        print("[step] density vs cost/time (duo)", flush=True)
        _plot_density_vs_cost(rows_synth_duo, title="density vs cost per node (duo)", out_path=out_dir / "density_vs_cost.png")
        plot_density_vs_time(rows_synth_duo, title="density vs time (duo)", out_path=out_dir / "density_vs_time")

    # 6) Pareto for random / scale_free (duo)
    print("[step] pareto (duo)", flush=True)
    t2 = perf_counter()
    for g in ["random", "scale_free"]:
        sub = [r for r in rows_synth_duo if str(r.get("graph", "")) == g]
        if sub:
            plot_pareto(sub, title=f"Pareto -- {g} (duo)", out_path=out_dir / f"pareto_{g}")
    print(f"[done] pareto in {perf_counter() - t2:.2f}s", flush=True)

    # 7) Facebook results -- use runs benchmark_real if present; otherwise skip
    rows_real = [r for r in rows_all if str(r.get("graph", "")) == "facebook_ego"]
    # If leaderboards already exist in results, copy them to simple names
    for src, dst in [
        (results_dir / "benchmark_real_all" / "all" / "leaderboard_cost.png", out_dir / "facebook_cost_comparison.png"),
        (results_dir / "benchmark_real_all" / "all" / "leaderboard_time.png", out_dir / "facebook_time_comparison.png"),
    ]:
        if src.exists():
            shutil.copy2(src, dst)

    # 8) ILP limits (synthetic)
    if rows_synth_all:
        print("[step] ILP limits/success", flush=True)
        plot_ilp_boundary(rows_synth_all, out_dir)
        _plot_ilp_success_rate(rows_synth_all, out_dir / "ilp_success_rate.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="path to results dir")
    ap.add_argument("--runs", default="runs", help="path to runs dir")
    ap.add_argument("--out", default="results/new_figures", help="output dir for new figures")
    args = ap.parse_args()

    results_dir = Path(args.results)
    runs_dir = Path(args.runs)
    out_dir = Path(args.out)

    build_new_figures(results_dir, runs_dir, out_dir)
    print(f"New figures written to {out_dir}")


if __name__ == "__main__":
    main()
