#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx
import pickle

from glopt.io.graph_generator import GraphGeneratorFactory


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def adjust_params(name: str, n: int, params: dict[str, Any]) -> dict[str, Any]:
    p = dict(params)
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


def save_graph(g: nx.Graph, meta: dict[str, Any], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with out_path.open("wb") as f:
        pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)
    meta_path = out_path.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-generate and cache synthetic graphs (ER, BA, WS) for benchmarks.")
    ap.add_argument("--out-dir", default="data/graphs_cache", help="Base directory for cached graphs")
    ap.add_argument("--graph-types", default="random,scale_free,small_world", help="Comma-separated list of graph types")
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--max-n", type=int, default=3000)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--samples", type=int, default=4, help="Number of samples per (type,n)")
    ap.add_argument("--base-seed", type=int, default=42, help="Base seed used to derive per-sample seeds")
    # Defaults aligned with benchmark GRAPH_DEFAULTS
    ap.add_argument("--random-p", type=float, default=0.10)
    ap.add_argument("--scale-free-m", type=int, default=2)
    ap.add_argument("--small-world-k", type=int, default=6)
    ap.add_argument("--small-world-p", type=float, default=0.05)
    args = ap.parse_args()

    out_base = Path(args.out_dir)
    types = [t.strip() for t in args.graph_types.split(",") if t.strip()]

    defaults: dict[str, dict[str, Any]] = {
        "random": {"p": args.random_p},
        "scale_free": {"m": args.scale_free_m},
        "small_world": {"k": args.small_world_k, "p": args.small_world_p},
    }

    gens = {name: GraphGeneratorFactory.get(name) for name in types}

    sizes = list(range(args.min_n, args.max_n + 1, args.step))
    print(f"Generating cache under {out_base} for types={types}, sizes={sizes[0]}..{sizes[-1]} step {args.step}, samples={args.samples}")

    total = 0
    for name in types:
        for n in sizes:
            base_params = defaults.get(name, {})
            base_params = adjust_params(name, n, base_params)
            for s in range(args.samples):
                # Derive deterministic seed per (type,n,sample)
                seed = int(args.base_seed + s + 100000 * hash((name, n)) % 10_000)
                params = dict(base_params)
                params["seed"] = seed
                out_path = out_base / name / f"n{n:04d}" / f"s{s}.gpickle"
                if out_path.exists():
                    # Skip existing
                    continue
                g = gens[name](n_nodes=n, **params)
                meta = {"type": name, "n": n, "params": params, "sample": s}
                save_graph(g, meta, out_path)
                total += 1
                if total % 50 == 0:
                    print(f"  saved {total} graphs so far … (last: {name} n={n} s={s})")

    print(f"Done. Saved new graphs: {total}")


if __name__ == "__main__":
    main()
