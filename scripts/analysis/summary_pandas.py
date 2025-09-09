from __future__ import annotations

from pathlib import Path

from .commons import ensure_dir


def write_pandas_summaries(csv_path: Path, out_dir: Path) -> None:
    try:
        import pandas as pd  # type: ignore
        import numpy as np
    except Exception:  # pandas optional
        return

    df = pd.read_csv(csv_path)
    # Derived metrics
    if "n_nodes" in df.columns and "total_cost" in df.columns:
        df["cost_per_node"] = df["total_cost"] / df["n_nodes"].clip(lower=1)

    # Basic group stats by (algorithm, graph, n_nodes)
    group_cols = [c for c in ["algorithm", "graph", "n_nodes", "license_config"] if c in df.columns]
    if not group_cols:
        return
    agg = df.groupby(group_cols).agg(
        cost_mean=("total_cost", "mean"),
        cost_std=("total_cost", "std"),
        time_ms_mean=("time_ms", "mean"),
        time_ms_std=("time_ms", "std"),
        cost_per_node_mean=("cost_per_node", "mean") if "cost_per_node" in df.columns else ("total_cost", "mean"),
        rep=("total_cost", "count"),
    )
    agg = agg.reset_index()
    # 95% CI
    agg["cost_ci95"] = 1.96 * (agg["cost_std"].fillna(0) / np.sqrt(agg["rep"].clip(lower=1)))
    agg["time_ms_ci95"] = 1.96 * (agg["time_ms_std"].fillna(0) / np.sqrt(agg["rep"].clip(lower=1)))

    ensure_dir(out_dir)
    out1 = out_dir / "summary_by_algo_graph_n.csv"
    agg.to_csv(out1, index=False)

    # Pivot tables for quick inspection
    if "license_config" in df.columns:
        piv = agg.pivot_table(index=["algorithm", "graph"], columns="license_config", values="cost_per_node_mean")
        piv.to_csv(out_dir / "pivot_cost_per_node.csv")

