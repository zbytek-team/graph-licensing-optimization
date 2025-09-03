from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .commons import GENERATE_PDF, ensure_dir


def plot_dynamic_warm_cold(rows: list[dict[str, Any]], title_prefix: str, out_dir: Path) -> None:
    ensure_dir(out_dir)
    # Group by algorithm and warm_start flag
    by_alg: dict[str, dict[bool, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        try:
            alg = str(r["algorithm"])
            warm = str(r.get("warm_start", "False")) in {"True", "true", "1"}
            by_alg[alg][warm].append(r)
        except Exception:  # robust parsing
            continue

    for alg, modes in by_alg.items():
        plt.figure(figsize=(8, 5))
        for warm in (True, False):
            seq = sorted(modes.get(warm, []), key=lambda x: int(float(x.get("step", 0))))
            xs = [int(float(r.get("step", 0))) for r in seq]
            costs = [float(r.get("total_cost", 0.0)) for r in seq]
            if xs:
                plt.plot(xs, costs, marker="o", label=f"{'warm' if warm else 'cold'}")
        plt.xlabel("step")
        plt.ylabel("total_cost")
        plt.title(f"{title_prefix} — {alg} cost per step")
        plt.legend()
        out = out_dir / f"{alg}_cost_per_step"
        plt.tight_layout()
        plt.savefig(out.with_suffix(".png"), dpi=220)
        if GENERATE_PDF:
            plt.savefig(out.with_suffix(".pdf"))
        plt.close()

        plt.figure(figsize=(8, 5))
        for warm in (True, False):
            seq = sorted(modes.get(warm, []), key=lambda x: int(float(x.get("step", 0))))
            xs = [int(float(r.get("step", 0))) for r in seq]
            times = [float(r.get("time_ms", 0.0)) for r in seq]
            if xs:
                plt.plot(xs, times, marker="o", label=f"{'warm' if warm else 'cold'}")
        plt.xlabel("step")
        plt.ylabel("time_ms")
        plt.title(f"{title_prefix} — {alg} time per step")
        plt.legend()
        out = out_dir / f"{alg}_time_per_step"
        plt.tight_layout()
        plt.savefig(out.with_suffix(".png"), dpi=220)
        if GENERATE_PDF:
            plt.savefig(out.with_suffix(".pdf"))
        plt.close()
