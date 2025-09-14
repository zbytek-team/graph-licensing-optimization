from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib.pyplot as plt

from .commons import GENERATE_PDF, ensure_dir, load_rows
from .plots_compare_configs import plot_compare_configs
from .plots_license_mix import plot_license_mix


def _roman_p_value(name: str) -> float | None:
    # roman_p_1_5 -> 1.5
    if not name.startswith("roman_p_"):
        return None
    tail = name[len("roman_p_"):]
    try:
        return float(tail.replace("_", "."))
    except Exception:
        return None


def aggregate_by(rows: list[dict[str, Any]], key_fields: tuple[str, ...], metric: str) -> dict[tuple, float]:
    # mean over instances for given key tuple
    agg: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        try:
            key = tuple(str(r.get(k, "")) for k in key_fields)
            v = float(r.get(metric, "nan"))
            if not (v == v) or math.isinf(v):
                continue
            agg[key].append(v)
        except Exception:
            continue
    return {k: mean(vs) for k, vs in agg.items() if vs}


def plot_configs_grouped(rows: list[dict[str, Any]], configs: list[str], title: str, out_path: Path, metric: str = "cost_per_node") -> None:
    # group by (algorithm) and plot bars per config
    per_alg_cfg: dict[str, dict[str, float]] = defaultdict(dict)
    for cfg in configs:
        sub = [r for r in rows if str(r.get("license_config", "")) == cfg]
        a = aggregate_by(sub, ("algorithm",), metric)
        for (alg,), v in a.items():
            per_alg_cfg[alg][cfg] = v
    algs = sorted(per_alg_cfg.keys())
    if not algs:
        return
    x = range(len(algs))
    width = max(0.1, min(0.8 / max(1, len(configs)), 0.25))
    plt.figure(figsize=(max(6.5, 0.65 * len(algs)), 5))
    for j, cfg in enumerate(configs):
        vals = [per_alg_cfg.get(alg, {}).get(cfg, float("nan")) for alg in algs]
        xs = [i + (j - (len(configs) - 1) / 2) * width for i in x]
        plt.bar(xs, vals, width=width, label=cfg)
    plt.xticks(list(x), algs, rotation=30, ha="right")
    ylabel = metric.replace("_", " ")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=220)
    if GENERATE_PDF:
        plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def plot_roman_p_curve(rows: list[dict[str, Any]], title: str, out_dir: Path, metric: str = "cost_per_node") -> None:
    # x-axis = p value, y = mean metric per algorithm
    ensure_dir(out_dir)
    # collect per (alg, p)
    by_ap: dict[tuple[str, float], list[float]] = defaultdict(list)
    for r in rows:
        name = str(r.get("license_config", ""))
        p = _roman_p_value(name)
        if p is None:
            continue
        alg = str(r.get("algorithm", ""))
        try:
            v = float(r.get(metric, "nan"))
        except Exception:
            continue
        if not alg or not (v == v):
            continue
        by_ap[(alg, p)].append(v)
    algs = sorted({a for a, _ in by_ap.keys()})
    if not algs:
        return
    plt.figure(figsize=(8.5, 5))
    for alg in algs:
        ps = sorted({p for (a, p) in by_ap.keys() if a == alg})
        ys = [mean(by_ap[(alg, p)]) for p in ps]
        if ps:
            plt.plot(ps, ys, marker="o", label=alg)
    plt.xlabel("parametr p (roman_p)")
    plt.ylabel(metric.replace("_", " "))
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    out_path = out_dir / f"roman_p_curve_{metric}"
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=220)
    if GENERATE_PDF:
        plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True, help="ścieżka do CSV z runs/")
    ap.add_argument("--out", required=True, help="katalog wyjściowy na podsumowania")
    ap.add_argument("--figdir", default="docs/thesis/assets/figures/extensions", help="gdzie zapisać wykresy")
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for p in args.csv:
        rows.extend(load_rows(Path(p)))

    fig_root = Path(args.figdir)
    ensure_dir(fig_root)

    # 1) Roman variants: grouped comparison vs duolingo_super and vs roman_domination
    romans = ["roman_domination", "roman_p_1_5", "roman_p_2_0", "roman_p_2_5", "roman_p_3_0"]
    present_romans = [c for c in romans if any(str(r.get("license_config")) == c for r in rows)]
    if present_romans:
        plot_configs_grouped(rows, present_romans, title="Roman variants — cost per node (by algorithm)", out_path=fig_root / "roman_variants_cost", metric="cost_per_node")
        plot_configs_grouped(rows, present_romans, title="Roman variants — time [ms] (by algorithm)", out_path=fig_root / "roman_variants_time", metric="time_ms")
        plot_roman_p_curve(rows, title="roman_p: cost per node vs parameter p", out_dir=fig_root)

    # 2) Spotify vs Duolingo: grouped comparison and license mix
    cfgA, cfgB = "duolingo_super", "spotify"
    if any(str(r.get("license_config")) == cfgB for r in rows):
        # Compare metrics
        plot_compare_configs(rows, cfgA, cfgB, title_prefix="extensions", out_dir=fig_root / "duo_vs_spotify")
        # License mix per config
        sub_duo = [r for r in rows if str(r.get("license_config")) == cfgA]
        sub_spo = [r for r in rows if str(r.get("license_config")) == cfgB]
        plot_license_mix(sub_duo, title="duolingo_super — license mix by algorithm", out_path=fig_root / "duolingo_super_license_mix")
        plot_license_mix(sub_spo, title="spotify — license mix by algorithm", out_path=fig_root / "spotify_license_mix")

    # 3) If other duolingo variants appear in future runs, compare them automatically
    duo_like = sorted({str(r.get("license_config")) for r in rows if str(r.get("license_config", "")).startswith("duolingo_")})
    duo_ext = [c for c in duo_like if c != "duolingo_super"]
    if duo_ext:
        plot_configs_grouped(rows, ["duolingo_super"] + duo_ext, title="Duolingo variants — cost per node (by algorithm)", out_path=fig_root / "duolingo_variants_cost", metric="cost_per_node")
        plot_configs_grouped(rows, ["duolingo_super"] + duo_ext, title="Duolingo variants — time [ms] (by algorithm)", out_path=fig_root / "duolingo_variants_time", metric="time_ms")


if __name__ == "__main__":
    main()

