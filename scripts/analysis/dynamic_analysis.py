from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt

from .commons import GENERATE_PDF, ensure_dir, load_rows
from .dynamic_warmcold import plot_dynamic_warm_cold


@dataclass
class WarmAdvantage:
    dataset: str
    license_config: str
    graph: str
    algorithm: str
    steps: int
    median_cost_delta: float  # cold - warm (positive => warm lepszy)
    improved_steps_share: float  # odsetek kroków, gdy warm < cold
    median_time_ratio: float  # warm/cold ( <1 => warm szybszy )


def _to_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _to_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _parse_mutations_cell(text: str) -> dict[str, int]:
    # Expected formats from DynamicNetworkSimulator:
    # - "Added nodes: [..]" / "Removed nodes: [..]"
    # - "Added X edges" / "Removed X edges"
    out = {"nodes_added": 0, "nodes_removed": 0, "edges_added": 0, "edges_removed": 0}
    if not text:
        return out
    for part in str(text).split(";"):
        p = part.strip()
        if not p:
            continue
        if p.startswith("Added nodes:"):
            # count commas + 1 inside brackets
            try:
                inside = p.split(":", 1)[1].strip()
                if inside.startswith("[") and inside.endswith("]"):
                    body = inside[1:-1].strip()
                    out["nodes_added"] += 0 if body == "" else body.count(",") + 1
            except Exception:
                pass
        elif p.startswith("Removed nodes:"):
            try:
                inside = p.split(":", 1)[1].strip()
                if inside.startswith("[") and inside.endswith("]"):
                    body = inside[1:-1].strip()
                    out["nodes_removed"] += 0 if body == "" else body.count(",") + 1
            except Exception:
                pass
        elif p.startswith("Added ") and p.endswith(" edges"):
            try:
                num = int(p.split()[1])
                out["edges_added"] += num
            except Exception:
                pass
        elif p.startswith("Removed ") and p.endswith(" edges"):
            try:
                num = int(p.split()[1])
                out["edges_removed"] += num
            except Exception:
                pass
    return out


def mutation_intensity(rows: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    # Aggregate mean counts per step across all rows
    agg: dict[int, Counter[str]] = defaultdict(Counter)
    cnt: Counter[int] = Counter()
    for r in rows:
        step = _to_int(r.get("step"))
        muts = _parse_mutations_cell(str(r.get("mutations", "")))
        agg[step].update(muts)
        cnt[step] += 1
    out: dict[int, dict[str, float]] = {}
    for step, c in agg.items():
        denom = max(1, cnt[step])
        out[step] = {k: v / denom for k, v in c.items()}
    return out


def normalized_mutation(row: dict[str, Any]) -> float:
    """Return a scalar mutation intensity in (0, +inf) normalized by graph size.
    intensity = 0.5 * (nodes_changed / max(1,n)) + 0.5 * (edges_changed / max(1,m))
    where changed are counts of added+removed in this step.
    """
    try:
        n = int(float(row.get("n_nodes", 0)))
    except Exception:
        n = 0
    try:
        m = int(float(row.get("n_edges", 0)))
    except Exception:
        m = 0
    muts = _parse_mutations_cell(str(row.get("mutations", "")))
    nodes_changed = float(muts.get("nodes_added", 0) + muts.get("nodes_removed", 0))
    edges_changed = float(muts.get("edges_added", 0) + muts.get("edges_removed", 0))
    frac_nodes = nodes_changed / max(1.0, float(n))
    frac_edges = edges_changed / max(1.0, float(m))
    return 0.5 * frac_nodes + 0.5 * frac_edges


def plot_cost_delta_vs_mutation(rows: list[dict[str, Any]], title: str, out_path: Path) -> None:
    # Build mapping (key, step) -> cost_per_node
    def _key(r: dict[str, Any]) -> tuple[str, str, str, str, bool]:
        return (
            str(r.get("run_id", "")),
            str(r.get("license_config", "")),
            str(r.get("graph", "")),
            str(r.get("algorithm", "")),
            str(r.get("warm_start", "False")) in {"True", "true", "1"},
        )

    by_key: dict[tuple[str, str, str, str, bool], dict[int, float]] = defaultdict(dict)
    by_row: dict[tuple[str, str, str, str, bool], dict[int, dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        try:
            step = int(float(r.get("step", 0)))
            cpn = float(r.get("cost_per_node", r.get("total_cost", 0.0)))
            k = _key(r)
            by_key[k][step] = cpn
            by_row[k][step] = r
        except Exception:
            continue

    X: list[float] = []
    Y: list[float] = []
    C: list[str] = []  # color by warm/cold
    for k, series in by_key.items():
        steps = sorted(series.keys())
        for s in steps:
            if s == 0:
                continue
            prev = series.get(s - 1)
            cur = series.get(s)
            if prev is None or cur is None:
                continue
            delta = cur - prev
            mut = normalized_mutation(by_row[k][s])
            X.append(mut)
            Y.append(delta)
            C.append("warm" if k[-1] else "cold")

    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(8.5, 5.0))
    xs = np.array(X, dtype=float)
    ys = np.array(Y, dtype=float)
    colors = np.array(["tab:orange" if c == "warm" else "tab:blue" for c in C])
    plt.scatter(xs, ys, c=colors, alpha=0.55, s=16, edgecolors="none")
    # simple global trend line
    try:
        mask = np.isfinite(xs) & np.isfinite(ys)
        if mask.sum() >= 2:
            a, b = np.polyfit(xs[mask], ys[mask], deg=1)
            xr = np.linspace(xs[mask].min(), xs[mask].max(), 100)
            plt.plot(xr, a * xr + b, color="black", linewidth=1.2, label=f"trend: y={a:.2f}x{b:+.2f}")
    except Exception:
        pass
    plt.xlabel("intensywność mutacji na krok [ułamki węzłów/krawędzi]")
    plt.ylabel("Δ kosztu na wierzchołek (step) [jedn.]")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path.with_suffix(".png"), dpi=220)
    if GENERATE_PDF:
        plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def plot_mutation_intensity(intensity: dict[int, dict[str, float]], title: str, out_path: Path) -> None:
    steps = sorted(intensity.keys())
    series = {
        "nodes_added": [intensity[s].get("nodes_added", 0.0) for s in steps],
        "nodes_removed": [intensity[s].get("nodes_removed", 0.0) for s in steps],
        "edges_added": [intensity[s].get("edges_added", 0.0) for s in steps],
        "edges_removed": [intensity[s].get("edges_removed", 0.0) for s in steps],
    }
    plt.figure(figsize=(9, 5))
    for name, ys in series.items():
        plt.plot(steps, ys, marker="o", linewidth=1.2, label=name)
    plt.xlabel("krok symulacji (step)")
    plt.ylabel("średnia liczba zmian / wiersz")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path.with_suffix(".png"), dpi=220)
    if GENERATE_PDF:
        plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def compute_warm_advantage(rows: list[dict[str, Any]], dataset: str) -> list[WarmAdvantage]:
    # group by (license, graph, alg, step) → warm/cold values
    buckets: dict[tuple[str, str, str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    times: dict[tuple[str, str, str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        lic = str(r.get("license_config", ""))
        g = str(r.get("graph", ""))
        alg = str(r.get("algorithm", ""))
        step = _to_int(r.get("step"))
        w = "warm" if str(r.get("warm_start")) in {"True", "true", "1"} else "cold"
        buckets[(lic, g, alg, step)][w].append(_to_float(r.get("total_cost")))
        times[(lic, g, alg, step)][w].append(_to_float(r.get("time_ms")))

    # reduce to per (lic, graph, alg)
    out: list[WarmAdvantage] = []
    by_key: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for (lic, g, alg, step), d in buckets.items():
        if not d.get("warm") or not d.get("cold"):
            continue
        # median warm/cold cost at this step
        def _med(xs: Iterable[float]) -> float:
            s = sorted([x for x in xs if x == x])
            if not s:
                return float("nan")
            n = len(s)
            return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])

        cw = _med(d["cold"])
        ww = _med(d["warm"])
        tw = _med(times[(lic, g, alg, step)].get("warm", []))
        tc = _med(times[(lic, g, alg, step)].get("cold", []))
        if cw == cw and ww == ww:  # both not NaN
            by_key[(lic, g, alg)].append((cw - ww, (ww / tc) / (cw / tc) if (tc and cw) else (tw / tc if tc else float("nan"))))
            # note: store delta cost and crude time ratio placeholder, real per-step ratio below

    # finalize per (lic, graph, alg)
    for (lic, g, alg), vals in by_key.items():
        if not vals:
            continue
        deltas = [v[0] for v in vals]
        # recompute time ratio properly: median of warm/cold per step
        ratios: list[float] = []
        improved = 0
        steps = len(deltas)
        for (lic2, g2, alg2, step), d in buckets.items():
            if (lic2, g2, alg2) != (lic, g, alg):
                continue
            if not d.get("warm") or not d.get("cold"):
                continue
            def _med(xs: Iterable[float]) -> float:
                s = sorted([x for x in xs if x == x])
                if not s:
                    return float("nan")
                n = len(s)
                return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])
            cw = _med(d["cold"])
            ww = _med(d["warm"])
            tc = _med(times[(lic2, g2, alg2, step)].get("cold", []))
            tw = _med(times[(lic2, g2, alg2, step)].get("warm", []))
            if cw == cw and ww == ww:
                if ww < cw:
                    improved += 1
            if tc and tc == tc and tw == tw and tc > 0:
                ratios.append(tw / tc)
        # median delta and ratio
        deltas_sorted = sorted([d for d in deltas if d == d])
        if not deltas_sorted:
            continue
        n = len(deltas_sorted)
        median_delta = deltas_sorted[n // 2] if n % 2 == 1 else 0.5 * (deltas_sorted[n // 2 - 1] + deltas_sorted[n // 2])
        ratios_sorted = sorted([r for r in ratios if r == r])
        median_ratio = ratios_sorted[len(ratios_sorted) // 2] if ratios_sorted else float("nan")
        out.append(
            WarmAdvantage(
                dataset=dataset,
                license_config=lic,
                graph=g,
                algorithm=alg,
                steps=steps,
                median_cost_delta=median_delta,
                improved_steps_share=(improved / steps) if steps else 0.0,
                median_time_ratio=median_ratio,
            )
        )
    return out


def write_warm_advantage(rows: list[dict[str, Any]], dataset: str, out_csv: Path) -> None:
    ensure_dir(out_csv.parent)
    adv = compute_warm_advantage(rows, dataset)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset",
            "license_config",
            "graph",
            "algorithm",
            "steps",
            "median_cost_delta",
            "improved_steps_share",
            "median_time_ratio",
        ])
        for a in adv:
            w.writerow([
                a.dataset,
                a.license_config,
                a.graph,
                a.algorithm,
                a.steps,
                f"{a.median_cost_delta:.6f}",
                f"{a.improved_steps_share:.6f}",
                f"{a.median_time_ratio:.6f}" if a.median_time_ratio == a.median_time_ratio else "",
            ])


def run_plots_dynamic(rows: list[dict[str, Any]], title_prefix: str, out_root: Path, include_licenses: set[str] | None = None) -> None:
    # Per (license, graph) generate warm/cold plots per algorithm
    lics = sorted({str(r.get("license_config", "")) for r in rows if r.get("license_config")})
    graphs = sorted({str(r.get("graph", "")) for r in rows if r.get("graph")})
    for lic in lics:
        if include_licenses is not None and lic not in include_licenses:
            continue
        for g in graphs:
            sub = [r for r in rows if str(r.get("license_config")) == lic and str(r.get("graph")) == g]
            if not sub:
                continue
            out_dir = out_root / lic / g
            plot_dynamic_warm_cold(sub, f"{title_prefix} — {lic} — {g}", out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True, help="ścieżka do CSV z runs/")
    ap.add_argument("--tag", action="append", required=True, help="etykieta datasetu, np. dynamic20, full")
    ap.add_argument("--out", required=True, help="katalog wyjściowy na wyniki analizy")
    ap.add_argument("--figdir", default="docs/thesis/assets/figures/dynamic", help="gdzie zapisać wykresy")
    args = ap.parse_args()

    csvs = [Path(p) for p in args.csv]
    tags = list(args.tag)
    assert len(csvs) == len(tags), "Liczba --csv musi odpowiadać liczbie --tag"

    out_root = Path(args.out)
    ensure_dir(out_root)
    fig_root = Path(args.figdir)
    ensure_dir(fig_root)

    for csv_path, tag in zip(csvs, tags):
        rows = load_rows(csv_path)

        # Plot warm vs cold per (license, graph)
        if tag == "dynamic20":
            include = None  # wszystkie oprócz spotify w tekście, ale wykresy mogą być kompletne
        else:
            # pełny run -- skupiamy się na duolingo_super i roman_domination
            include = {"duolingo_super", "roman_domination"}
        run_plots_dynamic(rows, title_prefix=tag, out_root=fig_root / tag, include_licenses=include)

        # Mutation intensity
        inten = mutation_intensity(rows)
        plot_mutation_intensity(inten, title=f"{tag}: intensywność mutacji w czasie", out_path=fig_root / f"{tag}_mutation_intensity")

        # Cost delta vs mutation intensity
        plot_cost_delta_vs_mutation(rows, title=f"{tag}: Δ kosztu/node vs intensywność mutacji", out_path=fig_root / f"{tag}_cost_delta_vs_mutation")

        # Warm advantage summaries
        write_warm_advantage(rows, dataset=tag, out_csv=out_root / f"{tag}_warm_advantage.csv")


if __name__ == "__main__":
    main()
