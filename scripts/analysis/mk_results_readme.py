from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def load_csv_rows(p: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    return rows


def summarize_aggregates(agg_csv: Path) -> tuple[str, list[str]]:
    rows = load_csv_rows(agg_csv)
    lines: list[str] = []
    title = agg_csv.parent.parent.name  # e.g., benchmark_all
    if not rows:
        return title, ["- Brak danych agregatów (aggregates.csv nie znaleziony)."]

    # Per algorithm summary across graphs and n: mean of means
    per_alg_cost: dict[str, list[float]] = defaultdict(list)
    per_alg_time: dict[str, list[float]] = defaultdict(list)
    graphs: set[str] = set()
    for r in rows:
        try:
            alg = r["algorithm"].strip()
            g = r.get("graph", "").strip()
            graphs.add(g)
            per_alg_cost[alg].append(float(r.get("cost_mean", "nan")))
            per_alg_time[alg].append(float(r.get("time_ms_mean", "nan")))
        except Exception:
            pass

    def _best(d: dict[str, list[float]], reverse: bool = False) -> tuple[str, float]:
        best_name = ""
        best_val = float("inf") if not reverse else float("-inf")
        for k, vs in d.items():
            vals = [v for v in vs if v == v]  # drop NaNs
            if not vals:
                continue
            m = mean(vals)
            if (m < best_val and not reverse) or (m > best_val and reverse):
                best_val, best_name = m, k
        return best_name, best_val

    best_cost_alg, best_cost = _best(per_alg_cost)
    best_time_alg, best_time = _best(per_alg_time)

    lines.append(f"- Liczba grafów: {len([g for g in graphs if g])}")
    if best_cost_alg:
        lines.append(f"- Najniższy średni koszt: {best_cost_alg} (≈ {best_cost:.3f})")
    if best_time_alg:
        lines.append(f"- Najlepszy średni czas: {best_time_alg} (≈ {best_time:.3f} ms)")

    # Quick tie-in guidance
    if best_cost_alg and best_time_alg and best_cost_alg != best_time_alg:
        lines.append("- Występuje trade‑off koszt vs czas — różni zwycięzcy.")

    return title, lines


def _load_avg_ranks(path: Path) -> list[tuple[str, float]]:
    if not path.exists():
        return []
    rows = load_csv_rows(path)
    out: list[tuple[str, float]] = []
    for r in rows:
        try:
            out.append((r["algorithm"], float(r["avg_rank"])))
        except Exception:
            pass
    out.sort(key=lambda x: x[1])
    return out


def _load_time_scaling(path: Path) -> dict[str, float]:
    med: dict[str, list[float]] = defaultdict(list)
    if not path.exists():
        return {}
    for r in load_csv_rows(path):
        try:
            alg = r["algorithm"].strip()
            b = float(r["slope_b"])  # time ≈ a*n^b
        except Exception:
            continue
        if alg:
            med[alg].append(b)
    # median (manual)
    res: dict[str, float] = {}
    for alg, vals in med.items():
        if not vals:
            continue
        vals = sorted(vals)
        m = len(vals)
        res[alg] = vals[m // 2] if m % 2 == 1 else 0.5 * (vals[m // 2 - 1] + vals[m // 2])
    return res


def _load_sig_counts(path: Path, alpha: float = 0.05) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if not path.exists():
        return {}
    for r in load_csv_rows(path):
        try:
            a, b = r["alg_a"], r["alg_b"]
            p = float(r["p_approx"])
        except Exception:
            continue
        if p < alpha:
            counts[a] += 1
            counts[b] += 1
    return dict(counts)


def _interpret_block(set_name: str, set_dir: Path) -> list[str]:
    lines: list[str] = []
    all_dir = set_dir / "all"
    # Friedman ranks
    ranks_cost = _load_avg_ranks(all_dir / "stats_friedman_cost.csv")
    ranks_time = _load_avg_ranks(all_dir / "stats_friedman_time.csv")
    # Time scaling medians
    slopes = _load_time_scaling(all_dir / "time_scaling.csv")
    # Pairwise significance counts (how many significant differences an algorithm participates in)
    sig_cost = _load_sig_counts(all_dir / "stats_wilcoxon_cost.csv")
    sig_time = _load_sig_counts(all_dir / "stats_wilcoxon_time.csv")

    if ranks_cost:
        best_cost_alg, best_cost_rank = ranks_cost[0]
        lines.append(f"- Ranking kosztu (Friedman): 1) {best_cost_alg} (avg rank≈{best_cost_rank:.2f}), dalej: " + ", ".join(f"{a}({r:.2f})" for a, r in ranks_cost[1:3]))
    if ranks_time:
        best_time_alg, best_time_rank = ranks_time[0]
        lines.append(f"- Ranking czasu (Friedman): 1) {best_time_alg} (avg rank≈{best_time_rank:.2f}), dalej: " + ", ".join(f"{a}({r:.2f})" for a, r in ranks_time[1:3]))
    if slopes:
        # pokaż 2-3 najniższe b
        top_b = sorted(slopes.items(), key=lambda x: x[1])[:3]
        lines.append("- Empiryczna złożoność czasu: najniższe b (time≈a·n^b): " + ", ".join(f"{a} (b≈{b:.2f})" for a, b in top_b))
    if sig_cost or sig_time:
        # podsumowanie ile parowych różnic istotnych ma dany alg
        top_cost = sorted(sig_cost.items(), key=lambda x: -x[1])[:2]
        top_time = sorted(sig_time.items(), key=lambda x: -x[1])[:2]
        if top_cost:
            lines.append("- Istotne różnice kosztu (Wilcoxon p<0.05): " + ", ".join(f"{a} ({c})" for a, c in top_cost))
        if top_time:
            lines.append("- Istotne różnice czasu (Wilcoxon p<0.05): " + ", ".join(f"{a} ({c})" for a, c in top_time))
    # Leaderboard
    lb_cost = set_dir / "all" / "leaderboard_cost.csv"
    lb_time = set_dir / "all" / "leaderboard_time.csv"

    def _top_from_lb(p: Path) -> list[str]:
        rows = load_csv_rows(p)
        cnt: Counter[str] = Counter()
        for r in rows:
            try:
                alg = r["algorithm"].strip()
                wins = int(r["wins"])
            except Exception:
                continue
            cnt[alg] += wins
        top = [f"{a} ({c})" for a, c in cnt.most_common(2)]
        return top

    if lb_cost.exists():
        top = _top_from_lb(lb_cost)
        if top:
            lines.append("- Leaderboard kosztu (wygrane per instancja zgrupowane po binach n): " + ", ".join(top))
    if lb_time.exists():
        top = _top_from_lb(lb_time)
        if top:
            lines.append("- Leaderboard czasu (wygrane per instancja zgrupowane po binach n): " + ", ".join(top))

    if not lines:
        lines.append("- (Brak danych statystycznych do interpretacji)")
    return lines


def write_results_readme(root: Path) -> Path:
    out = root / "README.md"
    lines: list[str] = []
    lines.append("# Wyniki — Podsumowanie\n")
    lines.append("Ten plik zbiera najważniejsze wnioski z analiz w katalogach `results/benchmark_all` oraz `results/benchmark_real_all`.\n")

    # Sections
    sections = {
        "benchmark_all": root / "benchmark_all" / "all" / "aggregates.csv",
        "benchmark_real_all": root / "benchmark_real_all" / "all" / "aggregates.csv",
    }

    for name, path in sections.items():
        title, bullets = summarize_aggregates(path)
        lines.append(f"## {title}\n")
        if bullets:
            lines.extend(bullets)
        else:
            lines.append("- Brak danych do podsumowania.")
        # Interpretacja dodatkowa (Friedman/Wilcoxon/time_scaling)
        interp = _interpret_block(name, root / name)
        if interp:
            lines.append("\n### Interpretacja\n")
            lines.extend(interp)
        lines.append("")

    # Directions to figures
    lines.append("## Figury i tabele\n")
    lines.append("- Pełne wykresy i tabele: patrz odpowiednio `results/benchmark_all/**` i `results/benchmark_real_all/**`.\n")

    root.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    root = Path("results")
    p = write_results_readme(root)
    print(p)


if __name__ == "__main__":
    main()
