from __future__ import annotations

import shutil
from pathlib import Path


def _copy_if_exists(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)


def export_figures(
    results_dir: Path = Path("results"),
    out_dir: Path = Path("docs/thesis/assets/figures"),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Benchmark all (synthetic)
    ba = results_dir / "benchmark_all"
    if ba.exists():
        # Global
        _copy_if_exists(ba / "all" / "heatmap_cost.png", out_dir / "ba_heatmap_cost.png")
        _copy_if_exists(ba / "all" / "leaderboard_cost.png", out_dir / "ba_leaderboard_cost.png")
        _copy_if_exists(ba / "all" / "leaderboard_time.png", out_dir / "ba_leaderboard_time.png")
        _copy_if_exists(ba / "all" / "ilp_timeout_boundary.png", out_dir / "ba_ilp_timeout_boundary.png")

        # Per graph for two key configs
        graphs = ["random", "small_world", "scale_free"]
        configs = [("duolingo_super", "duo"), ("roman_domination", "roman")]
        metrics = [
            ("{g}_cost_vs_n.png", "cost_vs_n"),
            ("{g}_time_vs_n.png", "time_vs_n"),
            ("perf_profile_cost.png", "perf_profile_cost"),
            ("perf_profile_time.png", "perf_profile_time"),
            ("{g}_pareto_cost_time.png", "pareto"),
            ("{g}_density_vs_time.png", "density_vs_time"),
        ]
        for g in graphs:
            for cfg, tag in configs:
                src_dir = ba / cfg / g
                for pattern, mtag in metrics:
                    name = pattern.format(g=g)
                    _copy_if_exists(src_dir / name, out_dir / f"ba_{g}_{tag}_{mtag}.png")

        # Duo vs Roman comparisons
        cmp_dir = ba / "compare_duo_roman"
        _copy_if_exists(cmp_dir / "compare_cost_per_node_duolingo_super_vs_roman_domination.png", out_dir / "ba_compare_cost_duo_vs_roman.png")
        _copy_if_exists(cmp_dir / "compare_time_ms_duolingo_super_vs_roman_domination.png", out_dir / "ba_compare_time_duo_vs_roman.png")
        # By graph
        for g in graphs:
            bg = cmp_dir / "by_graph" / g
            _copy_if_exists(bg / f"compare_cost_per_node_duolingo_super_vs_roman_domination_{g}.png", out_dir / f"ba_compare_cost_duo_vs_roman_{g}.png")
            _copy_if_exists(bg / f"compare_time_ms_duolingo_super_vs_roman_domination_{g}.png", out_dir / f"ba_compare_time_duo_vs_roman_{g}.png")

    # Benchmark real (facebook ego)
    br = results_dir / "benchmark_real_all"
    if br.exists():
        _copy_if_exists(br / "all" / "leaderboard_cost.png", out_dir / "br_leaderboard_cost.png")
        _copy_if_exists(br / "all" / "leaderboard_time.png", out_dir / "br_leaderboard_time.png")
        _copy_if_exists(br / "all" / "ilp_timeout_boundary.png", out_dir / "br_ilp_timeout_boundary.png")
        # Two key configs on facebook_ego
        g = "facebook_ego"
        for cfg, tag in [("duolingo_super", "duo"), ("roman_domination", "roman")]:
            src_dir = br / cfg / g
            for pattern, mtag in [
                ("{g}_cost_vs_n.png", "cost_vs_n"),
                ("{g}_time_vs_n.png", "time_vs_n"),
                ("{g}_pareto_cost_time.png", "pareto"),
                ("{g}_density_vs_time.png", "density_vs_time"),
                ("perf_profile_cost.png", "perf_profile_cost"),
                ("perf_profile_time.png", "perf_profile_time"),
            ]:
                name = pattern.format(g=g)
                _copy_if_exists(src_dir / name, out_dir / f"br_{g}_{tag}_{mtag}.png")
        # Duo vs Roman comparisons
        cmp_dir = br / "compare_duo_roman"
        _copy_if_exists(cmp_dir / "compare_cost_per_node_duolingo_super_vs_roman_domination.png", out_dir / "br_compare_cost_duo_vs_roman.png")
        _copy_if_exists(cmp_dir / "compare_time_ms_duolingo_super_vs_roman_domination.png", out_dir / "br_compare_time_duo_vs_roman.png")
        bg = cmp_dir / "by_graph" / g
        _copy_if_exists(bg / f"compare_cost_per_node_duolingo_super_vs_roman_domination_{g}.png", out_dir / f"br_compare_cost_duo_vs_roman_{g}.png")
        _copy_if_exists(bg / f"compare_time_ms_duolingo_super_vs_roman_domination_{g}.png", out_dir / f"br_compare_time_duo_vs_roman_{g}.png")


def main() -> None:
    export_figures()
    print("Exported figures to docs/thesis/assets/figures")


if __name__ == "__main__":
    main()
