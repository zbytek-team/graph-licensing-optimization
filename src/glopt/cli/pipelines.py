from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RunInfo:
    run_id: str
    csv: Path
    run_dir: Path


def _now_id(suffix: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"


def _run_benchmark_all() -> None:
    bench = importlib.import_module("glopt.cli.benchmark")
    run_id = _now_id("benchmark")
    bench.RUN_ID = run_id
    bench.SIZES = bench.SIZES_SMALL + [300, 600, 1000]  # skrócone, lecz przekrojowe
    bench.TIMEOUT_SECONDS = 60.0
    bench.main()


def _run_benchmark() -> None:
    bench = importlib.import_module("glopt.cli.benchmark")
    run_id = _now_id("benchmark")
    bench.RUN_ID = run_id
    bench.SIZES = bench.SIZES
    bench.SAMPLES_PER_SIZE = 3
    bench.REPEATS_PER_GRAPH = 2
    bench.TIMEOUT_SECONDS = 60.0
    bench.main()



def _run_benchmark_real() -> None:
    m = importlib.import_module("glopt.cli.benchmark_real")
    run_id = _now_id("benchmark_real")
    m.RUN_ID = run_id
    m.REPEATS_PER_GRAPH = 2
    m.TIMEOUT_SECONDS = 60.0
    m.main()


def _run_trees() -> None:
    t = importlib.import_module("glopt.cli.trees")
    run_id = _now_id("trees")
    t.RUN_ID = run_id
    t.SIZES = [20, 50, 100, 200, 400, 800]
    t.SAMPLES_PER_SIZE = 2
    t.REPEATS_PER_GRAPH = 1
    t.TIMEOUT_SECONDS = 45.0
    t.main()



def _run_dynamic() -> None:
    d = importlib.import_module("glopt.cli.dynamic")
    run_id = _now_id("dynamic")
    d.RUN_ID = run_id
    d.SIZES = [20, 40, 80, 160, 320, 640]
    d.NUM_STEPS = 30
    d.REPEATS_PER_GRAPH = 1
    d.TIMEOUT_SECONDS = 45.0
    d.LICENSE_CONFIG_NAMES = ["duolingo_super", "roman_domination"]
    d.main()



def _run_dynamic_realistic(tag: str, sizes: list[int], steps: int, nodes_mode: str, edges_mode: str) -> None:
    d = importlib.import_module("glopt.cli.dynamic")
    run_id = _now_id(f"dynamic_{tag}")
    d.RUN_ID = run_id
    d.SIZES = sizes
    d.NUM_STEPS = steps
    d.REPEATS_PER_GRAPH = 1
    d.TIMEOUT_SECONDS = 45.0
    d.LICENSE_CONFIG_NAMES = ["duolingo_super", "roman_domination"]
    d.MODE_NODES = nodes_mode
    d.MODE_EDGES = edges_mode
    d.main()



def _run_extensions_static() -> None:
    bench = importlib.import_module("glopt.cli.benchmark")
    run_id = _now_id("extensions")
    bench.RUN_ID = run_id
    bench.SIZES = [20, 50, 100, 200]
    bench.SAMPLES_PER_SIZE = 2
    bench.REPEATS_PER_GRAPH = 1
    bench.TIMEOUT_SECONDS = 45.0
    bench.LICENSE_CONFIG_NAMES = [
        "spotify",
        "netflix",
        *[f"roman_p_{p}" for p in ("1_5", "2_5", "3_0")],
        *[f"duolingo_p_{p}" for p in ("1_5", "2_0", "2_5")],
    ]
    bench.main()



def _run_extensions_dynamic() -> None:
    d = importlib.import_module("glopt.cli.dynamic")
    run_id = _now_id("extensions_dyn")
    d.RUN_ID = run_id
    d.SIZES = [40, 80, 160]
    d.NUM_STEPS = 15
    d.REPEATS_PER_GRAPH = 1
    d.TIMEOUT_SECONDS = 45.0
    d.LICENSE_CONFIG_NAMES = [
        "spotify",
        "netflix",
        *[f"roman_p_{p}" for p in ("1_5", "2_5", "3_0")],
        *[f"duolingo_p_{p}" for p in ("1_5", "2_0", "2_5")],
    ]
    d.main()


def _run_meta_sweep() -> Path:
    out = Path("results") / "meta_params_sweep.csv"
    import subprocess

    subprocess.run([sys.executable, "-m", "scripts.analysis.meta_params_sweep", "--out", str(out)], check=True)
    return out


def static_all() -> None:
    _run_benchmark_all()


def real_ego() -> None:
    _run_benchmark_real()


def trees() -> None:
    _run_trees()


def extensions_static() -> None:
    _run_extensions_static()


def extensions_dynamic() -> None:
    _run_extensions_dynamic()


def export_thesis_figs() -> None:
    import subprocess

    subprocess.run([sys.executable, "-m", "scripts.analysis.export_thesis_figs"], check=True)




def meta_sweep() -> None:
    _run_meta_sweep()


def _run_dynamic_synthetic(label: str, sizes_with_20: list[int], steps: int, add_nodes: float, rem_nodes: float, add_edges: float, rem_edges: float) -> None:
    d = importlib.import_module("glopt.cli.dynamic")
    run_id = _now_id(f"dynamic_{label}")
    d.RUN_ID = run_id
    d.SIZES = sizes_with_20
    d.NUM_STEPS = steps
    d.REPEATS_PER_GRAPH = 1
    d.TIMEOUT_SECONDS = 45.0
    d.LICENSE_CONFIG_NAMES = ["duolingo_super", "roman_domination"]
    d.ADD_NODES_PROB = add_nodes
    d.REMOVE_NODES_PROB = rem_nodes
    d.ADD_EDGES_PROB = add_edges
    d.REMOVE_EDGES_PROB = rem_edges
    d.MODE_NODES = "random"
    d.MODE_EDGES = "random"
    d.main()

def dynamic() -> None:
    # Three intensity variants on synthetic graphs: low / medium / high; each run includes size 20
    sizes = [20, 40, 80, 160, 320, 640]
    # _run_dynamic_synthetic("low", sizes, 30, add_nodes=0.02, rem_nodes=0.01, add_edges=0.06, rem_edges=0.04)
    _run_dynamic_synthetic("med", sizes, 30, add_nodes=0.06, rem_nodes=0.04, add_edges=0.18, rem_edges=0.12)
    _run_dynamic_synthetic("high", sizes, 30, add_nodes=0.12, rem_nodes=0.08, add_edges=0.30, rem_edges=0.20)


def dynamic_real() -> None:
    # Three realistic mutation variants on synthetic graphs
    _run_dynamic_realistic("pref_triadic", [40, 80, 160, 320], 30, nodes_mode="preferential", edges_mode="triadic")
    _run_dynamic_realistic("pref_pref", [40, 80, 160, 320, 640], 30, nodes_mode="preferential", edges_mode="preferential")
    _run_dynamic_realistic("rand_rewire", [40, 80, 160, 320, 640], 30, nodes_mode="random", edges_mode="rewire_ws")


def quick() -> None:
    # static_all()
    # trees()
    # real_ego()
    dynamic()
    dynamic_real()
    extensions_static()
    extensions_dynamic()
    return None


def full() -> None:
    _run_benchmark()
    _run_trees()
    _run_benchmark_real()
    dynamic()
    dynamic_real()
    extensions_static()
    extensions_dynamic()
    return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m glopt.cli.pipelines <command>")
        print("commands: static_all | trees | real_ego | dynamic | dynamic_real | extensions_static | extensions_dynamic | meta_sweep | all | quick | full")
        sys.exit(2)
    cmd = sys.argv[1]
    globals().get(cmd, lambda: (_ for _ in ()).throw(SystemExit(f"unknown command: {cmd}")))()
