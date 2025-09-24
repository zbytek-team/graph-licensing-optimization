from __future__ import annotations

import importlib
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable, Mapping, Sequence

__all__ = [
    "PIPELINES",
    "PIPELINE_ALIASES",
    "Pipeline",
    "Step",
    "StepResult",
    "main",
]

_MISSING = object()
OverrideFactory = Callable[[ModuleType], object]
ModuleOverride = object | OverrideFactory


def _now_id(suffix: str | None) -> str:
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{suffix}" if suffix else base


@contextmanager
def _patched_module(module: ModuleType, updates: Mapping[str, object]) -> Iterable[ModuleType]:
    originals: dict[str, object] = {}
    try:
        for attr, value in updates.items():
            originals[attr] = getattr(module, attr, _MISSING)
            setattr(module, attr, value)
        yield module
    finally:
        for attr, previous in originals.items():
            if previous is _MISSING:
                try:
                    delattr(module, attr)
                except AttributeError:
                    pass
            else:
                setattr(module, attr, previous)


@dataclass(frozen=True)
class StepResult:
    key: str
    run_id: str | None = None
    notes: Mapping[str, object] | None = None


Runner = Callable[[], StepResult]


@dataclass(frozen=True)
class Step:
    key: str
    description: str
    runner: Runner

    def run(self) -> StepResult:
        print(f"[pipeline] → {self.key}: {self.description}")
        try:
            result = self.runner()
        except Exception as exc:  # pragma: no cover - surface the failure context
            print(f"[pipeline] ✗ {self.key}: failed ({exc!r})")
            raise
        if result.run_id:
            print(f"[pipeline] ✓ {self.key}: completed (run_id={result.run_id})")
        else:
            print(f"[pipeline] ✓ {self.key}: completed")
        return result


@dataclass(frozen=True)
class Pipeline:
    name: str
    description: str
    steps: Sequence[Step]

    def run(self) -> list[StepResult]:
        print(f"[pipeline] === {self.name}: {self.description}")
        results: list[StepResult] = []
        for step in self.steps:
            results.append(step.run())
        return results


def _compute_overrides(module: ModuleType, overrides: Mapping[str, ModuleOverride]) -> dict[str, object]:
    resolved: dict[str, object] = {}
    for attr, value in overrides.items():
        resolved[attr] = value(module) if callable(value) else value
    return resolved


def _make_module_step(
    key: str,
    module_name: str,
    description: str,
    *,
    run_id_suffix: str | None,
    overrides: Mapping[str, ModuleOverride] | None = None,
    main_attr: str = "main",
) -> Step:
    overrides = overrides or {}

    def _runner() -> StepResult:
        module = importlib.import_module(module_name)
        computed = _compute_overrides(module, overrides)
        run_id = None
        if run_id_suffix and hasattr(module, "RUN_ID"):
            run_id = _now_id(run_id_suffix)
            computed.setdefault("RUN_ID", run_id)
        with _patched_module(module, computed):
            getattr(module, main_attr)()
        return StepResult(key=key, run_id=run_id)

    return Step(key=key, description=description, runner=_runner)


def _resolve_analysis_module(primary: str, fallback: str | None = None) -> str:
    try:
        importlib.import_module(primary)
        return primary
    except ModuleNotFoundError:
        if fallback is None:
            raise
        importlib.import_module(fallback)
        return fallback


def _make_subprocess_step(
    key: str,
    description: str,
    argv_factory: Callable[[], Sequence[str]],
    *,
    cwd: Path | None = None,
) -> Step:
    def _runner() -> StepResult:
        argv = list(argv_factory())
        subprocess.run(argv, check=True, cwd=cwd)
        return StepResult(key=key)

    return Step(key=key, description=description, runner=_runner)


DEFAULT_LICENSES = ["duolingo_super", "roman_domination"]


def _extension_license_names() -> list[str]:
    names = ["spotify", "netflix"]
    names.extend([f"roman_p_{p}" for p in ("3", "4", "5")])
    names.extend([f"duolingo_p_{p}" for p in ("2", "4", "5")])
    return names


# --- Smoke steps (fast) ----------------------------------------------------

benchmark_smoke = _make_module_step(
    key="benchmark-smoke",
    module_name="glopt.cli.benchmark",
    description="Synthetic benchmark on n≤150 (1 sample per size).",
    run_id_suffix="benchmark_smoke",
    overrides={
        "SIZES": [20, 50, 100, 150],
        "SAMPLES_PER_SIZE": 1,
        "REPEATS_PER_GRAPH": 1,
        "TIMEOUT_SECONDS": 25.0,
        "LICENSE_CONFIG_NAMES": list(DEFAULT_LICENSES),
    },
)

trees_smoke = _make_module_step(
    key="trees-smoke",
    module_name="glopt.cli.trees",
    description="Tree benchmark (n up to 100, single sample).",
    run_id_suffix="trees_smoke",
    overrides={
        "SIZES": [20, 50, 100],
        "SAMPLES_PER_SIZE": 1,
        "REPEATS_PER_GRAPH": 1,
        "TIMEOUT_SECONDS": 20.0,
    },
)

dynamic_smoke = _make_module_step(
    key="dynamic-smoke",
    module_name="glopt.cli.dynamic",
    description="Dynamic synthetic graphs with mild churn (fast).",
    run_id_suffix="dynamic_smoke",
    overrides={
        "SIZES": [20, 40, 80],
        "NUM_STEPS": 12,
        "REPEATS_PER_GRAPH": 1,
        "TIMEOUT_SECONDS": 30.0,
        "LICENSE_CONFIG_NAMES": list(DEFAULT_LICENSES),
        "ADD_NODES_PROB": 0.05,
        "REMOVE_NODES_PROB": 0.03,
        "ADD_EDGES_PROB": 0.15,
        "REMOVE_EDGES_PROB": 0.10,
        "MODE_NODES": "random",
        "MODE_EDGES": "random",
    },
)

dynamic_real_smoke = _make_module_step(
    key="dynamic-real-smoke",
    module_name="glopt.cli.dynamic",
    description="Dynamic graphs with preferential attachment (compact).",
    run_id_suffix="dynamic_real_smoke",
    overrides={
        "SIZES": [40, 80],
        "NUM_STEPS": 12,
        "REPEATS_PER_GRAPH": 1,
        "TIMEOUT_SECONDS": 30.0,
        "LICENSE_CONFIG_NAMES": list(DEFAULT_LICENSES),
        "MODE_NODES": "preferential",
        "MODE_EDGES": "triadic",
        "ADD_NODES_PROB": 0.06,
        "REMOVE_NODES_PROB": 0.04,
        "ADD_EDGES_PROB": 0.18,
        "REMOVE_EDGES_PROB": 0.12,
    },
)


# --- Full steps ------------------------------------------------------------

benchmark_full = _make_module_step(
    key="benchmark-full",
    module_name="glopt.cli.benchmark",
    description="Synthetic benchmark sweep with extended sizes.",
    run_id_suffix="benchmark",
    overrides={
        "SIZES": lambda module: list(module.SIZES_SMALL) + [300, 600, 1000],
        "TIMEOUT_SECONDS": 60.0,
    },
)

benchmark_real = _make_module_step(
    key="benchmark-real",
    module_name="glopt.cli.benchmark_real",
    description="Benchmark on facebook ego-network instances.",
    run_id_suffix="benchmark_real",
    overrides={
        "REPEATS_PER_GRAPH": 2,
        "TIMEOUT_SECONDS": 60.0,
    },
)

trees_full = _make_module_step(
    key="trees-full",
    module_name="glopt.cli.trees",
    description="Tree benchmarks across medium sizes.",
    run_id_suffix="trees",
    overrides={
        "SIZES": [20, 50, 100, 200, 400, 800],
        "SAMPLES_PER_SIZE": 2,
        "REPEATS_PER_GRAPH": 1,
        "TIMEOUT_SECONDS": 45.0,
    },
)

dynamic_low = _make_module_step(
    key="dynamic-low",
    module_name="glopt.cli.dynamic",
    description="Dynamic synthetic graphs – low mutation intensity.",
    run_id_suffix="dynamic_low",
    overrides={
        "SIZES": [20, 40, 80, 160, 320, 640],
        "NUM_STEPS": 30,
        "REPEATS_PER_GRAPH": 1,
        "TIMEOUT_SECONDS": 45.0,
        "LICENSE_CONFIG_NAMES": list(DEFAULT_LICENSES),
        "ADD_NODES_PROB": 0.02,
        "REMOVE_NODES_PROB": 0.01,
        "ADD_EDGES_PROB": 0.06,
        "REMOVE_EDGES_PROB": 0.04,
        "MODE_NODES": "random",
        "MODE_EDGES": "random",
    },
)

dynamic_med = _make_module_step(
    key="dynamic-med",
    module_name="glopt.cli.dynamic",
    description="Dynamic synthetic graphs – medium mutation intensity.",
    run_id_suffix="dynamic_med",
    overrides={
        "SIZES": [20, 40, 80, 160, 320, 640],
        "NUM_STEPS": 30,
        "REPEATS_PER_GRAPH": 1,
        "TIMEOUT_SECONDS": 45.0,
        "LICENSE_CONFIG_NAMES": list(DEFAULT_LICENSES),
        "ADD_NODES_PROB": 0.06,
        "REMOVE_NODES_PROB": 0.04,
        "ADD_EDGES_PROB": 0.18,
        "REMOVE_EDGES_PROB": 0.12,
        "MODE_NODES": "random",
        "MODE_EDGES": "random",
    },
)

dynamic_high = _make_module_step(
    key="dynamic-high",
    module_name="glopt.cli.dynamic",
    description="Dynamic synthetic graphs – high mutation intensity.",
    run_id_suffix="dynamic_high",
    overrides={
        "SIZES": [20, 40, 80, 160, 320, 640],
        "NUM_STEPS": 30,
        "REPEATS_PER_GRAPH": 1,
        "TIMEOUT_SECONDS": 45.0,
        "LICENSE_CONFIG_NAMES": list(DEFAULT_LICENSES),
        "ADD_NODES_PROB": 0.12,
        "REMOVE_NODES_PROB": 0.08,
        "ADD_EDGES_PROB": 0.30,
        "REMOVE_EDGES_PROB": 0.20,
        "MODE_NODES": "random",
        "MODE_EDGES": "random",
    },
)

dynamic_real_pref_triadic = _make_module_step(
    key="dynamic-real-pref-triadic",
    module_name="glopt.cli.dynamic",
    description="Dynamic graphs – preferential nodes, triadic edges.",
    run_id_suffix="dynamic_pref_triadic",
    overrides={
        "SIZES": [40, 80, 160, 320],
        "NUM_STEPS": 30,
        "REPEATS_PER_GRAPH": 1,
        "TIMEOUT_SECONDS": 45.0,
        "LICENSE_CONFIG_NAMES": list(DEFAULT_LICENSES),
        "MODE_NODES": "preferential",
        "MODE_EDGES": "triadic",
    },
)

dynamic_real_pref_pref = _make_module_step(
    key="dynamic-real-pref-pref",
    module_name="glopt.cli.dynamic",
    description="Dynamic graphs – preferential nodes and edges.",
    run_id_suffix="dynamic_pref_pref",
    overrides={
        "SIZES": [40, 80, 160, 320, 640],
        "NUM_STEPS": 30,
        "REPEATS_PER_GRAPH": 1,
        "TIMEOUT_SECONDS": 45.0,
        "LICENSE_CONFIG_NAMES": list(DEFAULT_LICENSES),
        "MODE_NODES": "preferential",
        "MODE_EDGES": "preferential",
    },
)

dynamic_real_rand_rewire = _make_module_step(
    key="dynamic-real-rand-rewire",
    module_name="glopt.cli.dynamic",
    description="Dynamic graphs – random nodes, Watts–Strogatz rewiring.",
    run_id_suffix="dynamic_rand_rewire",
    overrides={
        "SIZES": [40, 80, 160, 320, 640],
        "NUM_STEPS": 30,
        "REPEATS_PER_GRAPH": 1,
        "TIMEOUT_SECONDS": 45.0,
        "LICENSE_CONFIG_NAMES": list(DEFAULT_LICENSES),
        "MODE_NODES": "random",
        "MODE_EDGES": "rewire_ws",
    },
)

extensions_static_full = _make_module_step(
    key="extensions-static",
    module_name="glopt.cli.benchmark",
    description="Static benchmark for license extensions portfolio.",
    run_id_suffix="extensions",
    overrides={
        "SIZES": [20, 50, 100, 200],
        "SAMPLES_PER_SIZE": 2,
        "REPEATS_PER_GRAPH": 1,
        "TIMEOUT_SECONDS": 45.0,
        "LICENSE_CONFIG_NAMES": _extension_license_names,
    },
)

extensions_dynamic_full = _make_module_step(
    key="extensions-dynamic",
    module_name="glopt.cli.dynamic",
    description="Dynamic benchmark for license extensions portfolio.",
    run_id_suffix="extensions_dyn",
    overrides={
        "SIZES": [40, 80, 160],
        "NUM_STEPS": 15,
        "REPEATS_PER_GRAPH": 1,
        "TIMEOUT_SECONDS": 45.0,
        "LICENSE_CONFIG_NAMES": _extension_license_names,
    },
)


# --- Analysis helpers ------------------------------------------------------

_meta_output = Path("results") / "meta_params_sweep.csv"

meta_sweep_step = _make_subprocess_step(
    key="meta-sweep",
    description="Run meta-parameter sweep analysis pipeline.",
    argv_factory=lambda: [
        sys.executable,
        "-m",
        _resolve_analysis_module("glopt.analysis.meta_params_sweep", "scripts.analysis.meta_params_sweep"),
        "--out",
        str(_meta_output),
    ],
)

export_thesis_figs_step = _make_subprocess_step(
    key="export-thesis-figs",
    description="Export thesis figures via the analysis toolkit.",
    argv_factory=lambda: [
        sys.executable,
        "-m",
        _resolve_analysis_module("glopt.analysis.export_thesis_figs", "scripts.analysis.export_thesis_figs"),
    ],
)


# --- Pipeline registry -----------------------------------------------------

PIPELINES: dict[str, Pipeline] = {
    "quick": Pipeline(
        name="quick",
        description="Fast coverage of the main scenarios (smoke-level).",
        steps=[benchmark_smoke, trees_smoke, dynamic_smoke, dynamic_real_smoke],
    ),
    "full": Pipeline(
        name="full",
        description="Complete suite covering static, dynamic, and extension runs.",
        steps=[
            benchmark_full,
            trees_full,
            benchmark_real,
            dynamic_low,
            dynamic_med,
            dynamic_high,
            dynamic_real_pref_triadic,
            dynamic_real_pref_pref,
            dynamic_real_rand_rewire,
            extensions_static_full,
            extensions_dynamic_full,
        ],
    ),
    "static_all": Pipeline(
        name="static_all",
        description="Synthetic benchmark sweep across sizes.",
        steps=[benchmark_full],
    ),
    "real_ego": Pipeline(
        name="real_ego",
        description="Benchmark on real-world ego networks.",
        steps=[benchmark_real],
    ),
    "trees": Pipeline(
        name="trees",
        description="Tree graph experiments (medium sizes).",
        steps=[trees_full],
    ),
    "dynamic": Pipeline(
        name="dynamic",
        description="Dynamic synthetic benchmarks at three intensities.",
        steps=[dynamic_low, dynamic_med, dynamic_high],
    ),
    "dynamic_real": Pipeline(
        name="dynamic_real",
        description="Dynamic runs on realistic mutation models.",
        steps=[dynamic_real_pref_triadic, dynamic_real_pref_pref, dynamic_real_rand_rewire],
    ),
    "extensions_static": Pipeline(
        name="extensions_static",
        description="Static experiments for extended license configurations.",
        steps=[extensions_static_full],
    ),
    "extensions_dynamic": Pipeline(
        name="extensions_dynamic",
        description="Dynamic experiments for extended license configurations.",
        steps=[extensions_dynamic_full],
    ),
    "meta_sweep": Pipeline(
        name="meta_sweep",
        description="Sweep solver meta-parameters and aggregate results.",
        steps=[meta_sweep_step],
    ),
    "export_thesis_figs": Pipeline(
        name="export_thesis_figs",
        description="Generate thesis-ready figures using the analysis toolkit.",
        steps=[export_thesis_figs_step],
    ),
}

PIPELINE_ALIASES: dict[str, str] = {
    "smoke": "quick",
    "all": "full",
    "extensions": "extensions_static",
}


def available_pipelines() -> list[tuple[str, str]]:
    items = sorted(PIPELINES.items(), key=lambda item: item[0])
    return [(name, pipeline.description) for name, pipeline in items]


def resolve_pipeline(name: str) -> Pipeline | None:
    actual = PIPELINE_ALIASES.get(name, name)
    return PIPELINES.get(actual)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args or args[0] in {"-h", "--help", "help"}:
        print("usage: python -m glopt.cli.pipelines <pipeline|list>")
        print("available pipelines:")
        for name, desc in available_pipelines():
            print(f"  {name:18s} {desc}")
        return 0

    cmd = args[0]
    if cmd in {"list", "ls"}:
        for name, desc in available_pipelines():
            print(f"{name:18s} {desc}")
        return 0

    pipeline = resolve_pipeline(cmd)
    if pipeline is None:
        print(f"unknown pipeline: {cmd}")
        print("use 'python -m glopt.cli.pipelines list' to see available options")
        return 2

    pipeline.run()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
