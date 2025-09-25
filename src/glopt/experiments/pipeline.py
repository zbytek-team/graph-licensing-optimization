from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from glopt.experiments import dynamic, static_real, static_synthetic
from glopt.experiments.common import build_run_id

DEFAULT_LICENSES = ["duolingo_super", "roman_domination"]


def _extension_license_names() -> list[str]:
    names = ["spotify", "netflix"]
    names.extend([f"roman_p_{p}" for p in ("3", "4", "5")])
    names.extend([f"duolingo_p_{p}" for p in ("2", "4", "5")])
    return names

def _static_synth_sizes(module: Any) -> list[int]:
    return list(module.SIZES_SMALL) + [300, 600, 1000]


@dataclass
class Step:
    label: str
    module: Any
    run_id_suffix: str | None
    overrides: dict[str, Any]


def _resolve_override(module: Any, value: Any) -> Any:
    if callable(value):
        return value(module)
    return value


@contextmanager
def _module_overrides(module: Any, overrides: dict[str, Any]):
    sentinel = object()
    original: dict[str, Any] = {}
    try:
        for key, val in overrides.items():
            original[key] = getattr(module, key, sentinel)
            setattr(module, key, val)
        yield
    finally:
        for key, val in original.items():
            if val is sentinel:
                delattr(module, key)
            else:
                setattr(module, key, val)


def run_step(step: Step) -> str | None:
    computed = {
        key: _resolve_override(step.module, val)
        for key, val in step.overrides.items()
    }
    run_id = None
    if step.run_id_suffix and hasattr(step.module, "RUN_ID"):
        run_id = build_run_id(step.run_id_suffix)
        computed.setdefault("RUN_ID", run_id)
    print(f"==> {step.label}")
    with _module_overrides(step.module, computed):
        step.module.main()
    return run_id


STATIC_SYNTHETIC_STEP = Step(
    label="static synthetic",
    module=static_synthetic,
    run_id_suffix="static_synth",
    overrides={
        "GRAPH_NAMES": ["random", "small_world", "scale_free"],
        "SIZES": _static_synth_sizes,
        "SAMPLES_PER_SIZE": 2,
        "REPEATS_PER_GRAPH": 1,
        "TIMEOUT_SECONDS": 60.0,
        "LICENSE_CONFIG_NAMES": list(DEFAULT_LICENSES),
    },
)

STATIC_REAL_STEP = Step(
    label="static real",
    module=static_real,
    run_id_suffix="static_real",
    overrides={
        "LICENSE_CONFIG_NAMES": list(DEFAULT_LICENSES),
        "REPEATS_PER_GRAPH": 2,
        "TIMEOUT_SECONDS": 60.0,
    },
)

DYNAMIC_LOW_STEP = Step(
    label="dynamic low",
    module=dynamic,
    run_id_suffix="dynamic_low",
    overrides={
        "SIZES": [40, 80, 160, 320],
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

DYNAMIC_MED_STEP = Step(
    label="dynamic medium",
    module=dynamic,
    run_id_suffix="dynamic_med",
    overrides={
        "SIZES": [80, 160, 320, 640],
        "NUM_STEPS": 45,
        "REPEATS_PER_GRAPH": 1,
        "TIMEOUT_SECONDS": 60.0,
        "LICENSE_CONFIG_NAMES": list(DEFAULT_LICENSES),
        "ADD_NODES_PROB": 0.08,
        "REMOVE_NODES_PROB": 0.05,
        "ADD_EDGES_PROB": 0.22,
        "REMOVE_EDGES_PROB": 0.14,
        "MODE_NODES": "random",
        "MODE_EDGES": "random",
    },
)

DYNAMIC_HIGH_STEP = Step(
    label="dynamic high",
    module=dynamic,
    run_id_suffix="dynamic_high",
    overrides={
        "SIZES": [160, 320, 640, 960],
        "NUM_STEPS": 60,
        "REPEATS_PER_GRAPH": 1,
        "TIMEOUT_SECONDS": 75.0,
        "LICENSE_CONFIG_NAMES": list(DEFAULT_LICENSES),
        "ADD_NODES_PROB": 0.10,
        "REMOVE_NODES_PROB": 0.06,
        "ADD_EDGES_PROB": 0.26,
        "REMOVE_EDGES_PROB": 0.18,
        "MODE_NODES": "random",
        "MODE_EDGES": "random",
    },
)

DYNAMIC_PREF_TRIADIC_STEP = Step(
    label="dynamic pref-triadic",
    module=dynamic,
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

DYNAMIC_PREF_PREF_STEP = Step(
    label="dynamic pref-pref",
    module=dynamic,
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

DYNAMIC_RAND_REWIRE_STEP = Step(
    label="dynamic rand-rewire",
    module=dynamic,
    run_id_suffix="dynamic_rand_rewire",
    overrides={
        "SIZES": [40, 80, 160, 320],
        "NUM_STEPS": 30,
        "REPEATS_PER_GRAPH": 1,
        "TIMEOUT_SECONDS": 45.0,
        "LICENSE_CONFIG_NAMES": list(DEFAULT_LICENSES),
        "MODE_NODES": "random",
        "MODE_EDGES": "rewire_ws",
    },
)

EXT_STATIC_STEP = Step(
    label="extensions static",
    module=static_synthetic,
    run_id_suffix="extensions_static",
    overrides={
        "GRAPH_NAMES": ["random", "small_world"],
        "SIZES": [20, 50, 100, 200],
        "SAMPLES_PER_SIZE": 2,
        "REPEATS_PER_GRAPH": 1,
        "TIMEOUT_SECONDS": 45.0,
        "LICENSE_CONFIG_NAMES": _extension_license_names,
    },
)

EXT_DYNAMIC_STEP = Step(
    label="extensions dynamic",
    module=dynamic,
    run_id_suffix="extensions_dynamic",
    overrides={
        "SIZES": [40, 80, 160],
        "NUM_STEPS": 30,
        "REPEATS_PER_GRAPH": 1,
        "TIMEOUT_SECONDS": 45.0,
        "LICENSE_CONFIG_NAMES": _extension_license_names,
        "MODE_NODES": "preferential",
        "MODE_EDGES": "triadic",
    },
)


ALL_STEPS: list[Step] = [
    STATIC_SYNTHETIC_STEP,
    STATIC_REAL_STEP,
    DYNAMIC_LOW_STEP,
    DYNAMIC_MED_STEP,
    DYNAMIC_HIGH_STEP,
    DYNAMIC_PREF_TRIADIC_STEP,
    DYNAMIC_PREF_PREF_STEP,
    DYNAMIC_RAND_REWIRE_STEP,
    EXT_STATIC_STEP,
    EXT_DYNAMIC_STEP,
]


def main(selected: Iterable[str] | None = None) -> None:
    names = {step.label: step for step in ALL_STEPS}
    if selected is None:
        steps = ALL_STEPS
    else:
        steps = [names[name] for name in selected]
    for step in steps:
        run_step(step)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:] or None
    if args is None:
        main()
    else:
        labels = [step.label for step in ALL_STEPS]
        try:
            main(args)
        except KeyError as exc:
            print(f"unknown step: {exc.args[0]}")
            print("available steps:")
            for label in labels:
                print(f"  {label}")
            sys.exit(2)
