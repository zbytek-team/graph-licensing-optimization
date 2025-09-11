from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from time import perf_counter
from typing import Any


def build_run_id(suffix: str, run_id: str | None = None) -> str:
    base = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base}_{suffix}" if suffix else base


def fmt_ms(ms: float | int) -> str:
    try:
        v = float(ms)
    except Exception:
        return str(ms)
    if v >= 1000.0:
        return f"{v / 1000.0:.3f}s"
    return f"{v:.1f}ms"


def fmt_s(seconds: float | int) -> str:
    try:
        v = float(seconds)
    except Exception:
        return str(seconds)
    return f"{v:.3f}s"


def start_timer() -> float:
    return perf_counter()


def elapsed_s(t0: float) -> float:
    return perf_counter() - t0


def print_banner(title: str, params: Mapping[str, Any]) -> None:
    print(f"== {title} ==")
    for k in params:
        print(f"{k}: {params[k]}")


def print_footer(summary: Mapping[str, Any]) -> None:
    print("== summary ==")
    for k in summary:
        print(f"{k}: {summary[k]}")


def print_stage(label: str) -> None:
    print(f"== {label} ==")


def print_step(prefix: str, **kv: Any) -> None:
    parts = [prefix]
    for k, v in kv.items():
        parts.append(f"{k}={v}")
    print(" ".join(parts))
