from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from typing import Any

from glopt.core import LicenseType


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


def print_banner(title: str, params: Mapping[str, Any]) -> None:
    print(title)
    for k in params:
        print(f"{k}: {params[k]}")


def print_footer(summary: Mapping[str, Any]) -> None:
    print("Summary")
    for k in summary:
        print(f"{k}: {summary[k]}")


def print_stage(label: str) -> None:
    print(label)


def print_step(prefix: str, **kv: Any) -> None:
    parts = [prefix]
    for k, v in kv.items():
        parts.append(f"{k}={v}")
    print(" ".join(parts))


def normalize_license_costs(license_types: list[LicenseType]) -> list[LicenseType]:
    if not license_types:
        return []
    singles = [lt for lt in license_types if lt.min_capacity == 1 and lt.max_capacity == 1 and lt.cost > 0]
    if singles:
        baseline = singles[0].cost
    else:
        positive = [lt.cost for lt in license_types if lt.cost > 0]
        baseline = min(positive) if positive else 1.0
    if baseline == 0:
        return license_types
    return [replace(lt, cost=lt.cost / baseline) for lt in license_types]
