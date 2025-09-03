from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .commons import ensure_dir, GENERATE_PDF


def plot_license_mix(rows: list[dict[str, Any]], title: str, out_path: Path) -> None:
    lic_agg: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        try:
            alg = str(r["algorithm"]) ; js = r.get("license_counts_json", "{}")
            counts = json.loads(js) if isinstance(js, str) else (js or {})
            for k, v in counts.items():
                lic_agg[alg][k] += int(v)
        except Exception:  # robust parsing
            continue
    if not lic_agg:
        return
    algs = sorted(lic_agg.keys())
    all_licenses = sorted({lic for c in lic_agg.values() for lic in c.keys()})
    W = len(algs)
    vals = np.zeros((len(all_licenses), W), dtype=float)
    for j, alg in enumerate(algs):
        total = sum(lic_agg[alg].values()) or 1
        for i, lic in enumerate(all_licenses):
            vals[i, j] = lic_agg[alg].get(lic, 0) / total
    plt.figure(figsize=(max(6.5, 0.7 * W), 5))
    bottom = np.zeros(W)
    for i, lic in enumerate(all_licenses):
        plt.bar(range(W), vals[i], bottom=bottom, label=lic)
        bottom += vals[i]
    plt.xticks(range(W), algs, rotation=30, ha='right')
    plt.ylim(0, 1)
    plt.ylabel('share of license types')
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    ensure_dir(out_path.parent)
    plt.tight_layout(); plt.savefig(out_path.with_suffix('.png'), dpi=220)
    if GENERATE_PDF:
        plt.savefig(out_path.with_suffix('.pdf'))
    plt.close()
