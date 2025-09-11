from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .commons import ensure_dir, load_rows


def export_run_csv(run_id: str, out_csv: Path) -> Path:
    run_dir = Path("runs") / run_id
    csvs = sorted(run_dir.glob("csv/*.csv"))
    if not csvs:
        raise SystemExit(f"No CSV files under {run_dir}/csv")
    rows: list[dict[str, object]] = []
    for p in csvs:
        try:
            rows.extend(load_rows(p))
        except Exception:
            pass
    ensure_dir(out_csv.parent)
    headers: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in headers:
                headers.append(k)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = export_run_csv(a.run, Path(a.out))
    print(out)


if __name__ == "__main__":
    main()
