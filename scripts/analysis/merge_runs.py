from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from .commons import ensure_dir, load_rows


def _safe_sheet_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
    return s[:31] or "sheet"


def merge_runs(out_run: str, runs: list[str], out_csv_path: Path | None = None) -> Path:
    """Merge CSVs from multiple runs/ into a single combined CSV under runs/<out_run>/csv.

    Also attempts to write an Excel workbook (one sheet per input run + combined) if pandas is available.
    Returns the combined CSV path.
    """
    runs_dir = Path("runs")
    if out_csv_path is not None:
        ensure_dir(out_csv_path.parent)
        combined_csv = out_csv_path
        out_dir = combined_csv.parent
    else:
        out_dir = runs_dir / out_run
        csv_dir = out_dir / "csv"
        ensure_dir(csv_dir)
        combined_csv = csv_dir / f"{out_run}.csv"

    # Collect rows from each input run
    all_rows: list[dict[str, object]] = []
    per_run: dict[str, list[dict[str, object]]] = {}
    for r in runs:
        rdir = runs_dir / r
        csvs = sorted(rdir.glob("csv/*.csv"))
        rows_r: list[dict[str, object]] = []
        for p in csvs:
            try:
                rows_r.extend(load_rows(p))
            except Exception:
                pass
        per_run[r] = rows_r
        all_rows.extend(rows_r)

    if not all_rows:
        raise SystemExit("No CSV rows found in the specified runs.")

    # Write combined CSV (superset of headers)
    headers: list[str] = []
    for rs in [all_rows] + list(per_run.values()):
        for r in rs:
            for k in r.keys():
                if k not in headers:
                    headers.append(k)

    with combined_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # Optional Excel workbook
    try:
        import pandas as pd  # type: ignore

        xlsx_path = combined_csv.parent / f"{combined_csv.stem}.xlsx"
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:  # type: ignore[call-arg]
            # Individual sheets
            for r, rs in per_run.items():
                if not rs:
                    continue
                df = pd.DataFrame(rs)
                df.to_excel(writer, sheet_name=_safe_sheet_name(r), index=False)
            # Combined sheet
            pd.DataFrame(all_rows).to_excel(writer, sheet_name=_safe_sheet_name("combined"), index=False)
    except Exception:
        # pandas not installed or writer unavailable -- skip Excel
        pass

    return combined_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="name of the output run directory under runs/ (ignored if --out-csv is set)")
    ap.add_argument("--out-csv", default="", help="optional explicit output CSV path (e.g., results/benchmark_real_finale.csv)")
    ap.add_argument("runs", nargs="+", help="list of run ids under runs/ to merge")
    args = ap.parse_args()

    out_csv_path = Path(args.out_csv) if args.out_csv else None
    out_csv = merge_runs(args.out, args.runs, out_csv_path)
    print(out_csv)


if __name__ == "__main__":
    main()
