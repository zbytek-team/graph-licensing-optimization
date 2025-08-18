from pathlib import Path
import runpy
import sys


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root))
    runpy.run_path(project_root / "scripts" / "run_comparison.py", run_name="__main__")
