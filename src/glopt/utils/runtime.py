from __future__ import annotations

import sys


def ensure_python_313() -> None:
    """Abort if interpreter is not Python 3.13.x.

    We enforce major==3 and minor==13 exactly, to guarantee consistent
    behavior with tooling and compiled wheels used in benchmarks.
    """
    vi = sys.version_info
    if not (vi.major == 3 and vi.minor == 13):
        raise RuntimeError(
            f"Python 3.13 is required; detected {vi.major}.{vi.minor}.{vi.micro}.\n"
            "Install 3.13 and re-run (see README 'Requirements')."
        )

