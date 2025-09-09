from __future__ import annotations

import logging
import os
from pathlib import Path

from glopt.logging_config import get_logger, setup_logging


def test_setup_logging_creates_file_when_run_id() -> None:
    run_id = "test_run_logging"
    try:
        log_path = setup_logging(run_id=run_id)
        assert log_path is not None and Path(log_path).exists()
    finally:
        # cleanup
        base = Path("runs") / run_id
        if base.exists():
            for p in sorted(base.rglob("*"), reverse=True):
                p.unlink(missing_ok=True)
            base.rmdir()


def test_setup_logging_respects_env_levels() -> None:
    os.environ["GLOPT_LOG_LEVELS"] = "glopt.cli=ERROR"
    try:
        setup_logging(run_id=None)
        lvl = logging.getLogger("glopt.cli").level
        assert lvl == logging.ERROR
    finally:
        os.environ.pop("GLOPT_LOG_LEVELS", None)

