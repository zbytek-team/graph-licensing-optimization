from __future__ import annotations

import json
import logging
import logging.config
import os
from pathlib import Path
from typing import Mapping


def _parse_levels_from_env(defaults: Mapping[str, str]) -> dict[str, str]:
    raw = os.getenv("GLOPT_LOG_LEVELS", "").strip()
    if not raw:
        return dict(defaults)
    out: dict[str, str] = dict(defaults)
    # allow JSON mapping or comma-separated k=v
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
            for k, v in data.items():
                out[str(k)] = str(v).upper()
            return out
        except Exception:
            pass
    for part in raw.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip().upper()
    return out


def setup_logging(
    *,
    run_id: str | None = None,
    base_dir: str | Path = "runs",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    default_levels: Mapping[str, str] | None = None,
) -> Path | None:
    """Configure unified logging for CLI and libraries.

    - Console handler at `console_level`.
    - File handler at `file_level` under `runs/<run_id>/glopt.log` (unless GLOPT_NO_FILE_LOG=1).
    - Per-module levels controlled via env `GLOPT_LOG_LEVELS` or `default_levels`.

    Returns the log file path (if enabled), else None.
    """
    defaults = default_levels or {
        "glopt": "WARNING",
        "glopt.cli": console_level,
        "glopt.core": "INFO",
        "glopt.algorithms": "WARNING",
        "glopt.io": "INFO",
    }
    levels = _parse_levels_from_env(defaults)

    log_file: Path | None = None
    no_file = os.getenv("GLOPT_NO_FILE_LOG", "0") == "1"
    if not no_file and run_id:
        base = Path(base_dir) / run_id
        base.mkdir(parents=True, exist_ok=True)
        log_file = base / "glopt.log"

    handlers: dict[str, dict] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": console_level,
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        }
    }
    if log_file is not None:
        handlers["file"] = {
            "class": "logging.FileHandler",
            "level": file_level,
            "formatter": "detailed",
            "filename": str(log_file),
            "encoding": "utf-8",
        }

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                    "datefmt": "%H:%M:%S",
                },
                "detailed": {
                    "format": "[%(asctime)s] %(levelname)s %(name)s (%(process)d): %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": handlers,
            "root": {"level": "WARNING", "handlers": list(handlers.keys())},
            "loggers": {k: {"level": v} for k, v in levels.items()},
        }
    )

    return log_file


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name or __name__)


def log_run_banner(logger: logging.Logger, title: str, params: Mapping[str, object]) -> None:
    logger.info("== %s ==", title)
    for k in sorted(params.keys()):
        logger.info("%s: %s", k, params[k])


def log_run_footer(logger: logging.Logger, summary: Mapping[str, object]) -> None:
    logger.info("== summary ==")
    for k in sorted(summary.keys()):
        logger.info("%s: %s", k, summary[k])
