.PHONY: install test lint format clean \
	benchmark benchmark_real dynamic dynamic_real analyze \
	all

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------
PYTHON := python3
PYPATH := PYTHONPATH=src

# ------------------------------------------------------------
# Setup & QA
# ------------------------------------------------------------
install:
	uv sync

test:
	$(PYPATH) uv run --with pytest pytest -q -vv

lint:
	uv run --with ruff ruff check --fix src scripts

format:
	uv run --with black black --line-length 160 src scripts

clean:
	rm -rf .pytest_cache .ruff_cache .venv __pycache__

# ------------------------------------------------------------
# Benchmarks
# ------------------------------------------------------------
benchmark:
	$(PYPATH) uv run -m glopt.cli.benchmark

benchmark_real:
	$(PYPATH) uv run -m glopt.cli.benchmark_real

dynamic:
	$(PYPATH) uv run -m glopt.cli.dynamic

dynamic_real:
	$(PYPATH) uv run -m glopt.cli.dynamic_real

# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------
analyze:
	$(PYTHON) scripts/analysis/main.py

# Convenience
all: install lint test benchmark analyze
