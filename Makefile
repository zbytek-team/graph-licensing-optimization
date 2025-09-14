.PHONY: install test lint format typecheck clean \
    benchmark benchmark_real dynamic dynamic_real trees \
    custom all_cli analyze report merge-runs check-python thesis-figs \
    new-figures \
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
	uv run --with black black src scripts

typecheck:
	$(PYPATH) uv run --with basedpyright basedpyright -p pyproject.toml --pythonversion 3.13 src scripts

typecheck-src:
	$(PYPATH) uv run --with basedpyright basedpyright -p pyproject.toml --pythonversion 3.13 src

clean:
	rm -rf .pytest_cache .ruff_cache .venv __pycache__

# ------------------------------------------------------------
# Python version check (must be 3.13)
# ------------------------------------------------------------
check-python:
	uv run -c "import sys; v=sys.version_info; assert (v.major,v.minor)==(3,13), f'Need Python 3.13.x; got {sys.version}'; print(f'Python {sys.version.split()[0]} OK')"

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

# Trees-only benchmark (tree graphs, includes tree DP)
trees:
	$(PYPATH) uv run -m glopt.cli.trees

# ------------------------------------------------------------
# CLI (single-run helpers)
# ------------------------------------------------------------
custom:
	$(PYPATH) uv run -m glopt.cli.custom

# Note: target name is 'all_cli' to avoid conflict with the
# convenience 'all' target defined at the bottom.
all_cli:
	$(PYPATH) uv run -m glopt.cli.all

# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------
analyze:
	$(PYPATH) uv run scripts/analysis/main.py

report:
	$(PYPATH) uv run scripts/analysis/mk_report.py --run $(RUN)


# Merge multiple runs into a single combined run with CSV and optional Excel
# Usage:
#   make merge-runs OUT=benchmark_real_combined RUNS="benchmark_real_all benchmark_real_ant_clony benchmark_real_ilp"
merge-runs:
	$(PYPATH) uv run -m scripts.analysis.merge_runs --out $(OUT) $(if $(OUT_CSV),--out-csv $(OUT_CSV),) $(RUNS)

# Export a run/<id>/csv/*.csv into a single CSV anywhere
export-run-csv:
	$(PYPATH) uv run -m scripts.analysis.export_run_csv --run $(RUN) --out $(OUT)

# Analyze an arbitrary CSV into an output directory under results/
analyze-csv:
	$(PYPATH) uv run -m scripts.analysis.analyze_csv --csv $(CSV) --out $(OUT)

# Create/update results/README.md with concise, data-driven conclusions
results-readme:
	$(PYTHON) scripts/analysis/mk_results_readme.py

# Export curated set of figures to thesis assets
thesis-figs:
	$(PYTHON) scripts/analysis/export_thesis_figs.py

# Generate a fresh set of figures into results/new_figures (or OUT=<dir>)
# Options:
#   OUT=results/new_figures_custom   # change output directory
#   FIG_MAX_DRAW_NODES=300           # cap nodes drawn for large graphs
#   ANALYZE_PDF=1                    # also emit PDFs next to PNGs
new-figures:
	# Use system Python for speed/stability; avoid slow `uv run` here
	# Requires deps installed (run `make install` once).
	$(PYPATH) MPLBACKEND=Agg FIG_MAX_DRAW_NODES=$(FIG_MAX_DRAW_NODES) ANALYZE_PDF=$(ANALYZE_PDF) \
		$(PYTHON) -m scripts.analysis.mk_new_figures $(if $(OUT),--out $(OUT),)

# Convenience
all: install lint test benchmark analyze
