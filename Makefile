UV_RUN=UV_CACHE_DIR=.cache/uv PYTHONPATH=src uv run

.PHONY: single static-synthetic static-real dynamic pipeline analyze clean-cache

single:
	$(UV_RUN) python -m glopt.experiments.single

static-synthetic:
	$(UV_RUN) python -m glopt.experiments.static_synthetic

static-real:
	$(UV_RUN) python -m glopt.experiments.static_real

dynamic:
	$(UV_RUN) python -m glopt.experiments.dynamic

pipeline:
	$(UV_RUN) python -m glopt.experiments.pipeline

analyze:
	$(UV_RUN) python glopt/analysis/static_synthetic_analyze.py
	$(UV_RUN) python glopt/analysis/static_real_analyze.py
	$(UV_RUN) python glopt/analysis/dynamic_analyze.py
	$(UV_RUN) python glopt/analysis/extensions_analyze.py
	$(UV_RUN) python glopt/analysis/extensions_dynamic_analyze.py
