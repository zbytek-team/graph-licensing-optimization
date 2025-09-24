.PHONY: help install install-analysis test quick pipelines thesis thesis-clean \
        clean clean-build clean-results clean-runs benchmark benchmark_real \
        dynamic extensions extensions_dynamic all ensure-uv

VENV ?= .venv
PYTHON ?= python3
PYTHON_BIN := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
RUNNER := PYTHONPATH=src $(PYTHON_BIN)
UV ?= uv
UV_CACHE_DIR ?= .uv_cache
THESIS_DIR := docs/thesis
THESIS_BUILD := $(THESIS_DIR)/build

# Używaj przez: $(UV_RUN) python -m modul ...
UV_RUN := UV_CACHE_DIR=$(UV_CACHE_DIR) $(UV) run

help:
	@echo "Available targets:"
	@echo "  install           - create virtualenv and install glopt + pytest"
	@echo "  install-analysis  - install optional analysis extras"
	@echo "  test              - run pytest suite"
	@echo "  quick             - run the quick CLI pipeline"
	@echo "  pipelines/<name>  - run a named CLI pipeline (see python -m glopt.cli.pipelines list)"
	@echo "  thesis            - build the thesis PDF (tectonic preferred)"
	@echo "  thesis-clean      - remove thesis build artifacts"
	@echo "  clean             - drop caches and build outputs"

$(PYTHON_BIN):
	$(PYTHON) -m venv $(VENV)
	$(PYTHON_BIN) -m pip install --upgrade pip

# Opcjonalnie: auto-instalacja uv jeśli brak
ensure-uv:
	@command -v $(UV) >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

# Instalacja zależności projektu + dev (domyślne zachowanie uv)
install: ensure-uv
	$(UV) sync

install-analysis: ensure-uv
	$(UV) sync --extra analysis

pipelines/%: ensure-uv
	$(UV_RUN) -m glopt.cli.pipelines $*

quick: pipelines/quick

pipelines: pipelines/list

pipelines/list: ensure-uv
	$(UV_RUN) -m glopt.cli.pipelines list

thesis: $(PYTHON_BIN)
	@if command -v tectonic >/dev/null 2>&1; then \
		echo "[thesis] Using tectonic"; \
		mkdir -p $(THESIS_BUILD); \
		tectonic -X compile --keep-logs --keep-intermediates --outdir $(THESIS_BUILD) $(THESIS_DIR)/main.tex; \
	else \
		echo "[thesis] Tectonic not available; using LuaLaTeX + Biber"; \
		mkdir -p $(THESIS_BUILD); \
		( cd $(THESIS_DIR) && lualatex -interaction=nonstopmode -output-directory=build main.tex || true ); \
		( cd $(THESIS_DIR) && biber build/main || true ); \
		( cd $(THESIS_DIR) && lualatex -interaction=nonstopmode -output-directory=build main.tex || true ); \
		( cd $(THESIS_DIR) && lualatex -interaction=nonstopmode -output-directory=build main.tex || true ); \
	fi

thesis-clean:
	rm -rf $(THESIS_BUILD)

clean-build:
	rm -rf .pytest_cache __pycache__ $(THESIS_BUILD)

clean-results:
	rm -rf results runs

clean: clean-build
clean-runs:
	rm -rf runs

# Analizy z użyciem uv run (auto lock+sync)
benchmark:
	$(UV_RUN) python scripts/benchmark/analyze.py
benchmark_real:
	$(UV_RUN) python scripts/benchmark_real/analyze.py
dynamic:
	$(UV_RUN) python scripts/dynamic/analyze.py
extensions:
	$(UV_RUN) python scripts/extensions/analyze.py
extensions_dynamic:
	$(UV_RUN) python scripts/extensions_dynamic/analyze.py

all: benchmark benchmark_real dynamic extensions extensions_dynamic
