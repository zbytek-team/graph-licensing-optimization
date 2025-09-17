.PHONY: install test clean quick full thesis thesis-clean thesis-all all

install:
	python3 -m venv .venv && . .venv/bin/activate && python -m pip -q install -U pip && python -m pip -q install -e . && python -m pip -q install pytest

test:
	. .venv/bin/activate && PYTHONPATH=src pytest -q -vv

clean:
	rm -rf .venv .pytest_cache __pycache__ runs results docs/reports docs/thesis/build

quick:
	. .venv/bin/activate && PYTHONPATH=src python -m glopt.cli.pipelines quick

full:
	. .venv/bin/activate && PYTHONPATH=src python -m glopt.cli.pipelines full

thesis:
	if command -v tectonic >/dev/null 2>&1; then tectonic -X compile --keep-logs --keep-intermediates --outdir docs/thesis/build docs/thesis/main.tex; else mkdir -p docs/thesis/build && cd docs/thesis && lualatex -interaction=nonstopmode -output-directory=build main.tex && biber build/main && lualatex -interaction=nonstopmode -output-directory=build main.tex && lualatex -interaction=nonstopmode -output-directory=build main.tex; fi

thesis-clean:
	rm -rf docs/thesis/build

thesis-all:
	$(MAKE) full && $(MAKE) thesis

all:
	$(MAKE) install && $(MAKE) quick

# ------------------------------------------------------------
# Thesis PDF build (portable)
# ------------------------------------------------------------
thesis:
	# Prefer tectonic for automatic package fetching
	@if command -v tectonic >/dev/null 2>&1; then \
		echo "[thesis] Using tectonic"; \
		mkdir -p docs/thesis/build; \
		tectonic -X compile --keep-logs --keep-intermediates --outdir docs/thesis/build docs/thesis/main.tex; \
	else \
		echo "[thesis] Tectonic not found; trying LuaLaTeX + Biber"; \
		mkdir -p docs/thesis/build; \
		( cd docs/thesis && lualatex -interaction=nonstopmode -output-directory=build main.tex || true ); \
		( cd docs/thesis && biber build/main || true ); \
		( cd docs/thesis && lualatex -interaction=nonstopmode -output-directory=build main.tex || true ); \
		( cd docs/thesis && lualatex -interaction=nonstopmode -output-directory=build main.tex || true ); \
	fi

thesis-clean:
	rm -rf docs/thesis/build
