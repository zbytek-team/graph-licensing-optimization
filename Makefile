.PHONY: run benchmark format lint install

run:
	uv run algorithm_comparison.py

benchmark:
	uv run benchmark.py

format:
	uv run --with black black --line-length 160 src/ *.py

lint:
	uv run --with ruff ruff check --fix src/ *.py

install:
	uv sync
