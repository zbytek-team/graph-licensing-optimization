.PHONY: test all benchmark dynamic format lint install

test:
	PYTHONPATH=src uv run --with pytest pytest -vv -q

all:
	PYTHONPATH=src uv run -m glopt.cli.all

benchmark:
	PYTHONPATH=src uv run -m glopt.cli.benchmark

dynamic:
	PYTHONPATH=src uv run -m glopt.cli.dynamic

format:
	uv run --with black black --line-length 160 src tests

lint:
	uv run --with ruff ruff check --fix src tests

install:
	uv sync
