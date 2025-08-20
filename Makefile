.PHONY: test all benchmark dynamic format lint install

test:
	uv run -m scripts.run_test

all:
	uv run -m scripts.run_all

benchmark:
	uv run -m scripts.run_benchmark

dynamic:
	uv run -m scripts.run_dynamic

format:
	uv run --with black black --line-length 160 src scripts

lint:
	uv run --with ruff ruff check --fix src scripts

install:
	uv sync
