.PHONY: test all benchmark dynamic format lint install run_custom

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

# Run the configurable CLI test runner.
# Parameters (optional):
#   GRAPH   - graph name (e.g., small_world, random)
#   N       - number of nodes (e.g., 100)
#   PARAMS  - space-separated key=value graph params (e.g., "p=0.1 seed=42")
#   LICENSE - license config name (e.g., spotify)
#   ALGOS   - comma-separated algorithms or pass multiple --algo via spaces (e.g., "GreedyAlgorithm,SimulatedAnnealing")
#   RUN_ID  - custom run id
#   LIMIT   - print issue limit (0 to suppress)
run-custom:
	PYTHONPATH=src uv run -m glopt.cli.run_custom \
		$(if $(GRAPH),--graph-name $(GRAPH),) \
		$(if $(N),--n-nodes $(N),) \
		$(foreach p,$(PARAMS),--param $(p) ) \
		$(if $(LICENSE),--license $(LICENSE),) \
		$(if $(ALGOS),--algo $(ALGOS),) \
		$(if $(RUN_ID),--run-id $(RUN_ID),) \
		$(if $(LIMIT),--print-issue-limit $(LIMIT),)
