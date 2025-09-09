# Graph Licensing Optimization (GLOPT)

Optimize the cost of software licenses in social networks. Model users as a graph and decide who buys an individual plan vs. a group plan to minimize total cost. Includes exact methods (ILP), heuristics, metaheuristics (GA/SA/Tabu/ACO), a static benchmark (synthetic and real graphs), and a dynamic benchmark (warm vs. cold start).

## Requirements
- Python 3.13 (exact minor)
- [uv](https://github.com/astral-sh/uv) for environment management (used by Makefile)

This project requires Python 3.13.x explicitly. The version is enforced by:
- `pyproject.toml` → `requires-python = ">=3.13"` with tools targeting `py313`.
- `.python-version` → `3.13` (used by pyenv/uv discovery).
- Import‑time guard in `glopt` that aborts on non‑3.13.

Recommended setup with uv:
```
# Ensure uv manages/uses Python 3.13
uv python install 3.13
uv venv --python 3.13

# Sync project dependencies
make install
```

## Install
```
make install
```

## Run benchmarks
- Static benchmark on synthetic graphs (uses on-disk cache, 60s timeout per run):
```
make benchmark
```
- Static benchmark on real graphs (Facebook ego networks in `data/facebook`):
```
make benchmark_real
```
- Dynamic benchmark (synthetic graphs): warm vs. cold start across mutation steps:
```
make dynamic
```
- Dynamic benchmark on real graphs:
```
make dynamic_real
```

Outputs are stored under `runs/<run_id>/*` (CSV under `runs/<run_id>/csv`).

### Logging
- All CLI commands log to console (INFO) and to a file: `runs/<run_id>/glopt.log` (DEBUG).
- Control log levels per module via env var `GLOPT_LOG_LEVELS`, e.g.:
  - `GLOPT_LOG_LEVELS="glopt.cli=INFO,glopt.algorithms=WARNING,glopt.core=INFO" make benchmark`
- To disable file logging set `GLOPT_NO_FILE_LOG=1`.

## Analyze results
Modular analysis that generates plots and tables under `runs/<run_id>/analysis/`:
```
make analyze
```
Set `ANALYZE_PDF=1` to also output PDFs.

## Common tasks
```
make test      # run unit tests
make lint      # ruff lint + fix
make format    # black formatting
```

## Project layout
- `src/glopt/` — core models, validators, algorithms, I/O, and CLI entrypoints
- `scripts/analysis/` — modular analysis (main entry: `scripts/analysis/main.py`)
- `data/` — graph caches and real datasets (e.g., `data/facebook` ego-networks)
- `runs/` — outputs (CSV, plots, aggregates)

## Thesis Mapping (Master’s)
- Goal: show that optimal low-cost license planning for a friend network (e.g., Duolingo Super: 1-person vs family for 6 people) corresponds to Roman domination in graphs.
- Static experiments: `make benchmark` (+ `benchmark_real`) — evaluate exact/heuristic/metaheuristic methods and price variants.
- Dynamic experiments: `make dynamic` (+ `dynamic_real`) — “non-simultaneous purchase” via mutations in the network.
- License models:
  - `duolingo_super` — realistic prices (1-seat and up to 6 seats).
  - `duolingo_p_<k>` — normalized Duolingo-style with 1-seat price=1.0 and family price=k (capacity=6), e.g., `duolingo_p_2_0`, `duolingo_p_3_0`.
  - `roman_domination` — normalized Roman model: Solo=1.0, Group=2.0, unbounded capacity (canonical equivalence).
  - `roman_p_<k>` — Roman model sweep: Group price=k, unbounded capacity.
- How this aligns with tasks:
  - Social network model → `src/glopt/io/graph_generator.py` + real ego-nets.
  - License schemes → `src/glopt/license_config.py`.
  - Roman domination mapping → `src/glopt/core/models.py` + `roman_*` configs.
  - Methods and tools → `src/glopt/algorithms/*` + ILP/heuristics/metaheuristics.
  - Other price versions → `roman_p_*`, `duolingo_p_*`.
  - Dynamic version → `dynamic*.py` (mutations and warm/cold starts).

## Notes
- Benchmarks enforce a hard 60s cap per algorithm run by killing the subprocess. Larger sizes for a (graph, algorithm) pair are skipped after the first timeout.
- License sweeps: use special configs like `roman_p_2_5` (group cost = 2.5) to evaluate price sensitivity.
