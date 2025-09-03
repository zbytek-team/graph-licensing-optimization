# Graph Licensing Optimization (GLOPT)

Optimize the cost of software licenses in social networks. Model users as a graph and decide who buys an individual plan vs. a group plan to minimize total cost. Includes exact methods (ILP), heuristics, metaheuristics (GA/SA/Tabu/ACO), a static benchmark (synthetic and real graphs), and a dynamic benchmark (warm vs. cold start).

## Requirements
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for environment management (used by Makefile)

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

## Notes
- Benchmarks enforce a hard 60s cap per algorithm run by killing the subprocess. Larger sizes for a (graph, algorithm) pair are skipped after the first timeout.
- License sweeps: use special configs like `roman_p_2_5` (group cost = 2.5) to evaluate price sensitivity.
