# glopt thesis module

Minimal tree prepared for thesis submission. The `src/glopt/experiments` package
contains lightweight runners around the core optimisation algorithms:

- `src.glopt.experiments.single` - run a single synthetic scenario for quick sanity
  checks.
- `src.glopt.experiments.static_synthetic` - sweep synthetic graphs.
- `src.glopt.experiments.static_real` - run on the Facebook ego networks.
- `src.glopt.experiments.dynamic` - dynamic graph experiments for synthetic variants.
- `src.glopt.experiments.pipeline` - convenience entrypoint that executes the
  sequential workflow (static synthetic -> static real -> dynamic variations ->
  extensions).

## Usage

Assuming [`uv`](https://github.com/astral-sh/uv) is installed, execute
commands from the repository root:

```bash
UV_CACHE_DIR=.cache/uv PYTHONPATH=src uv run python -m src.glopt.experiments.single
```

To run the full pipeline:

```bash
UV_CACHE_DIR=.cache/uv PYTHONPATH=src uv run python -m src.glopt.experiments.pipeline
```
