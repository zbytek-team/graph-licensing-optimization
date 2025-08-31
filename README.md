# Graph Licensing Optimization

Simple application for exploring cost-efficient license allocation in social networks.

## Repository structure

- `src/` – lightweight Python package with models, algorithms and helpers
- `scripts/` – runnable scripts used for experiments and demos
- `Makefile` – helper targets for common tasks (`test`, `benchmark`, `dynamic`, `format`, `lint`, `install`)

## Quick start

```sh
make install      # create virtual environment and install dependencies
make test         # execute unit tests verifying algorithm quality
make dynamic      # execute dynamic network simulation
```

Simulation and benchmark outputs are stored in the `results/` directory.
