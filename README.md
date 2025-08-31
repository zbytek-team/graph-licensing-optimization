# Graph Licensing Optimization

Simple application for exploring cost-efficient license allocation in social networks.

## Repository structure

- `src/glopt/` – project package with models, algorithms, I/O and CLI helpers
- `Makefile` – helper targets for common tasks (`test`, `benchmark`, `dynamic`, `format`, `lint`, `install`)

## Quick start

```sh
make install      # create virtual environment and install dependencies
make test         # execute unit tests verifying algorithm quality
make dynamic      # execute dynamic network simulation
```

Simulation and benchmark outputs are stored in the `runs/` directory.
