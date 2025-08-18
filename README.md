## Benchmark script

The benchmarking utility can be executed with various command-line options to
control graph generation and algorithm selection.

```bash
python scripts/run_benchmark.py [options]
```

### Available options

| Option | Description | Default |
|--------|-------------|---------|
| `--graph-type` | Graph generators to use (`random`, `scale_free`, `small_world`). Multiple values can be provided. | `random scale_free small_world` |
| `--nodes` | Numbers of nodes for generated graphs. Accepts multiple integers. | `10 15 20 25 30` |
| `--license-config` | License configurations to benchmark. | `roman_domination duolingo_super spotify` |
| `--algorithm` | Algorithms to run (`greedy`, `tabu`, `ilp`). | `greedy tabu ilp` |
| `--output-dir` | Directory for the resulting CSV files. | `results/stats` |
| `--seed` | Random seed used for graph generation. | `42` |

Example usage:

```bash
python scripts/run_benchmark.py --graph-type random --nodes 20 30 --algorithm greedy
```

