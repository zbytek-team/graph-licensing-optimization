# Graph Licensing Optimization

This repository contains tools and experiments for the graph licensing optimization problem.

## Generating reports

Use `scripts/generate_reports.py` to visualize benchmark results stored in a CSV file. The script creates summary plots for selected metrics and comparison plots of algorithms across graph sizes.

Example:

```bash
python scripts/generate_reports.py path/to/results.csv --metric cost --metric execution_time
```

Images are saved in the `results/plots/` directory.
