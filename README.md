# Graph Licensing Optimization

This project explores algorithms for optimizing licensing strategies on graphs.

## Graph Reports

Before executing algorithms, scripts such as `scripts/run_comparison.py` and `scripts/run_benchmark.py` generate a simple report with basic statistics of the created graph. The report is saved as a Markdown table in the `results` folder.

Example output:

| Metric | Value |
| --- | --- |
| Average Degree | 1.50 |
| Density | 0.50 |
| Average Clustering | 0.00 |

To generate a report manually:

```python
from src.graphs import generate_graph_report
import networkx as nx

G = nx.path_graph(4)
generate_graph_report(G, "example_report.md")
```

This creates `example_report.md` with the table shown above.
