import csv
from typing import Dict
import networkx as nx


def generate_graph_report(graph: nx.Graph, output_path: str) -> None:
    """Generate basic statistics for a graph and save to CSV or Markdown.

    Parameters
    ----------
    graph: nx.Graph
        Graph for which statistics will be computed.
    output_path: str
        Path to the output file. If it ends with ``.csv`` the report will be
        written in CSV format, otherwise a Markdown table will be produced.
    """
    if graph.number_of_nodes() == 0:
        avg_degree = 0.0
    else:
        avg_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
    stats: Dict[str, float] = {
        "average_degree": avg_degree,
        "density": nx.density(graph),
        "average_clustering": nx.average_clustering(graph),
    }

    if output_path.lower().endswith(".csv"):
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for key, value in stats.items():
                writer.writerow([key, f"{value:.4f}"])
    else:
        lines = ["| Metric | Value |", "| --- | --- |"]
        for key, value in stats.items():
            metric_name = key.replace("_", " ").title()
            lines.append(f"| {metric_name} | {value:.4f} |")
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
