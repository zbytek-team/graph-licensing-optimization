from __future__ import annotations

import os
from datetime import datetime

from glopt.dynamic_simulator import DynamicNetworkSimulator
from glopt.io import build_paths, ensure_dir
from glopt.io.graph_generator import GraphGeneratorFactory
from glopt.license_config import LicenseConfigFactory

# Configuration
GRAPH_NAME = "small_world"
GRAPH_PARAMS = {"k": 4, "p": 0.1, "seed": 42}
N_NODES = 30
LICENSE_CONFIG = "roman_domination"
NUM_STEPS = 5


def main() -> int:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _, _, csv_dir = build_paths(run_id)

    print("== glopt dynamic ==")
    print(f"run_id: {run_id}")
    print(f"graph: {GRAPH_NAME} n={N_NODES} params={GRAPH_PARAMS}")
    print(f"license: {LICENSE_CONFIG}")
    print(f"steps: {NUM_STEPS}")

    gen = GraphGeneratorFactory.get(GRAPH_NAME)
    graph = gen(n_nodes=N_NODES, **GRAPH_PARAMS)
    license_types = LicenseConfigFactory.get_config(LICENSE_CONFIG)

    simulator = DynamicNetworkSimulator()
    simulator.simulate(graph, license_types, num_steps=NUM_STEPS)

    ensure_dir(csv_dir)
    out_path = os.path.join(csv_dir, f"{run_id}_dynamic.csv")
    simulator.export_history_to_csv(out_path)

    summary = simulator.get_simulation_summary()
    print("== summary ==")
    print(
        f"initial_cost={summary.get('initial_cost', 0):.2f} final_cost={summary.get('final_cost', 0):.2f} "
        f"total_change={summary.get('total_cost_change', 0):.2f} steps={summary.get('num_steps', 0)}"
    )
    print(f"csv: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
