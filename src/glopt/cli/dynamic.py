from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime

from glopt.dynamic_simulator import DynamicNetworkSimulator
from glopt.io import build_paths, ensure_dir
from glopt.io.graph_generator import GraphGeneratorFactory
from glopt.license_config import LicenseConfigFactory

GRAPH_NAME = "small_world"
GRAPH_PARAMS = {"k": 4, "p": 0.1, "seed": 42}
N_NODES = 30
LICENSE_CONFIG = "roman_domination"
NUM_STEPS = 5


def main() -> int:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    _, _, csv_dir = build_paths(run_id)

    try:
        gen = GraphGeneratorFactory.get(GRAPH_NAME)
        graph = gen(n_nodes=N_NODES, **GRAPH_PARAMS)
    except Exception:
        traceback.print_exc(limit=10, file=sys.stderr)
        return 2

    try:
        license_types = LicenseConfigFactory.get_config(LICENSE_CONFIG)
    except Exception:
        traceback.print_exc(limit=10, file=sys.stderr)
        return 2

    simulator = DynamicNetworkSimulator()
    simulator.simulate(graph, license_types, num_steps=NUM_STEPS)

    try:
        ensure_dir(csv_dir)
        out_path = os.path.join(csv_dir, f"{run_id}_dynamic.csv")
        simulator.export_history_to_csv(out_path)
    except Exception:
        traceback.print_exc(limit=10, file=sys.stderr)
        return 2

    simulator.get_simulation_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
