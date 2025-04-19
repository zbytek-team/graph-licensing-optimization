from pathlib import Path
import time
from typing import Protocol, TypedDict
import networkx as nx
from src.algorithms.base import BaseSolver
from src.algorithms.static.greedy_basic import GreedyBasicSolver
from src.algorithms.static.ilp_solver import ILPSolver
from src.utils.solution_utils import (
    calculate_cost,
    validate_solution,
)
from src.utils.graph_vis import visualize_graph
from src.data.generators import (
    generate_erdos_renyi,
    generate_barabasi_albert,
)


class GraphGenerator(Protocol):
    def __call__(self, *args, **kwargs) -> nx.Graph: ...


class GraphConfig(TypedDict):
    name: str
    generator: GraphGenerator
    kwargs: dict[str, object]
    c_single: float
    c_group: float
    group_size: int


class SolverConfig(TypedDict):
    name: str
    cls: type[BaseSolver]
    kwargs: dict[str, object]


GRAPH_CONFIGS: list[GraphConfig] = [
    {
        "name": "ER_50_0_05",
        "generator": generate_erdos_renyi,
        "kwargs": {"n": 50, "p": 0.05, "seed": 42},
        "c_single": 3.0,
        "c_group": 5.0,
        "group_size": 3,
    },
    {
        "name": "BA_100_3",
        "generator": generate_barabasi_albert,
        "kwargs": {"n": 300, "m": 3, "seed": 7},
        "c_single": 1.0,
        "c_group": 2.5,
        "group_size": 6,
    },
]

SOLVER_CONFIGS: list[SolverConfig] = [
    {
        "name": "Greedy",
        "cls": GreedyBasicSolver,
        "kwargs": {},
    },
    {
        "name": "ILP",
        "cls": ILPSolver,
        "kwargs": {},
    },
]


def run_single(
    solver_cfg: SolverConfig,
    graph: nx.Graph,
    c_single: float,
    c_group: float,
    group_size: int,
    out_dir: Path,
) -> None:
    solver: BaseSolver = solver_cfg["cls"](**solver_cfg["kwargs"])
    start: float = time.perf_counter()
    solution = solver.solve(graph, c_single, c_group, group_size)
    duration: float = time.perf_counter() - start
    cost: float = calculate_cost(solution, c_single, c_group)
    errors = validate_solution(graph, solution, group_size)
    status = "OK" if not errors else f"INVALID:{'|'.join(errors)}"
    img_path: Path = out_dir / f"{solver_cfg['name']}.png"
    visualize_graph(
        graph,
        solution,
        title=f"{solver_cfg['name']} cost={cost:.2f} t={duration:.3f}s {status}",
        show=False,
        save_path=str(img_path),
    )
    print(f"{solver_cfg['name']:>10}  cost={cost:8.2f}  time={duration:6.3f}s  status={status}  img={img_path}")


def main() -> None:
    Path("results/images").mkdir(parents=True, exist_ok=True)
    for gcfg in GRAPH_CONFIGS:
        print(f"\n=== Graph: {gcfg['name']} ===")
        graph: nx.Graph = gcfg["generator"](**gcfg["kwargs"])
        out_dir: Path = Path("results/images") / gcfg["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        for scfg in SOLVER_CONFIGS:
            run_single(
                scfg,
                graph,
                gcfg["c_single"],
                gcfg["c_group"],
                gcfg["group_size"],
                out_dir,
            )


if __name__ == "__main__":
    main()
