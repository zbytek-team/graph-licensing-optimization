from src.algorithms import (
    ILPSolver,
    GreedyAlgorithm,
    TabuSearch,
    SimulatedAnnealing,  # RE-ENABLED for testing
    GeneticAlgorithm,  # RE-ENABLED for testing
    AntColonyOptimization,
    TreeDynamicProgramming,
    BranchAndBound,
    NaiveAlgorithm,
    DominatingSetAlgorithm,
    RandomizedAlgorithm,
)
from src.graphs import GraphGeneratorFactory, GraphVisualizer
from src.core import LicenseConfigFactory, SolutionValidator
from datetime import datetime
import os
import traceback
import sys

GRAPH_TYPE = "scale_free"
GRAPH_NODES = 20  # TESTING - reduced for faster execution
GRAPH_K = 4
GRAPH_P = 0.3
GRAPH_SEED = 42
LICENSE_CONFIG = "spotify"
ALGORITHMS = [
    ("Dominating Set", lambda: DominatingSetAlgorithm()),
    ("Randomized (70% greedy)", lambda: RandomizedAlgorithm(greedy_probability=0.7, seed=42)),
    ("Randomized (30% greedy)", lambda: RandomizedAlgorithm(greedy_probability=0.3, seed=42)),
    ("Greedy", lambda: GreedyAlgorithm()),
    ("Tabu Search", lambda: TabuSearch()),
    ("Simulated Annealing", lambda: SimulatedAnnealing()),  # RE-ENABLED
    ("Genetic Algorithm", lambda: GeneticAlgorithm()),  # RE-ENABLED
    ("Ant Colony Optimization", lambda: AntColonyOptimization()),
    ("Branch and Bound", lambda: BranchAndBound()),
    ("ILP Solver", lambda: ILPSolver()),
]

# Add naive algorithm only for small graphs (n ≤ 10)
if GRAPH_NODES <= 10:
    ALGORITHMS.insert(0, ("Naive Algorithm", lambda: NaiveAlgorithm()))

if GRAPH_TYPE == "tree":
    ALGORITHMS.append(("Tree Dynamic Programming", lambda: TreeDynamicProgramming()))


def test_algorithm(
    algorithm,
    algorithm_name,
    graph,
    license_types,
    visualizer=None,
    timestamp_folder=None,
):
    print(f"\n{'=' * 20} {algorithm_name} {'=' * 20}")
    try:
        if algorithm_name == "Branch and Bound":
            solution = algorithm.solve(graph, license_types, max_iterations=50000)
        elif algorithm_name == "Simulated Annealing":
            # Example run with custom parameters
            solution = algorithm.solve(
                graph, license_types, T0=100.0, alpha=0.95, max_iter=1000
            )
        else:
            solution = algorithm.solve(graph, license_types)
        print("Solution found!")
        print(f"- Total cost: {solution.total_cost}")
        print(f"- Number of groups: {len(solution.groups)}")
        print(f"- Covered nodes: {len(solution.covered_nodes)}/{len(graph.nodes())}")
        validator = SolutionValidator()
        try:
            is_valid = validator.is_valid_solution(solution, graph)
            print(f"- Solution is valid: {is_valid}")
        except Exception as validation_error:
            print(f"- Solution validation FAILED: {validation_error}")
            print(f"- Algorithm: {algorithm_name}")
            print(f"- Graph nodes: {len(graph.nodes())}")
            print(f"- Solution covered nodes: {len(solution.covered_nodes)}")
            all_nodes = set(graph.nodes())
            missing_nodes = all_nodes - solution.covered_nodes
            extra_nodes = solution.covered_nodes - all_nodes
            if missing_nodes:
                print(f"- Missing nodes ({len(missing_nodes)}): {sorted(list(missing_nodes))}")
            if extra_nodes:
                print(f"- Extra nodes ({len(extra_nodes)}): {sorted(list(extra_nodes))}")
            print(f"- Number of groups: {len(solution.groups)}")
            overlapping_nodes = set()
            all_group_nodes = set()
            for i, group in enumerate(solution.groups):
                group_nodes = group.all_members
                overlap = group_nodes & all_group_nodes
                if overlap:
                    overlapping_nodes.update(overlap)
                    print(f"- Group {i + 1} overlaps with previous groups: {sorted(list(overlap))}")
                all_group_nodes.update(group_nodes)
                if not (group.license_type.min_capacity <= group.size <= group.license_type.max_capacity):
                    print(
                        f"- Group {i + 1} violates size constraints: size={group.size}, "
                        f"min={group.license_type.min_capacity}, max={group.license_type.max_capacity}"
                    )
                owner_neighbors = set(graph.neighbors(group.owner)) | {group.owner}
                invalid_members = group.all_members - owner_neighbors
                if invalid_members:
                    print(f"- Group {i + 1} has invalid members {sorted(list(invalid_members))} not connected to owner {group.owner}")
                    print(f"  Owner neighbors: {sorted(list(owner_neighbors))}")
                    print(f"  Group members: {sorted(list(group.all_members))}")
            if overlapping_nodes:
                print(f"- Total overlapping nodes: {sorted(list(overlapping_nodes))}")
            return float("inf")
        if len(solution.groups) <= 10:
            print("\nLicense groups:")
            for i, group in enumerate(solution.groups, 1):
                print(f"  Group {i}: {group.license_type.name}")
                print(f"    Owner: {group.owner}, Other Members: {sorted(group.additional_members)}")
        else:
            print("\n(Too many groups to display - showing summary only)")
        if visualizer:
            solver_name = algorithm_name.lower().replace(" ", "_")
            visualizer.visualize_solution(
                graph,
                solution,
                solver_name=solver_name,
                timestamp_folder=timestamp_folder,
            )
        return solution.total_cost
    except Exception as e:
        print(f"\n🚨 ALGORITHM FAILURE: {algorithm_name}")
        print(f"📍 Error Type: {type(e).__name__}")
        print(f"💬 Error Message: {str(e)}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_list = traceback.extract_tb(exc_traceback)
        our_frames = [frame for frame in tb_list if "graph-licensing-optimization" in frame.filename]
        if our_frames:
            last_frame = our_frames[-1]
            print(f"📂 File: {last_frame.filename}")
            print(f"📏 Line: {last_frame.lineno}")
            print(f"🔧 Function: {last_frame.name}")
            print(f"💻 Code: {last_frame.line}")
        if exc_traceback:
            frame = exc_traceback.tb_frame
            print("\n🔍 Local Variables at Error:")
            for var_name, var_value in frame.f_locals.items():
                if not var_name.startswith("__"):
                    try:
                        if isinstance(var_value, (list, set, dict)) and len(str(var_value)) > 200:
                            print(f"  {var_name}: {type(var_value).__name__} (length: {len(var_value)})")
                        else:
                            print(f"  {var_name}: {repr(var_value)}")
                    except Exception:
                        print(f"  {var_name}: <unable to display>")
        print("\n📋 Full Stack Trace:")
        traceback.print_exc()
        print("\n🌐 Algorithm Context:")
        print(f"  - Graph nodes: {len(graph.nodes())}")
        print(f"  - Graph edges: {len(graph.edges())}")
        print(f"  - License types: {[lt.name for lt in license_types]}")
        print(f"  - Algorithm class: {algorithm.__class__.__name__}")
        return float("inf")


def main():
    print("Graph Licensing Optimization - Algorithm Comparison")
    print("=" * 60)
    generator = GraphGeneratorFactory.get_generator(GRAPH_TYPE)
    graph = generator(n_nodes=GRAPH_NODES, k=GRAPH_K, p=GRAPH_P, seed=GRAPH_SEED)
    print("Generated small world graph:")
    print(f"- Nodes: {len(graph.nodes())}")
    print(f"- Edges: {len(graph.edges())}")
    print(f"- Average degree: {sum(dict(graph.degree()).values()) / len(graph.nodes()):.2f}")
    license_types = LicenseConfigFactory.get_config(LICENSE_CONFIG)
    print(f"\nUsing license configuration: {LICENSE_CONFIG.upper()}")
    now = datetime.now()
    timestamp_folder = now.strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"results/graphs/{timestamp_folder}", exist_ok=True)
    results = {}
    visualizer = GraphVisualizer()
    for name, algorithm_factory in ALGORITHMS:
        algorithm_instance = algorithm_factory()  # Create fresh instance
        cost = test_algorithm(
            algorithm_instance,
            name,
            graph,
            license_types,
            visualizer=visualizer,
            timestamp_folder=timestamp_folder,
        )
        results[name] = cost
    print(f"\n{'=' * 20} COMPARISON {'=' * 20}")
    best_cost = min(results.values())
    sorted_results = sorted(results.items(), key=lambda x: x[1] if x[1] != float("inf") else float("inf"))
    for name, cost in sorted_results:
        if cost == float("inf"):
            print(f"{name}: FAILED")
        else:
            gap = ((cost - best_cost) / best_cost * 100) if best_cost > 0 else 0
            print(f"{name}: cost={cost:.1f} (gap: {gap:.1f}%)")


if __name__ == "__main__":
    main()
