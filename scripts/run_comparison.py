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
import argparse
import logging

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


logger = logging.getLogger(__name__)


def test_algorithm(
    algorithm,
    algorithm_name,
    graph,
    license_types,
    visualizer=None,
    timestamp_folder=None,
):
    logger.info("\n%s", f"{'=' * 20} {algorithm_name} {'=' * 20}")
    try:
        if algorithm_name == "Branch and Bound":
            solution = algorithm.solve(graph, license_types, max_iterations=50000)
        else:
            solution = algorithm.solve(graph, license_types)
        logger.info("Solution found!")
        logger.info("- Total cost: %s", solution.total_cost)
        logger.info("- Number of groups: %d", len(solution.groups))
        logger.info("- Covered nodes: %d/%d", len(solution.covered_nodes), len(graph.nodes()))
        validator = SolutionValidator()
        try:
            is_valid = validator.is_valid_solution(solution, graph)
            logger.info("- Solution is valid: %s", is_valid)
        except Exception as validation_error:
            logger.error("- Solution validation FAILED: %s", validation_error)
            logger.error("- Algorithm: %s", algorithm_name)
            logger.error("- Graph nodes: %d", len(graph.nodes()))
            logger.error("- Solution covered nodes: %d", len(solution.covered_nodes))
            all_nodes = set(graph.nodes())
            missing_nodes = all_nodes - solution.covered_nodes
            extra_nodes = solution.covered_nodes - all_nodes
            if missing_nodes:
                logger.error("- Missing nodes (%d): %s", len(missing_nodes), sorted(list(missing_nodes)))
            if extra_nodes:
                logger.error("- Extra nodes (%d): %s", len(extra_nodes), sorted(list(extra_nodes)))
            logger.error("- Number of groups: %d", len(solution.groups))
            overlapping_nodes = set()
            all_group_nodes = set()
            for i, group in enumerate(solution.groups):
                group_nodes = group.all_members
                overlap = group_nodes & all_group_nodes
                if overlap:
                    overlapping_nodes.update(overlap)
                    logger.error("- Group %d overlaps with previous groups: %s", i + 1, sorted(list(overlap)))
                all_group_nodes.update(group_nodes)
                if not (group.license_type.min_capacity <= group.size <= group.license_type.max_capacity):
                    logger.error(
                        "- Group %d violates size constraints: size=%d, min=%d, max=%d",
                        i + 1,
                        group.size,
                        group.license_type.min_capacity,
                        group.license_type.max_capacity,
                    )
                owner_neighbors = set(graph.neighbors(group.owner)) | {group.owner}
                invalid_members = group.all_members - owner_neighbors
                if invalid_members:
                    logger.error(
                        "- Group %d has invalid members %s not connected to owner %s",
                        i + 1,
                        sorted(list(invalid_members)),
                        group.owner,
                    )
                    logger.error("  Owner neighbors: %s", sorted(list(owner_neighbors)))
                    logger.error("  Group members: %s", sorted(list(group.all_members)))
            if overlapping_nodes:
                logger.error("- Total overlapping nodes: %s", sorted(list(overlapping_nodes)))
            return float("inf")
        if len(solution.groups) <= 10:
            logger.info("\nLicense groups:")
            for i, group in enumerate(solution.groups, 1):
                logger.info("  Group %d: %s", i, group.license_type.name)
                logger.info(
                    "    Owner: %s, Other Members: %s",
                    group.owner,
                    sorted(group.additional_members),
                )
        else:
            logger.info("\n(Too many groups to display - showing summary only)")
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
        logger.error("\n🚨 ALGORITHM FAILURE: %s", algorithm_name)
        logger.error("📍 Error Type: %s", type(e).__name__)
        logger.error("💬 Error Message: %s", str(e))
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_list = traceback.extract_tb(exc_traceback)
        our_frames = [frame for frame in tb_list if "graph-licensing-optimization" in frame.filename]
        if our_frames:
            last_frame = our_frames[-1]
            logger.error("📂 File: %s", last_frame.filename)
            logger.error("📏 Line: %d", last_frame.lineno)
            logger.error("🔧 Function: %s", last_frame.name)
            logger.error("💻 Code: %s", last_frame.line)
        if exc_traceback:
            frame = exc_traceback.tb_frame
            logger.error("\n🔍 Local Variables at Error:")
            for var_name, var_value in frame.f_locals.items():
                if not var_name.startswith("__"):
                    try:
                        if isinstance(var_value, (list, set, dict)) and len(str(var_value)) > 200:
                            logger.error("  %s: %s (length: %d)", var_name, type(var_value).__name__, len(var_value))
                        else:
                            logger.error("  %s: %r", var_name, var_value)
                    except Exception:
                        logger.error("  %s: <unable to display>", var_name)
        logger.error("\n📋 Full Stack Trace:")
        logger.exception("Exception during algorithm execution")
        logger.error("\n🌐 Algorithm Context:")
        logger.error("  - Graph nodes: %d", len(graph.nodes()))
        logger.error("  - Graph edges: %d", len(graph.edges()))
        logger.error("  - License types: %s", [lt.name for lt in license_types])
        logger.error("  - Algorithm class: %s", algorithm.__class__.__name__)
        return float("inf")


def main():
    parser = argparse.ArgumentParser(description="Run algorithm comparison")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Set the logging level (e.g., DEBUG, INFO, WARNING)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    logger.info("Graph Licensing Optimization - Algorithm Comparison")
    logger.info("=" * 60)
    generator = GraphGeneratorFactory.get_generator(GRAPH_TYPE)
    graph = generator(n_nodes=GRAPH_NODES, k=GRAPH_K, p=GRAPH_P, seed=GRAPH_SEED)
    logger.info("Generated small world graph:")
    logger.info("- Nodes: %d", len(graph.nodes()))
    logger.info("- Edges: %d", len(graph.edges()))
    logger.info("- Average degree: %.2f", sum(dict(graph.degree()).values()) / len(graph.nodes()))
    license_types = LicenseConfigFactory.get_config(LICENSE_CONFIG)
    logger.info("\nUsing license configuration: %s", LICENSE_CONFIG.upper())
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
    logger.info("\n%s", f"{'=' * 20} COMPARISON {'=' * 20}")
    best_cost = min(results.values())
    sorted_results = sorted(results.items(), key=lambda x: x[1] if x[1] != float("inf") else float("inf"))
    for name, cost in sorted_results:
        if cost == float("inf"):
            logger.info("%s: FAILED", name)
        else:
            gap = ((cost - best_cost) / best_cost * 100) if best_cost > 0 else 0
            logger.info("%s: cost=%.1f (gap: %.1f%%)", name, cost, gap)


if __name__ == "__main__":
    main()
