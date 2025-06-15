"""Benchmarking utilities for algorithm evaluation."""

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    import networkx as nx

    from ..algorithms.base import BaseAlgorithm
    from ..models.license import LicenseConfig, LicenseSolution


class Benchmark:
    """Benchmarking utility for licensing optimization algorithms."""

    def __init__(self) -> None:
        """Initialize the benchmark."""
        self.results: list[dict[str, Any]] = []

    def run_single_test(
        self,
        algorithm: "BaseAlgorithm",
        graph: "nx.Graph",
        config: "LicenseConfig",
        test_name: str = "",
        **kwargs,
    ) -> dict[str, Any]:
        """Run a single test and measure performance.

        Args:
            algorithm: Algorithm to test.
            graph: Graph to solve.
            config: License configuration.
            test_name: Name identifier for the test.
            **kwargs: Additional arguments to pass to the algorithm.

        Returns:
            Dictionary with test results.
        """
        # Record graph properties
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        edge_to_node_ratio = n_edges / n_nodes if n_nodes > 0 else 0

        # Measure solving time
        start_time = time.perf_counter()
        try:
            solution = algorithm.solve(graph, config, **kwargs)
            end_time = time.perf_counter()
            runtime = end_time - start_time
            success = True
            error_msg = None
        except Exception as e:
            end_time = time.perf_counter()
            runtime = end_time - start_time
            solution = None
            success = False
            error_msg = str(e)

        # Calculate solution metrics
        if success and solution:
            total_cost = solution.calculate_cost(config)
            
            # Count licenses by type
            license_counts = {}
            total_licenses = 0
            total_members = 0
            
            for license_type, groups in solution.licenses.items():
                license_counts[f"n_{license_type}_licenses"] = len(groups)
                total_licenses += len(groups)
                
                # Count members for this license type
                type_members = sum(len(members) for members in groups.values())
                license_counts[f"n_{license_type}_members"] = type_members
                total_members += type_members
            
            avg_group_size = total_members / total_licenses if total_licenses > 0 else 0
            is_valid = solution.is_valid(graph, config)
        else:
            total_cost = float("inf")
            license_counts = {}
            total_licenses = 0
            total_members = 0
            avg_group_size = 0
            is_valid = False

        # License configuration details
        license_config_details = {}
        for license_name, license_config in config.license_types.items():
            license_config_details[f"{license_name}_price"] = license_config.price
            license_config_details[f"{license_name}_min_size"] = license_config.min_size
            license_config_details[f"{license_name}_max_size"] = license_config.max_size
        
        result = {
            "test_name": test_name,
            "algorithm": algorithm.name,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "edge_to_node_ratio": edge_to_node_ratio,
            **license_config_details,
            "runtime_seconds": runtime,
            "success": success,
            "error_msg": error_msg,
            "total_cost": total_cost,
            **license_counts,
            "total_licenses": total_licenses,
            "total_members": total_members,
            "avg_group_size": avg_group_size,
            "is_valid_solution": is_valid,
        }

        self.results.append(result)
        return result


    def run_dynamic_test(
        self,
        algorithm: "BaseAlgorithm",
        initial_graph: "nx.Graph",
        config: "LicenseConfig",
        iterations: int = 10,
        modification_prob: float = 1.0,
        test_name: str = None,
    ) -> List[Dict[str, Any]]:
        """Run dynamic test with graph modifications.

        Args:
            algorithm: Algorithm to test.
            initial_graph: Initial graph state.
            config: License configuration.
            iterations: Number of iterations to run.
            modification_prob: Probability of modifications per iteration.
            test_name: Name for test results.

        Returns:
            List of test results for each iteration.
        """
        current_graph = initial_graph.copy()
        graph_states = [current_graph.copy()]
        solutions = []
        total_runtime = 0
        success = True
        error_msg = None

        try:
            # Run algorithm on each graph state
            for i in range(iterations):
                start_time = time.perf_counter()
                
                try:
                    solution = algorithm.solve(current_graph, config)
                    end_time = time.perf_counter()
                    iteration_runtime = end_time - start_time
                    
                    solutions.append((solution, iteration_runtime))
                    total_runtime += iteration_runtime
                    
                except Exception as e:
                    solutions.append((None, 0))
                    print(f"Warning: Algorithm failed at iteration {i}: {e}")
                
                # Modify graph for next iteration (except last one)
                if i < iterations - 1:
                    current_graph = algorithm._modify_graph(current_graph, modification_prob)
                    graph_states.append(current_graph.copy())
        
        except Exception as e:
            success = False
            error_msg = str(e)
            # Fill remaining iterations with None solutions
            while len(solutions) < iterations:
                solutions.append((None, 0))
            while len(graph_states) < iterations:
                graph_states.append(current_graph.copy())

        dynamic_results = []

        for i in range(iterations):
            iteration_test_name = f"{test_name}_iter_{i}" if test_name else f"iter_{i}"
            
            # Get graph state for this iteration
            iteration_graph = graph_states[i] if i < len(graph_states) else initial_graph
            n_nodes = iteration_graph.number_of_nodes()
            n_edges = iteration_graph.number_of_edges()
            edge_to_node_ratio = n_edges / n_nodes if n_nodes > 0 else 0
            
            # Get solution for this iteration
            solution, iteration_runtime = solutions[i] if i < len(solutions) else (None, 0)

            if success and solution:
                total_cost = solution.calculate_cost(config)
                
                # Count licenses by type
                license_counts = {}
                total_licenses = 0
                total_members = 0
                
                for license_type, groups in solution.licenses.items():
                    license_counts[f"n_{license_type}_licenses"] = len(groups)
                    total_licenses += len(groups)
                    
                    # Count members for this license type
                    type_members = sum(len(members) for members in groups.values())
                    license_counts[f"n_{license_type}_members"] = type_members
                    total_members += type_members
                
                avg_group_size = total_members / total_licenses if total_licenses > 0 else 0
                is_valid = solution.is_valid(iteration_graph, config)
            else:
                total_cost = float("inf")
                license_counts = {}
                total_licenses = 0
                total_members = 0
                avg_group_size = 0
                is_valid = False

            # License configuration details
            license_config_details = {}
            for license_name, license_config in config.license_types.items():
                license_config_details[f"{license_name}_price"] = license_config.price
                license_config_details[f"{license_name}_min_size"] = license_config.min_size
                license_config_details[f"{license_name}_max_size"] = license_config.max_size
            
            result = {
                "test_name": iteration_test_name,
                "algorithm": algorithm.name,
                "iteration": i,
                "total_iterations": iterations,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "edge_to_node_ratio": edge_to_node_ratio,
                **license_config_details,
                "total_runtime_seconds": total_runtime,
                "iteration_runtime_seconds": iteration_runtime,
                "avg_runtime_per_iteration": total_runtime / iterations,
                "success": success,
                "error_msg": error_msg,
                "total_cost": total_cost,
                **license_counts,
                "total_licenses": total_licenses,
                "total_members": total_members,
                "avg_group_size": avg_group_size,
                "is_valid_solution": is_valid,
            }

            dynamic_results.append(result)

        self.results.extend(dynamic_results)
        return dynamic_results

    def run_dynamic_test_with_states(
        self,
        algorithm: "BaseAlgorithm",
        initial_graph: "nx.Graph",
        config: "LicenseConfig",
        iterations: int,
        modification_prob: float = 0.1,
        test_name: str = "",
        **kwargs,
    ) -> tuple[list[dict[str, Any]], list["nx.Graph"], list["LicenseSolution"]]:
        """Test algorithm on dynamic graph changes, returning graph states and solutions.

        Args:
            algorithm: Algorithm to test.
            initial_graph: Initial graph state.
            config: License configuration.
            iterations: Number of dynamic iterations.
            modification_prob: Probability of graph modification.
            test_name: Name identifier for the test.
            **kwargs: Additional arguments to pass to the algorithm.

        Returns:
            Tuple of (results, graph_states, solutions).
        """
        start_time = time.perf_counter()
        
        # Track graph states and solutions at each iteration
        graph_states = []
        solutions_with_timing = []
        solutions_only = []
        current_graph = initial_graph.copy()
        previous_solution = None
        success = True
        error_msg = None

        try:
            for i in range(iterations):
                # Store current graph state
                graph_states.append(current_graph.copy())
                
                # Solve current graph
                iteration_start = time.perf_counter()
                solution = algorithm.solve(
                    current_graph, 
                    config, 
                    warm_start=previous_solution,
                    **kwargs
                )
                iteration_end = time.perf_counter()
                
                solutions_with_timing.append((solution, iteration_end - iteration_start))
                solutions_only.append(solution)
                previous_solution = solution

                # Modify graph for next iteration (except last iteration)
                if i < iterations - 1:
                    current_graph = algorithm._modify_graph(current_graph, modification_prob)
                    
            end_time = time.perf_counter()
            total_runtime = end_time - start_time
            
        except Exception as e:
            end_time = time.perf_counter()
            total_runtime = end_time - start_time
            success = False
            error_msg = str(e)
            # Fill remaining iterations with None solutions
            while len(solutions_with_timing) < iterations:
                solutions_with_timing.append((None, 0))
                solutions_only.append(None)
            while len(graph_states) < iterations:
                graph_states.append(current_graph.copy())

        dynamic_results = []

        for i in range(iterations):
            iteration_test_name = f"{test_name}_iter_{i}" if test_name else f"iter_{i}"
            
            # Get graph state for this iteration
            iteration_graph = graph_states[i] if i < len(graph_states) else initial_graph
            n_nodes = iteration_graph.number_of_nodes()
            n_edges = iteration_graph.number_of_edges()
            edge_to_node_ratio = n_edges / n_nodes if n_nodes > 0 else 0
            
            # Get solution for this iteration
            solution, iteration_runtime = solutions_with_timing[i] if i < len(solutions_with_timing) else (None, 0)

            if success and solution:
                total_cost = solution.calculate_cost(config)
                
                # Count licenses by type
                license_counts = {}
                total_licenses = 0
                total_members = 0
                
                for license_type, groups in solution.licenses.items():
                    license_counts[f"n_{license_type}_licenses"] = len(groups)
                    total_licenses += len(groups)
                    
                    # Count members for this license type
                    type_members = sum(len(members) for members in groups.values())
                    license_counts[f"n_{license_type}_members"] = type_members
                    total_members += type_members
                
                avg_group_size = total_members / total_licenses if total_licenses > 0 else 0
                is_valid = solution.is_valid(iteration_graph, config)
            else:
                total_cost = float("inf")
                license_counts = {}
                total_licenses = 0
                total_members = 0
                avg_group_size = 0
                is_valid = False

            # License configuration details
            license_config_details = {}
            for license_name, license_config in config.license_types.items():
                license_config_details[f"{license_name}_price"] = license_config.price
                license_config_details[f"{license_name}_min_size"] = license_config.min_size
                license_config_details[f"{license_name}_max_size"] = license_config.max_size

            result = {
                "test_name": iteration_test_name,
                "algorithm": algorithm.name,
                "iteration": i,
                "total_iterations": iterations,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "edge_to_node_ratio": edge_to_node_ratio,
                **license_config_details,
                "total_runtime_seconds": total_runtime,
                "iteration_runtime_seconds": iteration_runtime,
                "avg_runtime_per_iteration": total_runtime / iterations,
                "success": success,
                "error_msg": error_msg,
                "total_cost": total_cost,
                **license_counts,
                "total_licenses": total_licenses,
                "total_members": total_members,
                "avg_group_size": avg_group_size,
                "is_valid_solution": is_valid,
            }

            dynamic_results.append(result)

        self.results.extend(dynamic_results)
        return dynamic_results, graph_states, solutions_only

    def get_summary_statistics(self) -> dict[str, Any]:
        """Get summary statistics for all benchmark results.

        Returns:
            Dictionary with summary statistics.
        """
        if not self.results:
            return {}

        import pandas as pd

        df = pd.DataFrame(self.results)

        # Filter successful runs only
        successful_df = df[df["success"] == True]  # noqa: E712

        if successful_df.empty:
            return {"total_tests": len(self.results), "successful_tests": 0}

        summary = {
            "total_tests": len(self.results),
            "successful_tests": len(successful_df),
            "success_rate": len(successful_df) / len(self.results),
            "algorithms_tested": successful_df["algorithm"].nunique(),
            "avg_runtime": successful_df["runtime_seconds"].mean(),
            "min_runtime": successful_df["runtime_seconds"].min(),
            "max_runtime": successful_df["runtime_seconds"].max(),
            "avg_cost": successful_df["total_cost"].mean(),
            "min_cost": successful_df["total_cost"].min(),
            "max_cost": successful_df["total_cost"].max(),
        }

        # Per-algorithm statistics
        for algorithm in successful_df["algorithm"].unique():
            alg_df = successful_df[successful_df["algorithm"] == algorithm]
            summary[f"{algorithm}_avg_runtime"] = alg_df["runtime_seconds"].mean()
            summary[f"{algorithm}_avg_cost"] = alg_df["total_cost"].mean()
            summary[f"{algorithm}_success_rate"] = len(alg_df) / len(
                df[df["algorithm"] == algorithm],
            )

        return summary
