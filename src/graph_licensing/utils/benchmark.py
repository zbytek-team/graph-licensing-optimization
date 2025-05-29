"""Benchmarking utilities for algorithm evaluation."""

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import networkx as nx

    from ..algorithms.base import BaseAlgorithm
    from ..models.license import LicenseConfig


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
            n_solo = len(solution.solo_nodes)
            n_groups = len(solution.group_owners)
            total_group_members = sum(len(members) for members in solution.group_owners.values())
            avg_group_size = total_group_members / n_groups if n_groups > 0 else 0
            is_valid = solution.is_valid(graph, config)
        else:
            total_cost = float("inf")
            n_solo = 0
            n_groups = 0
            total_group_members = 0
            avg_group_size = 0
            is_valid = False

        result = {
            "test_name": test_name,
            "algorithm": algorithm.name,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "edge_to_node_ratio": edge_to_node_ratio,
            "solo_price": config.solo_price,
            "group_price": config.group_price,
            "group_size_limit": config.group_size,
            "price_ratio": config.price_ratio,
            "runtime_seconds": runtime,
            "success": success,
            "error_msg": error_msg,
            "total_cost": total_cost,
            "n_solo_licenses": n_solo,
            "n_group_licenses": n_groups,
            "total_group_members": total_group_members,
            "avg_group_size": avg_group_size,
            "is_valid_solution": is_valid,
        }

        self.results.append(result)
        return result

    def run_comparison(
        self,
        algorithms: list["BaseAlgorithm"],
        graph: "nx.Graph",
        config: "LicenseConfig",
        test_name: str = "",
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Compare multiple algorithms on the same problem instance.

        Args:
            algorithms: List of algorithms to compare.
            graph: Graph to solve.
            config: License configuration.
            test_name: Name identifier for the test.
            **kwargs: Additional arguments to pass to algorithms.

        Returns:
            List of results for each algorithm.
        """
        comparison_results = []

        for algorithm in algorithms:
            result = self.run_single_test(algorithm, graph, config, test_name, **kwargs)
            comparison_results.append(result)

        return comparison_results

    def run_scalability_test(
        self,
        algorithm: "BaseAlgorithm",
        graph_generator: Callable[[int], "nx.Graph"],
        config: "LicenseConfig",
        sizes: list[int],
        test_name: str = "",
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Test algorithm scalability across different graph sizes.

        Args:
            algorithm: Algorithm to test.
            graph_generator: Function that generates graphs given size.
            config: License configuration.
            sizes: List of graph sizes to test.
            test_name: Name identifier for the test.
            **kwargs: Additional arguments to pass to the algorithm.

        Returns:
            List of results for each graph size.
        """
        scalability_results = []

        for size in sizes:
            graph = graph_generator(size)
            size_test_name = f"{test_name}_size_{size}" if test_name else f"size_{size}"
            result = self.run_single_test(
                algorithm,
                graph,
                config,
                size_test_name,
                **kwargs,
            )
            scalability_results.append(result)

        return scalability_results

    def run_price_sensitivity_test(
        self,
        algorithm: "BaseAlgorithm",
        graph: "nx.Graph",
        base_config: "LicenseConfig",
        price_ratios: list[float],
        test_name: str = "",
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Test algorithm behavior across different price ratios.

        Args:
            algorithm: Algorithm to test.
            graph: Graph to solve.
            base_config: Base license configuration.
            price_ratios: List of group_price/solo_price ratios to test.
            test_name: Name identifier for the test.
            **kwargs: Additional arguments to pass to the algorithm.

        Returns:
            List of results for each price ratio.
        """
        from ..models.license import LicenseConfig

        sensitivity_results = []

        for ratio in price_ratios:
            config = LicenseConfig(
                solo_price=base_config.solo_price,
                group_price=base_config.solo_price * ratio,
                group_size=base_config.group_size,
            )
            ratio_test_name = f"{test_name}_ratio_{ratio:.1f}" if test_name else f"ratio_{ratio:.1f}"
            result = self.run_single_test(
                algorithm,
                graph,
                config,
                ratio_test_name,
                **kwargs,
            )
            sensitivity_results.append(result)

        return sensitivity_results

    def run_dynamic_test(
        self,
        algorithm: "BaseAlgorithm",
        initial_graph: "nx.Graph",
        config: "LicenseConfig",
        iterations: int,
        modification_prob: float = 0.1,
        test_name: str = "",
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Test algorithm on dynamic graph changes.

        Args:
            algorithm: Algorithm to test.
            initial_graph: Initial graph state.
            config: License configuration.
            iterations: Number of dynamic iterations.
            modification_prob: Probability of graph modification.
            test_name: Name identifier for the test.
            **kwargs: Additional arguments to pass to the algorithm.

        Returns:
            List of results for each iteration.
        """
        start_time = time.perf_counter()

        try:
            solutions = algorithm.solve_dynamic(
                initial_graph,
                config,
                iterations,
                modification_prob,
                **kwargs,
            )
            end_time = time.perf_counter()
            total_runtime = end_time - start_time
            success = True
            error_msg = None
        except Exception as e:
            end_time = time.perf_counter()
            total_runtime = end_time - start_time
            solutions = []
            success = False
            error_msg = str(e)

        dynamic_results = []
        current_graph = initial_graph.copy()

        for i, solution in enumerate(solutions):
            iteration_test_name = f"{test_name}_iter_{i}" if test_name else f"iter_{i}"

            if success and solution:
                total_cost = solution.calculate_cost(config)
                n_solo = len(solution.solo_nodes)
                n_groups = len(solution.group_owners)
                total_group_members = sum(len(members) for members in solution.group_owners.values())
                avg_group_size = total_group_members / n_groups if n_groups > 0 else 0
                is_valid = solution.is_valid(current_graph, config)
            else:
                total_cost = float("inf")
                n_solo = 0
                n_groups = 0
                total_group_members = 0
                avg_group_size = 0
                is_valid = False

            result = {
                "test_name": iteration_test_name,
                "algorithm": algorithm.name,
                "iteration": i,
                "total_iterations": iterations,
                "n_nodes": current_graph.number_of_nodes(),
                "n_edges": current_graph.number_of_edges(),
                "edge_to_node_ratio": (
                    current_graph.number_of_edges() / current_graph.number_of_nodes()
                    if current_graph.number_of_nodes() > 0
                    else 0
                ),
                "solo_price": config.solo_price,
                "group_price": config.group_price,
                "group_size_limit": config.group_size,
                "price_ratio": config.price_ratio,
                "total_runtime_seconds": total_runtime,
                "avg_runtime_per_iteration": total_runtime / iterations,
                "success": success,
                "error_msg": error_msg,
                "total_cost": total_cost,
                "n_solo_licenses": n_solo,
                "n_group_licenses": n_groups,
                "total_group_members": total_group_members,
                "avg_group_size": avg_group_size,
                "is_valid_solution": is_valid,
            }

            dynamic_results.append(result)

            # Modify graph for next iteration
            if i < len(solutions) - 1:
                current_graph = algorithm._modify_graph(
                    current_graph,
                    modification_prob,
                )

        self.results.extend(dynamic_results)
        return dynamic_results

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

    def clear_results(self) -> None:
        """Clear all stored results."""
        self.results.clear()

    def save_results(self, filepath: str) -> None:
        """Save benchmark results to CSV file.

        Args:
            filepath: Path to save the CSV file.
        """
        if not self.results:
            return

        import pandas as pd

        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)

    def get_summary(self) -> dict[str, Any]:
        """Get summary of benchmark results.

        Returns:
            Dictionary with summary information.
        """
        return self.get_summary_statistics()
