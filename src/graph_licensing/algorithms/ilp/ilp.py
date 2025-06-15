"""Flexible Integer Linear Programming solver for licensing optimization."""

from typing import TYPE_CHECKING, Optional

import pulp

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx
    from ...models.license import LicenseConfig, LicenseSolution


class ILPAlgorithm(BaseAlgorithm):
    """Integer Linear Programming solver supporting flexible license types."""

    def __init__(self) -> None:
        """Initialize the ILP algorithm."""
        super().__init__("ILP")

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None,
        time_limit: int | None = None,
        **kwargs,
    ) -> "LicenseSolution":
        """Solve using Integer Linear Programming with flexible license types.

        Args:
            graph: The social network graph.
            config: License configuration with multiple license types.
            warm_start: Previous solution for warm start (ignored).
            time_limit: Maximum solving time in seconds.
            **kwargs: Additional parameters.

        Returns:
            Optimal or near-optimal licensing solution.
        """
        from ...models.license import LicenseSolution

        nodes = list(graph.nodes())
        if not nodes:
            return LicenseSolution.create_empty()

        # Create the optimization problem
        prob = pulp.LpProblem("FlexibleLicensingOptimization", pulp.LpMinimize)

        # Decision variables for each license type and potential group
        # x[license_type][owner][members] = 1 if this assignment is chosen
        x = {}
        license_assignments = []
        
        for license_type, license_config in config.license_types.items():
            x[license_type] = {}
            
            # Generate all possible valid assignments for this license type
            for owner in nodes:
                x[license_type][owner] = {}
                
                # Try different group sizes within license constraints
                for group_size in range(license_config.min_size, license_config.max_size + 1):
                    if group_size == 1:
                        # Solo assignment
                        assignment_key = (owner,)
                        x[license_type][owner][assignment_key] = pulp.LpVariable(
                            f"x_{license_type}_{owner}_solo", cat="Binary"
                        )
                        license_assignments.append((license_type, license_config.price, owner, [owner], assignment_key))
                    
                    else:
                        # Group assignments - try all valid combinations
                        available_neighbors = [n for n in graph.neighbors(owner) if n != owner]
                        
                        if len(available_neighbors) >= group_size - 1:
                            # Generate combinations of neighbors
                            from itertools import combinations
                            for neighbor_combo in combinations(available_neighbors, group_size - 1):
                                members = [owner] + list(neighbor_combo)
                                assignment_key = tuple(sorted(members))
                                
                                var_name = f"x_{license_type}_{owner}_{'_'.join(map(str, assignment_key))}"
                                x[license_type][owner][assignment_key] = pulp.LpVariable(var_name, cat="Binary")
                                license_assignments.append((license_type, license_config.price, owner, members, assignment_key))

        # Objective function: minimize total cost
        objective_terms = []
        for license_type, price, owner, members, assignment_key in license_assignments:
            objective_terms.append(price * x[license_type][owner][assignment_key])
        
        prob += pulp.lpSum(objective_terms)

        # Constraint: Each node must be covered exactly once
        for node in nodes:
            coverage_terms = []
            for license_type, price, owner, members, assignment_key in license_assignments:
                if node in members:
                    coverage_terms.append(x[license_type][owner][assignment_key])
            
            prob += pulp.lpSum(coverage_terms) == 1

        # Additional constraint: Only one assignment per owner per license type
        for license_type in config.license_types:
            for owner in nodes:
                if owner in x[license_type]:
                    owner_assignments = []
                    for assignment_key in x[license_type][owner]:
                        owner_assignments.append(x[license_type][owner][assignment_key])
                    
                    if owner_assignments:
                        prob += pulp.lpSum(owner_assignments) <= 1

        # Solve the problem
        try:
            if time_limit:
                prob.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0))
            else:
                prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if prob.status != pulp.LpStatusOptimal:
                # Fallback to greedy solution if ILP fails
                from ..greedy import GreedyAlgorithm
                greedy_algo = GreedyAlgorithm()
                return greedy_algo.solve(graph, config)
        
        except Exception:
            # Fallback to greedy solution if ILP fails
            from ..greedy import GreedyAlgorithm
            greedy_algo = GreedyAlgorithm()
            return greedy_algo.solve(graph, config)

        # Extract solution
        licenses = {}
        
        for license_type, price, owner, members, assignment_key in license_assignments:
            var = x[license_type][owner][assignment_key]
            if var.value() and var.value() > 0.5:
                if license_type not in licenses:
                    licenses[license_type] = {}
                licenses[license_type][owner] = members
        
        return LicenseSolution(licenses=licenses)

    def supports_warm_start(self) -> bool:
        """ILP algorithm does not support warm start in this implementation."""
        return False
