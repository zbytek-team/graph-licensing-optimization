"""Simulated Annealing algorithm for licensing optimization."""

import math
import random
from typing import TYPE_CHECKING

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx

    from ...models.license import LicenseConfig, LicenseSolution


class SimulatedAnnealingAlgorithm(BaseAlgorithm):
    """Simulated Annealing algorithm for licensing optimization."""

    def __init__(
        self,
        initial_temp: float = 100.0,
        final_temp: float = 0.1,
        cooling_rate: float = 0.95,
        max_iterations: int = 1000,
        seed: int | None = None,
    ) -> None:
        """Initialize the simulated annealing algorithm.

        Args:
            initial_temp: Initial temperature.
            final_temp: Final temperature.
            cooling_rate: Temperature cooling rate.
            max_iterations: Maximum number of iterations.
            seed: Random seed for reproducibility.
        """
        super().__init__("SimulatedAnnealing")
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.seed = seed

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        **kwargs,
    ) -> "LicenseSolution":
        """Solve using simulated annealing.

        Args:
            graph: The social network graph.
            config: License configuration.
            **kwargs: Additional parameters (ignored).

        Returns:
            Best licensing solution found.
        """
        from ...models.license import LicenseSolution
        from ..approx.greedy import GreedyAlgorithm

        if self.seed is not None:
            random.seed(self.seed)

        nodes = list(graph.nodes())
        if not nodes:
            return LicenseSolution(solo_nodes=[], group_owners={})

        # Start with greedy solution
        greedy = GreedyAlgorithm()
        current_solution = greedy.solve(graph, config)
        current_cost = current_solution.calculate_cost(config)

        best_solution = current_solution
        best_cost = current_cost

        temperature = self.initial_temp

        for _iteration in range(self.max_iterations):
            if temperature < self.final_temp:
                break

            # Generate neighbor solution
            neighbor_solution = self._generate_neighbor(current_solution, graph, config)
            neighbor_cost = neighbor_solution.calculate_cost(config)

            # Accept or reject neighbor
            cost_diff = neighbor_cost - current_cost

            if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
                current_solution = neighbor_solution
                current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_solution = current_solution
                    best_cost = current_cost

            # Cool down
            temperature *= self.cooling_rate

        return best_solution

    def _generate_neighbor(
        self,
        solution: "LicenseSolution",
        graph: "nx.Graph",
        config: "LicenseConfig",
    ) -> "LicenseSolution":
        """Generate a neighbor solution by making a small modification.

        Args:
            solution: Current solution.
            graph: The social network graph.
            config: License configuration.

        Returns:
            Neighbor solution.
        """
        from ...models.license import LicenseSolution

        nodes = list(graph.nodes())
        if not nodes:
            return solution

        # Choose a random modification type
        modification_type = random.choice(["change_solo", "change_group", "swap_group"])

        new_solo = solution.solo_nodes.copy()
        new_groups = {k: v.copy() for k, v in solution.group_owners.items()}

        if modification_type == "change_solo" and new_solo:
            # Convert a solo node to group member or vice versa
            node = random.choice(new_solo)
            new_solo.remove(node)

            # Try to add to an existing group or create new one
            neighbors = list(graph.neighbors(node))
            if neighbors and random.random() < 0.5:
                # Join existing group if possible
                potential_owners = [n for n in neighbors if n in new_groups]
                if potential_owners:
                    owner = random.choice(potential_owners)
                    if len(new_groups[owner]) < config.group_size:
                        new_groups[owner].append(node)
                    else:
                        new_solo.append(node)  # Can't join, stay solo
                else:
                    # Create new group
                    potential_members = [n for n in neighbors if n in new_solo]
                    if potential_members:
                        member = random.choice(potential_members)
                        new_solo.remove(member)
                        new_groups[node] = [node, member]
                    else:
                        new_solo.append(node)  # No neighbors available
            else:
                new_solo.append(node)  # Keep as solo

        elif modification_type == "change_group" and new_groups:
            # Modify an existing group
            owner = random.choice(list(new_groups.keys()))
            group = new_groups[owner]

            if len(group) > 1:
                # Remove a member
                members = [m for m in group if m != owner]
                if members:
                    member_to_remove = random.choice(members)
                    group.remove(member_to_remove)
                    new_solo.append(member_to_remove)

                    if len(group) == 1:  # Only owner left
                        del new_groups[owner]
                        new_solo.append(owner)
            else:
                # Try to add a member
                neighbors = [n for n in graph.neighbors(owner) if n in new_solo]
                if neighbors and len(group) < config.group_size:
                    new_member = random.choice(neighbors)
                    new_solo.remove(new_member)
                    group.append(new_member)

        return LicenseSolution(solo_nodes=new_solo, group_owners=new_groups)
