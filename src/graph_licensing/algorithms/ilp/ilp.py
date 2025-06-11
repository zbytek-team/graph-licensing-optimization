"""Integer Linear Programming algorithm for exact optimization."""

from typing import TYPE_CHECKING, Optional

import pulp

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx

    from ...models.license import LicenseConfig, LicenseSolution


class ILPAlgorithm(BaseAlgorithm):
    """Integer Linear Programming solver for exact optimization."""

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
        """Solve using Integer Linear Programming.

        Args:
            graph: The social network graph.
            config: License configuration.
            warm_start: Previous solution (ignored for exact algorithm).
            time_limit: Time limit in seconds for the solver.
            **kwargs: Additional parameters (ignored).

        Returns:
            Optimal licensing solution.
        """
        from ...models.license import LicenseSolution

        nodes = list(graph.nodes())
        n = len(nodes)

        if n == 0:
            return LicenseSolution(solo_nodes=[], group_owners={})

        # Create the optimization problem
        prob = pulp.LpProblem("LicensingOptimization", pulp.LpMinimize)

        # Decision variables
        # x_i = 1 if node i has solo license
        x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in nodes}

        # y_i = 1 if node i owns a group license
        y = {i: pulp.LpVariable(f"y_{i}", cat="Binary") for i in nodes}

        # z_ij = 1 if node j is a member of group owned by node i
        z = {}
        for i in nodes:
            for j in graph.neighbors(i):
                z[(i, j)] = pulp.LpVariable(f"z_{i}_{j}", cat="Binary")
            # Node can be member of its own group
            z[(i, i)] = pulp.LpVariable(f"z_{i}_{i}", cat="Binary")

        # Objective function: minimize total cost
        prob += config.solo_price * pulp.lpSum(x[i] for i in nodes) + config.group_price * pulp.lpSum(
            y[i] for i in nodes
        )

        # Constraints
        # Each node must have exactly one license type
        for i in nodes:
            member_sum = pulp.lpSum(z.get((j, i), 0) for j in nodes if (j, i) in z)
            prob += x[i] + member_sum == 1

        # Group size constraints
        for i in nodes:
            group_members = pulp.lpSum(z.get((i, j), 0) for j in nodes if (i, j) in z)
            prob += group_members <= config.group_size * y[i]
            prob += group_members >= y[i]  # If y[i] = 1, group must have at least owner

        # Group membership implies group ownership
        for i in nodes:
            if (i, i) in z:
                prob += z[(i, i)] == y[i]

        # Can only be member of adjacent nodes' groups (or own group)
        for i in nodes:
            for j in nodes:
                if (j, i) in z and i != j and not graph.has_edge(i, j):
                    prob += z[(j, i)] == 0

        # Solve the problem
        if time_limit:
            prob.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0))
        else:
            prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # Extract solution
        solo_nodes = []
        group_owners = {}

        for i in nodes:
            if x[i].value() and x[i].value() > 0.5:
                solo_nodes.append(i)
            elif y[i].value() and y[i].value() > 0.5:
                members = []
                for j in nodes:
                    if (i, j) in z and z[(i, j)].value() and z[(i, j)].value() > 0.5:
                        members.append(j)
                if members:
                    group_owners[i] = members

        return LicenseSolution(solo_nodes=solo_nodes, group_owners=group_owners)
