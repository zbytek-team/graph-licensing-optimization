import networkx as nx
from mip import Model, xsum, minimize, BINARY, OptimizationStatus

from src.solvers.base import Solver, SolverResult


class MIPSolver(Solver):
    def _solve(self, graph: nx.Graph) -> SolverResult:
        m = Model()

        nodes = list(graph.nodes)
        n = len(nodes)
        neighbors = {node: list(graph.neighbors(node)) for node in nodes}

        # Create binary variables x[i] indicating if node i is individual.
        x = [m.add_var(var_type=BINARY) for _ in range(n)]
        # Create binary variables y[i] indicating if node i is a group holder.
        y = [m.add_var(var_type=BINARY) for _ in range(n)]
        # Create binary variables z[i, j] indicating if node i is in group led by j.
        z = [[m.add_var(var_type=BINARY) for _ in range(n)] for _ in range(n)]

        # Set the objective function to minimize the total cost.
        m.objective = minimize(
            xsum(x[i] * self.individual_cost for i in range(n))
            + xsum(y[i] * self.group_cost for i in range(n))  # type: ignore
        )

        # Constraint 1: Each node is either individual or in a group.
        for i in range(n):
            m += x[i] + xsum(z[i][j] for j in range(n)) == 1

        # Constraint 2: A node can be in a group only if the group holder exists.
        for i in range(n):
            for j in range(n):
                m += z[i][j] <= y[j]

        # Constraint 3: The group holder is also a member of the group.
        for j in range(n):
            m += z[j][j] == y[j]

        # Constraint 4: The group size must be between 2 and group_size.
        for j in range(n):
            m += xsum(z[i][j] for i in range(n)) >= 2 * y[j]  # type: ignore
            m += xsum(z[i][j] for i in range(n)) <= self.group_size * y[j]  # type: ignore

        # Constraint 5: A holder can only be connected to its neighbors.
        for i in range(n):
            for j in range(n):
                if nodes[i] not in neighbors[nodes[j]] and i != j:
                    m += z[i][j] == 0

        status = m.optimize()
        if (
            status != OptimizationStatus.OPTIMAL
            and status != OptimizationStatus.FEASIBLE
        ):
            raise RuntimeError(f"MIP solver failed with status: {status}")

        result: SolverResult = {"individual": set(), "group": {}}
        for i in range(n):
            if x[i].x >= 0.99:  # type: ignore
                result["individual"].add(nodes[i])
            else:
                for j in range(n):
                    if z[i][j].x >= 0.99:  # type: ignore
                        if nodes[j] not in result["group"]:
                            result["group"][nodes[j]] = set()
                        result["group"][nodes[j]].add(nodes[i])

        return result
