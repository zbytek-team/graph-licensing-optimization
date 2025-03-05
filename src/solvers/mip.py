import networkx as nx
from ortools.linear_solver import pywraplp

from src.solvers.base import AssignmentResult, BaseStaticSolver


class MIPSolver(BaseStaticSolver):
    def _solve(self, graph: nx.Graph) -> AssignmentResult:
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            raise RuntimeError("Failed to initialize OR-Tools solver.")

        nodes: list[int] = list(graph.nodes)
        n = len(nodes)
        neighbors: dict[int, list[int]] = {node: list(graph.neighbors(node)) for node in nodes}

        # Decision Variables:
        # x[i]: node i gets an individual license.
        x = [solver.BoolVar(f"x_{i}") for i in range(n)]
        # y[i]: node i is chosen as a license holder for a group.
        y = [solver.BoolVar(f"y_{i}") for i in range(n)]
        # z[i][j]:node i is assigned to the group license with license holder j.
        z = [[solver.BoolVar(f"z_{i}_{j}") for j in range(n)] for i in range(n)]

        # Objective Function:
        solver.Minimize(
            sum(x[i] * self.individual_cost for i in range(n)) + sum(y[i] * self.group_cost for i in range(n))
        )

        # Constraint 1: Each node must be covered exactly once.
        # A node is either assigned an individual license or included in one group (via z variables).
        for i in range(n):
            solver.Add(x[i] + sum(z[i][j] for j in range(n)) == 1)

        # Constraint 2: A node can only be assigned to a group if that node is a license holder.
        # That is, z[i][j] can be 1 only if y[j] is 1.
        for i in range(n):
            for j in range(n):
                solver.Add(z[i][j] <= y[j])

        # Constraint 3: Ensure that a license holder is assigned to its own group.
        # For each node j, if y[j] is 1 then z[j][j] must be 1.
        for j in range(n):
            solver.Add(z[j][j] == y[j])

        # Constraint 4: Enforce minimum and maximum group sizes.
        # A license holder must cover at least 2 nodes (including itself) and at most 'self.group_size' nodes.
        for j in range(n):
            solver.Add(sum(z[i][j] for i in range(n)) >= 2 * y[j])
            solver.Add(sum(z[i][j] for i in range(n)) <= self.group_size * y[j])

        # Constraint 5: Only assign nodes to groups where they are directly connected to the license holder.
        for i in range(n):
            for j in range(n):
                if nodes[i] not in neighbors[nodes[j]] and i != j:
                    solver.Add(z[i][j] == 0)

        status = solver.Solve()
        if status != pywraplp.Solver.OPTIMAL and status != pywraplp.Solver.FEASIBLE:
            raise RuntimeError(f"OR-Tools MIP solver failed with status: {status}")

        result: AssignmentResult = {"individual": set(), "group": {}}
        for i in range(n):
            if x[i].solution_value() >= 0.99:
                result["individual"].add(nodes[i])
            else:
                for j in range(n):
                    if z[i][j].solution_value() >= 0.99:
                        if nodes[j] not in result["group"]:
                            result["group"][nodes[j]] = set()
                        result["group"][nodes[j]].add(nodes[i])

        return result
