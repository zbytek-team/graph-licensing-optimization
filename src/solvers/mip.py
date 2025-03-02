import networkx as nx
from ortools.linear_solver import pywraplp
from .base import StaticSolver, AssignmentResult


class MIPSolver(StaticSolver):
    def _solve(self, graph: nx.Graph) -> AssignmentResult:
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            raise RuntimeError("Failed to initialize OR-Tools solver.")

        nodes = list(graph.nodes)
        n = len(nodes)
        neighbors = {node: list(graph.neighbors(node)) for node in nodes}

        x = [solver.BoolVar(f"x_{i}") for i in range(n)]
        y = [solver.BoolVar(f"y_{i}") for i in range(n)]
        z = [[solver.BoolVar(f"z_{i}_{j}") for j in range(n)] for i in range(n)]

        solver.Minimize(
            sum(x[i] * self.individual_cost for i in range(n)) + sum(y[i] * self.group_cost for i in range(n))
        )

        for i in range(n):
            solver.Add(x[i] + sum(z[i][j] for j in range(n)) == 1)

        for i in range(n):
            for j in range(n):
                solver.Add(z[i][j] <= y[j])

        for j in range(n):
            solver.Add(z[j][j] == y[j])

        for j in range(n):
            solver.Add(sum(z[i][j] for i in range(n)) >= 2 * y[j])
            solver.Add(sum(z[i][j] for i in range(n)) <= self.group_size * y[j])

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
