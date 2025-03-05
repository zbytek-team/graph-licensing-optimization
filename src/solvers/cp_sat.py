import networkx as nx
from ortools.sat.python import cp_model

from src.solvers.base import AssignmentResult, BaseStaticSolver


class CPSATSolver(BaseStaticSolver):
    def _solve(self, graph: nx.Graph) -> AssignmentResult:
        model = cp_model.CpModel()
        solver = cp_model.CpSolver()

        nodes = list(graph.nodes)
        n = len(nodes)
        neighbors = {node: list(graph.neighbors(node)) for node in nodes}

        x = [model.NewBoolVar(f"x_{i}") for i in range(n)]
        y = [model.NewBoolVar(f"y_{j}") for j in range(n)]
        z = [[model.NewBoolVar(f"z_{i}_{j}") for j in range(n)] for i in range(n)]

        model.Minimize(
            sum(x[i] * self.individual_cost for i in range(n)) + sum(y[j] * self.group_cost for j in range(n))
        )

        for i in range(n):
            model.Add(x[i] + sum(z[i][j] for j in range(n)) == 1)

        for i in range(n):
            for j in range(n):
                model.Add(z[i][j] <= y[j])

        for j in range(n):
            model.Add(z[j][j] == y[j])

        for j in range(n):
            model.Add(sum(z[i][j] for i in range(n)) >= 2 * y[j])
            model.Add(sum(z[i][j] for i in range(n)) <= self.group_size * y[j])

        for i in range(n):
            for j in range(n):
                if nodes[i] not in neighbors[nodes[j]] and i != j:
                    model.Add(z[i][j] == 0)

        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"CP-SAT solver failed with status: {status}")

        result: AssignmentResult = {"individual": set(), "group": {}}
        for i in range(n):
            if solver.Value(x[i]) == 1:
                result["individual"].add(nodes[i])
            else:
                for j in range(n):
                    if solver.Value(z[i][j]) == 1:
                        if nodes[j] not in result["group"]:
                            result["group"][nodes[j]] = set()
                        result["group"][nodes[j]].add(nodes[i])

        return result
