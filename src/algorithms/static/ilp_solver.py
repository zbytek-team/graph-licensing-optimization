from typing import override
import networkx as nx
import pulp
from ..base import BaseSolver, Solution


class ILPSolver(BaseSolver):
    def _round(self, val):
        v = pulp.value(val)
        if isinstance(v, float):
            return round(v)
        return v

    @override
    def solve(self, graph: nx.Graph, c_single: float, c_group: float, group_size: int) -> Solution:
        vertices: list[int] = sorted(graph.nodes())

        adj: dict[int, dict[int, int]] = {
            v: {u: 1 if u in graph[v] or u == v else 0 for u in vertices} for v in vertices
        }

        prob = pulp.LpProblem("LicenseILP", pulp.LpMinimize)

        x_indiv = {v: pulp.LpVariable(f"x_{v}", cat=pulp.LpBinary) for v in vertices}
        x_holder = {v: pulp.LpVariable(f"y_{v}", cat=pulp.LpBinary) for v in vertices}
        z = {(u, h): pulp.LpVariable(f"z_{u}_{h}", cat=pulp.LpBinary) for u in vertices for h in vertices}

        for v in vertices:
            prob += x_indiv[v] + x_holder[v] <= 1

        for u in vertices:
            prob += x_indiv[u] + pulp.lpSum(z[(u, h)] for h in vertices) == 1

        for h in vertices:
            for u in vertices:
                prob += z[(u, h)] <= x_holder[h]

        for h in vertices:
            prob += z[(h, h)] == x_holder[h]

        for h in vertices:
            for u in vertices:
                prob += z[(u, h)] <= adj[h][u]

        for h in vertices:
            prob += pulp.lpSum(z[(u, h)] for u in vertices) >= 2 * x_holder[h]
            prob += pulp.lpSum(z[(u, h)] for u in vertices) <= group_size * x_holder[h]

        prob += pulp.lpSum(c_single * x_indiv[v] + c_group * x_holder[v] for v in vertices)

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[prob.status] not in {"Optimal", "Integer Feasible"}:
            raise RuntimeError(f"ILP solver status: {pulp.LpStatus[prob.status]}")

        solution: Solution = {"singles": [], "groups": []}

        for v in vertices:
            if self._round(x_indiv[v]) == 1:
                solution["singles"].append(v)
            elif self._round(x_holder[v]) == 1:
                members = [u for u in vertices if self._round(z[(u, v)]) == 1]
                solution["groups"].append({"license_holder": v, "members": members})

        return solution
