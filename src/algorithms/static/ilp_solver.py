from typing import override

import networkx as nx
import pulp

from ..base import BaseSolver, Solution


class ILPSolver(BaseSolver):
    @override
    def solve(self, graph: nx.Graph, c_single: float, c_group: float, group_size: int) -> Solution:
        vertices = sorted(graph.nodes())

        adjacency_matrix = {}
        for v in vertices:
            adjacency_matrix[v] = {}
            for u in vertices:
                adjacency_matrix[v][u] = 0
            for nei in graph[v]:
                adjacency_matrix[v][nei] = 1
            adjacency_matrix[v][v] = 1

        prob = pulp.LpProblem("LicenseILP", pulp.LpMinimize)

        x1 = {}
        x2 = {}
        y = {}

        for v in vertices:
            x1[v] = pulp.LpVariable(f"x1_{v}", cat=pulp.LpBinary)
            x2[v] = pulp.LpVariable(f"x2_{v}", cat=pulp.LpBinary)
            for u in vertices:
                y[(u, v)] = pulp.LpVariable(f"y_{u}_{v}", cat=pulp.LpBinary)

        for v in vertices:
            prob += (x1[v] + x2[v] <= 1, f"unique_license_{v}")

        for u in vertices:
            prob += (
                pulp.lpSum([y[(u, v)] for v in vertices]) + x1[u] + x2[u] >= 1,
                f"coverage_{u}",
            )

        for v in vertices:
            for u in vertices:
                prob += (y[(u, v)] <= x2[v], f"only_active_{u}_{v}")

        for v in vertices:
            for u in vertices:
                prob += (y[(u, v)] <= adjacency_matrix[v][u], f"adj_{u}_{v}")

        for v in vertices:
            prob += (
                pulp.lpSum([y[(u, v)] for u in vertices]) <= group_size * x2[v],
                f"group_size_{v}",
            )

        for v in vertices:
            prob += (y[(v, v)] == x2[v], f"self_cover_{v}")

        prob += (
            pulp.lpSum([c_single * x1[v] + c_group * x2[v] for v in vertices]),
            "MinimizeCost",
        )

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[prob.status] not in [
            "Optimal",
            "Integer Feasible",
            "Optimal_infeasible",
            "Feasible",
        ]:
            raise RuntimeError(f"ILP solver failed with status {pulp.LpStatus[prob.status]}")

        licenses: Solution = {"singles": [], "groups": []}

        cost_sum = 0.0

        for v in vertices:
            xv1 = round(pulp.value(x1[v]))
            xv2 = round(pulp.value(x2[v]))
            if xv1 == 1:
                licenses["singles"].append(v)
                cost_sum += c_single
            elif xv2 == 1:
                members: list[int] = []
                for u in vertices:
                    yu = round(number=pulp.value(y[(u, v)]))
                    if yu == 1:
                        members.append(u)

                licenses["groups"].append({"license_holder": v, "members": members})
                cost_sum += c_group
            else:
                pass

        return licenses
