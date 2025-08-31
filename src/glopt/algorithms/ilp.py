from typing import Any

import networkx as nx
import pulp

from glopt.core import Algorithm, LicenseGroup, LicenseType, Solution

VAR_TRUE_THRESHOLD = 0.5


class ILPSolver(Algorithm):
    @property
    def name(self) -> str:
        return "ilp"

    def solve(
        self,
        graph: nx.Graph,
        license_types: list[LicenseType],
        **kwargs: Any,
    ) -> Solution:
        time_limit: int | None = kwargs.get("time_limit")

        nodes: list[Any] = list(graph.nodes())
        model = pulp.LpProblem("graph_licensing_optimization", pulp.LpMinimize)

        assign_vars: dict[tuple[Any, Any, int], pulp.LpVariable] = {}
        for i in nodes:
            neighborhood_i: set[Any] = set(graph.neighbors(i)) | {i}
            for j in neighborhood_i:
                for t_idx, _lt in enumerate(license_types):
                    assign_vars[i, j, t_idx] = pulp.LpVariable(f"x_{i}_{j}_{t_idx}", cat="Binary")

        active_vars: dict[tuple[Any, int], pulp.LpVariable] = {}
        for i in nodes:
            for t_idx, _lt in enumerate(license_types):
                active_vars[i, t_idx] = pulp.LpVariable(f"group_active_{i}_{t_idx}", cat="Binary")

        model += pulp.lpSum(active_vars[i, t_idx] * lt.cost for i in nodes for t_idx, lt in enumerate(license_types))

        for j in nodes:
            neighborhood_j: set[Any] = set(graph.neighbors(j)) | {j}
            model += pulp.lpSum(assign_vars.get((i, j, t_idx), 0) for i in neighborhood_j for t_idx in range(len(license_types))) == 1

        for i in nodes:
            neighborhood_i = set(graph.neighbors(i)) | {i}
            for t_idx, lt in enumerate(license_types):
                group_size = pulp.lpSum(assign_vars.get((i, j, t_idx), 0) for j in neighborhood_i)
                model += group_size <= active_vars[i, t_idx] * lt.max_capacity
                model += group_size >= active_vars[i, t_idx] * lt.min_capacity

        for i in nodes:
            for t_idx, _lt in enumerate(license_types):
                var = assign_vars.get((i, i, t_idx))
                if var is not None:
                    model += var >= active_vars[i, t_idx]

        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit) if time_limit else pulp.PULP_CBC_CMD(msg=0)
        model.solve(solver)

        if model.status != pulp.LpStatusOptimal:
            msg = f"ilp solver failed with status {pulp.LpStatus[model.status]}"
            raise RuntimeError(msg)

        groups: list[LicenseGroup] = []
        for i in nodes:
            for t_idx, lt in enumerate(license_types):
                if active_vars[i, t_idx].varValue and active_vars[i, t_idx].varValue > VAR_TRUE_THRESHOLD:
                    members: set[Any] = set()
                    for j in set(graph.neighbors(i)) | {i}:
                        var = assign_vars.get((i, j, t_idx))
                        if var and var.varValue and var.varValue > VAR_TRUE_THRESHOLD:
                            members.add(j)
                    if members:
                        groups.append(
                            LicenseGroup(
                                license_type=lt,
                                owner=i,
                                additional_members=frozenset(members - {i}),
                            ),
                        )

        return Solution(groups=tuple(groups))
