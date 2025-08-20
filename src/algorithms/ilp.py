from typing import Any, Dict, List, Set, Tuple
import networkx as nx
import pulp

from src.core import Algorithm, LicenseGroup, LicenseType, Solution


class ILPSolver(Algorithm):
    @property
    def name(self) -> str:
        return "ilp"

    def solve(
        self,
        graph: nx.Graph,
        license_types: List[LicenseType],
        **kwargs: Any,
    ) -> Solution:
        time_limit: int | None = kwargs.get("time_limit")

        nodes: List[Any] = list(graph.nodes())
        model = pulp.LpProblem("graph_licensing_optimization", pulp.LpMinimize)

        # x[i,j,t] = 1 if j joins group owned by i with license t
        assign_vars: Dict[Tuple[Any, Any, int], pulp.LpVariable] = {}
        for i in nodes:
            neighborhood_i: Set[Any] = set(graph.neighbors(i)) | {i}
            for j in neighborhood_i:
                for t_idx, _lt in enumerate(license_types):
                    assign_vars[i, j, t_idx] = pulp.LpVariable(f"x_{i}_{j}_{t_idx}", cat="Binary")

        # active[i,t] = 1 if group owned by i with license t is active
        active_vars: Dict[Tuple[Any, int], pulp.LpVariable] = {}
        for i in nodes:
            for t_idx, _lt in enumerate(license_types):
                active_vars[i, t_idx] = pulp.LpVariable(f"group_active_{i}_{t_idx}", cat="Binary")

        # objective
        model += pulp.lpSum(active_vars[i, t_idx] * lt.cost for i in nodes for t_idx, lt in enumerate(license_types))

        # each node belongs to exactly one group among its closed neighborhood
        for j in nodes:
            neighborhood_j: Set[Any] = set(graph.neighbors(j)) | {j}
            model += pulp.lpSum(assign_vars.get((i, j, t_idx), 0) for i in neighborhood_j for t_idx in range(len(license_types))) == 1

        # capacity constraints per owner and license type
        for i in nodes:
            neighborhood_i = set(graph.neighbors(i)) | {i}
            for t_idx, lt in enumerate(license_types):
                group_size = pulp.lpSum(assign_vars.get((i, j, t_idx), 0) for j in neighborhood_i)
                model += group_size <= active_vars[i, t_idx] * lt.max_capacity
                model += group_size >= active_vars[i, t_idx] * lt.min_capacity

        # if group active then owner must be in it
        for i in nodes:
            for t_idx, _lt in enumerate(license_types):
                var = assign_vars.get((i, i, t_idx))
                if var is not None:
                    model += var >= active_vars[i, t_idx]

        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit) if time_limit else pulp.PULP_CBC_CMD(msg=0)
        model.solve(solver)

        if model.status != pulp.LpStatusOptimal:
            raise RuntimeError(f"ilp solver failed with status {pulp.LpStatus[model.status]}")

        # build groups
        groups: List[LicenseGroup] = []
        for i in nodes:
            for t_idx, lt in enumerate(license_types):
                if active_vars[i, t_idx].varValue and active_vars[i, t_idx].varValue > 0.5:
                    members: Set[Any] = set()
                    for j in set(graph.neighbors(i)) | {i}:
                        var = assign_vars.get((i, j, t_idx))
                        if var and var.varValue and var.varValue > 0.5:
                            members.add(j)
                    if members:
                        groups.append(
                            LicenseGroup(
                                license_type=lt,
                                owner=i,
                                additional_members=frozenset(members - {i}),
                            )
                        )

        # Solution now computes cost and coverage itself
        return Solution(groups=tuple(groups))
