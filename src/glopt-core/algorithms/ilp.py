from collections.abc import Sequence
from typing import Any

import networkx as nx
import pulp

from glopt.core import Algorithm, LicenseGroup, LicenseType, Solution

VAR_TRUE_THRESHOLD = 0.5  # Threshold for interpreting binary variable values as True


class ILPSolver(Algorithm):
    @property
    def name(self) -> str:
        return "ilp"

    def solve(
        self,
        graph: nx.Graph,
        license_types: Sequence[LicenseType],
        **kwargs: Any,
    ) -> Solution:
        time_limit: int | None = kwargs.get("time_limit")

        nodes: list[Any] = list(graph.nodes())
        # Create the ILP minimization model
        model = pulp.LpProblem("graph_licensing_optimization", pulp.LpMinimize)

        # Decision variables: assign_vars[(i, j, t_idx)] == 1 if node j is assigned to group owned by i with license t_idx
        assign_vars: dict[tuple[Any, Any, int], pulp.LpVariable] = {}
        for i in nodes:
            neighborhood_i: set[Any] = set(graph.neighbors(i)) | {i}
            for j in neighborhood_i:
                for t_idx, _lt in enumerate(license_types):
                    assign_vars[i, j, t_idx] = pulp.LpVariable(f"x_{i}_{j}_{t_idx}", cat="Binary")

        # Decision variables: active_vars[(i, t_idx)] == 1 if group owned by i with license t_idx is active
        active_vars: dict[tuple[Any, int], pulp.LpVariable] = {}
        for i in nodes:
            for t_idx, _lt in enumerate(license_types):
                active_vars[i, t_idx] = pulp.LpVariable(f"group_active_{i}_{t_idx}", cat="Binary")

        # Objective: minimize total cost of active groups
        model += pulp.lpSum(active_vars[i, t_idx] * lt.cost for i in nodes for t_idx, lt in enumerate(license_types))

        # Constraint: each owner can activate at most one license type
        for i in nodes:
            model += pulp.lpSum(active_vars[i, t_idx] for t_idx in range(len(license_types))) <= 1

        # Constraint: each node must be assigned to exactly one group
        for j in nodes:
            neighborhood_j: set[Any] = set(graph.neighbors(j)) | {j}
            model += pulp.lpSum(assign_vars.get((i, j, t_idx), 0) for i in neighborhood_j for t_idx in range(len(license_types))) == 1

        # Constraints: group size must be within license capacity bounds if group is active
        for i in nodes:
            neighborhood_i = set(graph.neighbors(i)) | {i}
            for t_idx, lt in enumerate(license_types):
                group_size = pulp.lpSum(assign_vars.get((i, j, t_idx), 0) for j in neighborhood_i)
                # Upper bound: group size <= max_capacity * active
                model += group_size <= active_vars[i, t_idx] * lt.max_capacity
                # Lower bound: group size >= min_capacity * active
                model += group_size >= active_vars[i, t_idx] * lt.min_capacity

        # Constraint: if group is active, owner must be assigned to their own group
        for i in nodes:
            for t_idx, _lt in enumerate(license_types):
                var: Any | None = assign_vars.get((i, i, t_idx))
                if var is not None:
                    model += var >= active_vars[i, t_idx]

        # Solve the ILP model with optional time limit
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit) if time_limit else pulp.PULP_CBC_CMD(msg=False)
        model.solve(solver)

        # Expose diagnostics for logging
        try:
            self.last_status = pulp.LpStatus[model.status]
            obj_val = pulp.value(model.objective) if model.objective is not None else None
            if isinstance(obj_val, (int, float)):
                self.last_objective = float(obj_val)
            else:
                self.last_objective = float("nan")
        except Exception:
            self.last_status = "UNKNOWN"
            self.last_objective = float("nan")

        # Extract solution: build LicenseGroup objects for each active group
        groups: list[LicenseGroup] = []
        for i in nodes:
            for t_idx, lt in enumerate(license_types):
                val = float(active_vars[i, t_idx].varValue or 0.0)
                if val > VAR_TRUE_THRESHOLD:
                    members: set[Any] = set()
                    for j in set(graph.neighbors(i)) | {i}:
                        var = assign_vars.get((i, j, t_idx))
                        if var and float(var.varValue or 0.0) > VAR_TRUE_THRESHOLD:
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
