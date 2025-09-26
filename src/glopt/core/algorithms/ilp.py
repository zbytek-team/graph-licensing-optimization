from collections.abc import Sequence
from typing import Any

import networkx as nx
import pulp

from glopt.core import Algorithm, LicenseGroup, LicenseType, Solution


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
        # G = (V,E), N[i] = neighbors(i) union {i}
        nodes: list[Any] = list(graph.nodes())
        Nhood: dict[Any, set[Any]] = {i: set(graph.neighbors(i)) | {i} for i in nodes}
        degp1: dict[Any, int] = {i: len(Nhood[i]) for i in nodes}

        model = pulp.LpProblem("graph_licensing_optimization", pulp.LpMinimize)

        # active[i,t] = 1 gdy wlasciciel i otwiera grupe typu t
        active: dict[tuple[Any, int], pulp.LpVariable] = {}
        for i in nodes:
            for t_idx, lt in enumerate(license_types):
                feasible_owner_type = (lt.min_capacity <= degp1[i]) and (lt.max_capacity >= 1)
                if feasible_owner_type:
                    active[i, t_idx] = pulp.LpVariable(f"a_{i}_{t_idx}", cat="Binary")
                else:
                    # eliminacja niemozliwych par i,t
                    active[i, t_idx] = pulp.LpVariable(f"a_{i}_{t_idx}", lowBound=0, upBound=0, cat="Binary")

        # assign[i,j,t] = 1 gdy j nalezy do grupy wlasciciela i typu t
        assign: dict[tuple[Any, Any, int], pulp.LpVariable] = {}
        for i in nodes:
            for t_idx, lt in enumerate(license_types):
                if active[i, t_idx].upBound == 0:
                    continue
                if lt.max_capacity == 1:
                    # typ indywidualny, tylko wlasciciel
                    assign[i, i, t_idx] = pulp.LpVariable(f"x_{i}_{i}_{t_idx}", cat="Binary")
                else:
                    for j in Nhood[i]:
                        assign[i, j, t_idx] = pulp.LpVariable(f"x_{i}_{j}_{t_idx}", cat="Binary")

        # cel: min sum c_t * active[i,t]
        model += pulp.lpSum(active[i, t_idx] * license_types[t_idx].cost for i in nodes for t_idx in range(len(license_types)))

        # co najwyzej jedna licencja na wlasciciela
        for i in nodes:
            model += pulp.lpSum(active[i, t_idx] for t_idx in range(len(license_types))) <= 1

        # pokrycie dokladnie raz
        for j in nodes:
            model += pulp.lpSum(assign.get((i, j, t_idx), 0) for i in Nhood[j] for t_idx in range(len(license_types))) == 1

        # sprzezenie i pojemnosc
        for i in nodes:
            for t_idx, lt in enumerate(license_types):
                if active[i, t_idx].upBound == 0:
                    continue
                # wlasciciel nalezy do swojej grupy
                model += assign.get((i, i, t_idx), 0) == active[i, t_idx]
                # brak przypisan bez aktywacji
                for j in Nhood[i]:
                    var = assign.get((i, j, t_idx))
                    if var is not None:
                        model += var <= active[i, t_idx]
                # min i max pojemnosci tylko gdy aktywna
                group_size = pulp.lpSum(assign.get((i, j, t_idx), 0) for j in Nhood[i])
                model += group_size <= active[i, t_idx] * lt.max_capacity
                model += group_size >= active[i, t_idx] * lt.min_capacity

        # rozwiazanie ilp
        solver = pulp.PULP_CBC_CMD(msg=False)
        model.solve(solver)
        status = pulp.LpStatus.get(model.status, "Unknown")

        # fallback gdy brak rozwiazania dopuszczalnego
        if status in ("Infeasible", "Undefined", "Unbounded", "Not Solved"):
            singles = [lt for lt in license_types if lt.min_capacity <= 1 <= lt.max_capacity]
            if not singles:
                raise RuntimeError(f"ILP {status}: no single license available")
            lt = min(singles, key=lambda x: x.cost)
            groups = [LicenseGroup(license_type=lt, owner=i, additional_members=frozenset()) for i in nodes]
            return Solution(groups=tuple(groups))

        # ekstrakcja rozwiazania
        groups = []
        for i in nodes:
            for t_idx, lt in enumerate(license_types):
                a = active.get((i, t_idx))
                a_val = float(a.varValue) if a is not None and a.varValue is not None else 0.0
                if a_val > 0.5:
                    members: set[Any] = set()
                    for j in Nhood[i]:
                        var = assign.get((i, j, t_idx))
                        v_val = float(var.varValue) if var is not None and var.varValue is not None else 0.0
                        if v_val > 0.5:
                            members.add(j)
                    if members:
                        groups.append(
                            LicenseGroup(
                                license_type=lt,
                                owner=i,
                                additional_members=frozenset(members - {i}),
                            )
                        )

        return Solution(groups=tuple(groups))
