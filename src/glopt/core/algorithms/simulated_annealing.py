import math
import random
from collections.abc import Sequence
from typing import Any

import networkx as nx

from glopt.core import Algorithm, LicenseGroup, LicenseType, Solution
from glopt.core.solution_builder import SolutionBuilder
from glopt.core.solution_validator import SolutionValidator

from .greedy import GreedyAlgorithm


class SimulatedAnnealing(Algorithm):
    @property
    def name(self) -> str:
        return "simulated_annealing"

    def __init__(
        self,
        temp_initial: float = 100.0,
        cooling_rate: float = 0.995,
        temp_min: float = 0.001,
        max_iterations: int = 20000,
        max_stall: int = 2000,
    ) -> None:
        self.temp_initial = temp_initial
        self.cooling_rate = cooling_rate
        self.temp_min = temp_min
        self.max_iterations = max_iterations
        self.max_stall = max_stall
        self.validator = SolutionValidator(debug=False)

    def solve(
        self,
        graph: nx.Graph,
        license_types: Sequence[LicenseType],
        **kwargs: Any,
    ) -> Solution:
        seed = kwargs.get("seed")
        if isinstance(seed, int):
            random.seed(seed)
        deadline = kwargs.get("deadline")
        initial: Solution | None = kwargs.get("initial_solution")
        max_iterations = int(kwargs.get("max_iterations", self.max_iterations))
        max_stall = int(kwargs.get("max_stall", self.max_stall))
        if initial is not None and self.validator.is_valid_solution(initial, graph):
            current = initial
        else:
            current = GreedyAlgorithm().solve(graph, license_types)
        ok, _ = self.validator.validate(current, graph)
        if not ok:
            current = self._fallback_singletons(graph, license_types)
        best = current
        temperature = self.temp_initial
        stall = 0
        from time import perf_counter as _pc

        for _ in range(max_iterations):
            if deadline is not None and _pc() >= float(deadline):
                break
            if self.temp_min > temperature:
                break
            neighbor = self._neighbor(current, graph, license_types)
            if neighbor is None:
                stall += 1
            else:
                d = neighbor.total_cost - current.total_cost
                if d < 0 or random.random() < math.exp(-d / max(temperature, 1e-12)):
                    current = neighbor
                    if current.total_cost < best.total_cost:
                        best = current
                        stall = 0
                    else:
                        stall += 1
                else:
                    stall += 1
            if stall >= max_stall:
                stall = 0
                temperature = max(self.temp_min, temperature * 0.5)
            temperature *= self.cooling_rate
        return best

    def _fallback_singletons(self, graph: nx.Graph, lts: Sequence[LicenseType]) -> Solution:
        lt1 = SolutionBuilder.find_cheapest_single_license(lts)
        groups = [LicenseGroup(lt1, n, frozenset()) for n in graph.nodes()]
        return Solution(groups=tuple(groups))

    def _neighbor(
        self,
        solution: Solution,
        graph: nx.Graph,
        lts: Sequence[LicenseType],
    ) -> Solution | None:
        moves = [
            self._mv_change_license,
            self._mv_move_member,
            self._mv_swap_members,
            self._mv_merge_groups,
            self._mv_split_group,
        ]
        for _ in range(12):
            mv = random.choice(moves)
            try:
                cand = mv(solution, graph, lts)
            except Exception:
                cand = None
            if cand:
                ok, _ = self.validator.validate(cand, graph)
                if ok:
                    return cand
        return None

    def _mv_change_license(
        self,
        solution: Solution,
        graph: nx.Graph,
        lts: Sequence[LicenseType],
    ) -> Solution | None:
        if not solution.groups:
            return None
        g = random.choice(solution.groups)
        compat = SolutionBuilder.get_compatible_license_types(g.size, lts, exclude=g.license_type)
        cheaper = [lt for lt in compat if lt.cost < g.license_type.cost]
        if not cheaper:
            return None
        new_lt = random.choice(cheaper)
        new_groups = [LicenseGroup(new_lt, g.owner, g.additional_members) if x is g else x for x in solution.groups]
        return Solution(groups=tuple(new_groups))

    def _mv_move_member(
        self,
        solution: Solution,
        graph: nx.Graph,
        lts: Sequence[LicenseType],
    ) -> Solution | None:
        donors = [g for g in solution.groups if g.additional_members and g.size > g.license_type.min_capacity]
        if not donors:
            return None
        from_g = random.choice(donors)
        member = random.choice(list(from_g.additional_members))
        receivers = [g for g in solution.groups if g is not from_g and g.size < g.license_type.max_capacity]
        if not receivers:
            return None
        to_g = random.choice(receivers)
        allowed = SolutionBuilder.get_owner_neighbors_with_self(graph, to_g.owner)
        if member not in allowed:
            return None
        new_groups = []
        for g in solution.groups:
            if g is from_g:
                new_groups.append(
                    LicenseGroup(
                        g.license_type,
                        g.owner,
                        g.additional_members - {member},
                    )
                )
            elif g is to_g:
                new_groups.append(
                    LicenseGroup(
                        g.license_type,
                        g.owner,
                        g.additional_members | {member},
                    )
                )
            else:
                new_groups.append(g)
        return Solution(groups=tuple(new_groups))

    def _mv_swap_members(
        self,
        solution: Solution,
        graph: nx.Graph,
        lts: Sequence[LicenseType],
    ) -> Solution | None:
        if len(solution.groups) < 2:
            return None
        g1, g2 = random.sample(list(solution.groups), 2)
        cand1 = list(g1.all_members)
        cand2 = list(g2.all_members)
        if not cand1 or not cand2:
            return None
        n1 = random.choice(cand1)
        n2 = random.choice(cand2)
        if n1 not in SolutionBuilder.get_owner_neighbors_with_self(graph, g2.owner):
            return None
        if n2 not in SolutionBuilder.get_owner_neighbors_with_self(graph, g1.owner):
            return None
        new_groups: list[LicenseGroup] = []
        for g in solution.groups:
            if g is g1:
                mem = g.all_members - {n1} | {n2}
                owner = g.owner if g.owner in mem else n2
                new_groups.append(
                    LicenseGroup(
                        g.license_type,
                        owner,
                        frozenset(mem - {owner}),
                    )
                )
            elif g is g2:
                mem = g.all_members - {n2} | {n1}
                owner = g.owner if g.owner in mem else n1
                new_groups.append(
                    LicenseGroup(
                        g.license_type,
                        owner,
                        frozenset(mem - {owner}),
                    )
                )
            else:
                new_groups.append(g)
        return Solution(groups=tuple(new_groups))

    def _mv_merge_groups(
        self,
        solution: Solution,
        graph: nx.Graph,
        lts: Sequence[LicenseType],
    ) -> Solution | None:
        if len(solution.groups) < 2:
            return None
        g1, g2 = random.sample(list(solution.groups), 2)
        merged = SolutionBuilder.merge_groups(g1, g2, graph, lts)
        if merged is None:
            return None
        new_groups = [g for g in solution.groups if g not in (g1, g2)]
        new_groups.append(merged)
        return Solution(groups=tuple(new_groups))

    def _mv_split_group(
        self,
        solution: Solution,
        graph: nx.Graph,
        lts: Sequence[LicenseType],
    ) -> Solution | None:
        splittable = [g for g in solution.groups if g.size >= 3]
        if not splittable:
            return None
        g = random.choice(splittable)
        members = list(g.all_members)
        for _ in range(4):
            random.shuffle(members)
            cut = random.randint(1, len(members) - 1)
            part1, part2 = (members[:cut], members[cut:])
            lt1 = SolutionBuilder.find_cheapest_license_for_size(len(part1), lts)
            lt2 = SolutionBuilder.find_cheapest_license_for_size(len(part2), lts)
            if not lt1 or not lt2:
                continue
            owner1 = random.choice(part1)
            owner2 = random.choice(part2)
            neigh1 = SolutionBuilder.get_owner_neighbors_with_self(graph, owner1)
            neigh2 = SolutionBuilder.get_owner_neighbors_with_self(graph, owner2)
            if not set(part1).issubset(neigh1) or not set(part2).issubset(neigh2):
                continue
            new_groups = [x for x in solution.groups if x is not g]
            new_groups.append(LicenseGroup(lt1, owner1, frozenset(set(part1) - {owner1})))
            new_groups.append(LicenseGroup(lt2, owner2, frozenset(set(part2) - {owner2})))
            return Solution(groups=tuple(new_groups))
        return None
