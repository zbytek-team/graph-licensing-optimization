from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .models import LicenseGroup, LicenseType, Solution
from .solution_builder import SolutionBuilder

if TYPE_CHECKING:
    from collections.abc import Sequence

    import networkx as nx


class MutationOperators:
    @staticmethod
    def generate_neighbors(
        base: Solution,
        graph: nx.Graph,
        license_types: Sequence[LicenseType],
        k: int = 10,
    ) -> list[Solution]:
        ops = (
            MutationOperators.change_license_type,
            MutationOperators.reassign_member,
            MutationOperators.merge_groups,
            MutationOperators.split_group,
        )
        weights = (0.30, 0.30, 0.20, 0.20)
        out: list[Solution] = []
        attempts = 0
        while len(out) < k and attempts < k * 10:
            attempts += 1
            op = random.choices(ops, weights=weights, k=1)[0]
            try:
                cand = op(base, graph, list(license_types))
            except Exception:
                cand = None
            if cand is not None:
                out.append(cand)
        return out

    @staticmethod
    def change_license_type(
        solution: Solution,
        graph: nx.Graph,
        license_types: list[LicenseType],
    ) -> Solution | None:
        if not solution.groups:
            return None
        group = random.choice(solution.groups)
        compatible = SolutionBuilder.get_compatible_license_types(group.size, license_types, exclude=group.license_type)
        if not compatible:
            return None
        new_lt = random.choice(compatible)

        new_groups = []
        for g in solution.groups:
            if g is group:
                new_groups.append(LicenseGroup(new_lt, g.owner, g.additional_members))
            else:
                new_groups.append(g)
        return SolutionBuilder.create_solution_from_groups(new_groups)

    @staticmethod
    def reassign_member(
        solution: Solution,
        graph: nx.Graph,
        license_types: list[LicenseType],
    ) -> Solution | None:
        if len(solution.groups) < 2:
            return None

        donors = [g for g in solution.groups if g.size > g.license_type.min_capacity and g.additional_members]
        receivers = [g for g in solution.groups if g.size < g.license_type.max_capacity]
        if not donors or not receivers:
            return None

        from_group = random.choice(donors)
        pot_receivers = [g for g in receivers if g is not from_group]
        if not pot_receivers:
            return None
        to_group = random.choice(pot_receivers)

        member = random.choice(list(from_group.additional_members))
        allowed = SolutionBuilder.get_owner_neighbors_with_self(graph, to_group.owner)
        if member not in allowed:
            return None

        new_groups = []
        for g in solution.groups:
            if g is from_group:
                new_groups.append(LicenseGroup(g.license_type, g.owner, g.additional_members - {member}))
            elif g is to_group:
                new_groups.append(LicenseGroup(g.license_type, g.owner, g.additional_members | {member}))
            else:
                new_groups.append(g)
        return SolutionBuilder.create_solution_from_groups(new_groups)

    @staticmethod
    def merge_groups(
        solution: Solution,
        graph: nx.Graph,
        license_types: list[LicenseType],
    ) -> Solution | None:
        if len(solution.groups) < 2:
            return None
        g1, g2 = random.sample(list(solution.groups), 2)
        merged = SolutionBuilder.merge_groups(g1, g2, graph, license_types)
        if merged is None:
            return None

        new_groups = [g for g in solution.groups if g not in (g1, g2)]
        new_groups.append(merged)
        return SolutionBuilder.create_solution_from_groups(new_groups)

    @staticmethod
    def split_group(
        solution: Solution,
        graph: nx.Graph,
        license_types: list[LicenseType],
    ) -> Solution | None:
        if not solution.groups:
            return None

        splittable = [g for g in solution.groups if g.size > 2]
        if not splittable:
            return None

        group = random.choice(splittable)
        members = list(group.all_members)

        for _ in range(4):
            random.shuffle(members)
            cut = random.randint(1, len(members) - 1)
            part1, part2 = members[:cut], members[cut:]

            compat1 = SolutionBuilder.get_compatible_license_types(len(part1), license_types)
            compat2 = SolutionBuilder.get_compatible_license_types(len(part2), license_types)
            if not compat1 or not compat2:
                continue

            owner1 = random.choice(part1)
            owner2 = random.choice(part2)

            neigh1 = SolutionBuilder.get_owner_neighbors_with_self(graph, owner1)
            neigh2 = SolutionBuilder.get_owner_neighbors_with_self(graph, owner2)
            if not set(part1).issubset(neigh1) or not set(part2).issubset(neigh2):
                continue

            lt1 = random.choice(compat1)
            lt2 = random.choice(compat2)

            g1 = LicenseGroup(lt1, owner1, frozenset(set(part1) - {owner1}))
            g2 = LicenseGroup(lt2, owner2, frozenset(set(part2) - {owner2}))

            new_groups = [g for g in solution.groups if g is not group]
            new_groups.extend([g1, g2])
            return SolutionBuilder.create_solution_from_groups(new_groups)

        return None
