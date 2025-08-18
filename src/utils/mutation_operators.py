"""Funkcje pomocnicze dla algorytmów: mutation operators.

Wejście zwykle obejmuje obiekt `networkx.Graph` oraz konfiguracje licencji (`LicenseType`, `LicenseGroup`).
"""

import random
from typing import List, Optional
import networkx as nx
from src.core import Solution, LicenseGroup, LicenseType
from .solution_builder import SolutionBuilder


class MutationOperators:
    @staticmethod
    def apply_random_mutation(
        solution: Solution, graph: nx.Graph, license_types: List[LicenseType]
    ) -> Optional[Solution]:
        operators = [
            MutationOperators.change_license_type,
            MutationOperators.reassign_member,
            MutationOperators.merge_groups,
            MutationOperators.split_group,
        ]
        weights = [0.3, 0.3, 0.2, 0.2]
        for _ in range(10):
            operator = random.choices(operators, weights=weights, k=1)[0]
            try:
                mutated = operator(solution, graph, license_types)
                if mutated:
                    return mutated
            except Exception:
                continue
        return None

    @staticmethod
    def change_license_type(
        solution: Solution, graph: nx.Graph, license_types: List[LicenseType]
    ) -> Optional[Solution]:
        if not solution.groups:
            return None
        group = random.choice(solution.groups)
        compatible_licenses = SolutionBuilder.get_compatible_license_types(
            group.size, license_types, exclude=group.license_type
        )
        if not compatible_licenses:
            return None
        new_license = random.choice(compatible_licenses)
        new_groups = []
        for g in solution.groups:
            if g == group:
                new_group = LicenseGroup(new_license, g.owner, g.additional_members)
                new_groups.append(new_group)
            else:
                new_groups.append(g)
        return SolutionBuilder.create_solution_from_groups(new_groups)

    @staticmethod
    def reassign_member(
        solution: Solution, graph: nx.Graph, license_types: List[LicenseType]
    ) -> Optional[Solution]:
        if len(solution.groups) < 2:
            return None
        can_lose = [
            g
            for g in solution.groups
            if g.size > g.license_type.min_capacity and g.additional_members
        ]
        can_gain = [g for g in solution.groups if g.size < g.license_type.max_capacity]
        if not can_lose or not can_gain:
            return None
        from_group = random.choice(can_lose)
        potential_to_groups = [g for g in can_gain if g != from_group]
        if not potential_to_groups:
            return None
        to_group = random.choice(potential_to_groups)
        if not from_group.additional_members:
            return None
        member = random.choice(list(from_group.additional_members))
        to_owner_neighbors = SolutionBuilder.get_owner_neighbors_with_self(
            graph, to_group.owner
        )
        if member not in to_owner_neighbors:
            return None
        new_groups = []
        for g in solution.groups:
            if g == from_group:
                new_additional = g.additional_members - {member}
                new_groups.append(LicenseGroup(g.license_type, g.owner, new_additional))
            elif g == to_group:
                new_additional = g.additional_members | {member}
                new_groups.append(LicenseGroup(g.license_type, g.owner, new_additional))
            else:
                new_groups.append(g)
        return SolutionBuilder.create_solution_from_groups(new_groups)

    @staticmethod
    def merge_groups(
        solution: Solution, graph: nx.Graph, license_types: List[LicenseType]
    ) -> Optional[Solution]:
        if len(solution.groups) < 2:
            return None
        group1, group2 = random.sample(solution.groups, 2)
        merged_group = SolutionBuilder.merge_groups(
            group1, group2, graph, license_types
        )
        if not merged_group:
            return None
        new_groups = [g for g in solution.groups if g not in [group1, group2]]
        new_groups.append(merged_group)
        return SolutionBuilder.create_solution_from_groups(new_groups)

    @staticmethod
    def split_group(
        solution: Solution, graph: nx.Graph, license_types: List[LicenseType]
    ) -> Optional[Solution]:
        if not solution.groups:
            return None
        splittable = [g for g in solution.groups if g.size > 2]
        if not splittable:
            return None
        group = random.choice(splittable)
        members = list(group.all_members)
        for _ in range(3):
            random.shuffle(members)
            split_point = random.randint(1, len(members) - 1)
            members1 = members[:split_point]
            members2 = members[split_point:]
            compatible1 = SolutionBuilder.get_compatible_license_types(
                len(members1), license_types
            )
            compatible2 = SolutionBuilder.get_compatible_license_types(
                len(members2), license_types
            )
            if not compatible1 or not compatible2:
                continue
            for lt1 in compatible1:
                for lt2 in compatible2:
                    owner1 = random.choice(members1)
                    owner1_neighbors = SolutionBuilder.get_owner_neighbors_with_self(
                        graph, owner1
                    )
                    owner2 = random.choice(members2)
                    owner2_neighbors = SolutionBuilder.get_owner_neighbors_with_self(
                        graph, owner2
                    )
                    if set(members1).issubset(owner1_neighbors) and set(
                        members2
                    ).issubset(owner2_neighbors):
                        additional1 = set(members1) - {owner1}
                        additional2 = set(members2) - {owner2}
                        group1 = LicenseGroup(lt1, owner1, additional1)
                        group2 = LicenseGroup(lt2, owner2, additional2)
                        new_groups = [g for g in solution.groups if g != group]
                        new_groups.extend([group1, group2])
                        return SolutionBuilder.create_solution_from_groups(new_groups)
        return None

    @staticmethod
    def local_search_optimization(
        solution: Solution, graph: nx.Graph, license_types: List[LicenseType]
    ) -> Solution:
        best_solution = solution
        for i, group in enumerate(solution.groups):
            compatible = SolutionBuilder.get_compatible_license_types(
                group.size, license_types, exclude=group.license_type
            )
            for license_type in compatible:
                if license_type.cost < group.license_type.cost:
                    new_groups = solution.groups.copy()
                    new_group = LicenseGroup(
                        license_type, group.owner, group.additional_members
                    )
                    new_groups[i] = new_group
                    candidate = SolutionBuilder.create_solution_from_groups(new_groups)
                    if candidate.total_cost < best_solution.total_cost:
                        best_solution = candidate
        if len(solution.groups) > 1:
            expensive_groups = [
                (i, g)
                for i, g in enumerate(solution.groups)
                if g.size <= 2 and g.license_type.cost > 20
            ]
            if len(expensive_groups) >= 2:
                (i1, g1), (i2, g2) = random.sample(expensive_groups, 2)
                merged = SolutionBuilder.merge_groups(g1, g2, graph, license_types)
                if merged:
                    new_groups = [
                        g for j, g in enumerate(solution.groups) if j not in [i1, i2]
                    ]
                    new_groups.append(merged)
                    candidate = SolutionBuilder.create_solution_from_groups(new_groups)
                    if candidate.total_cost < best_solution.total_cost:
                        best_solution = candidate
        return best_solution
