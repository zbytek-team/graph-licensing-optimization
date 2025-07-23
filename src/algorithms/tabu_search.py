from ..core.types import LicenseType, Solution, Algorithm, LicenseGroup
from .greedy import GreedyAlgorithm

from typing import Any, List
import random
import networkx as nx


class TabuSearch(Algorithm):
    @property
    def name(self) -> str:
        return "tabu_search"

    def solve(self, graph: nx.Graph, license_types: List[LicenseType], **kwargs: Any) -> Solution:
        max_iterations = kwargs.get("max_iterations", 1000)
        tabu_tenure = kwargs.get("tabu_tenure", 20)

        greedy_solver = GreedyAlgorithm()
        current_solution = greedy_solver.solve(graph, license_types)
        best_solution = current_solution

        print(f"Tabu Search starting from greedy solution with cost: {current_solution.total_cost}")

        tabu_list = set()

        for _ in range(max_iterations):
            neighbors = self._generate_neighbors(current_solution, graph, license_types)

            best_neighbor = None
            best_neighbor_cost = float("inf")

            for neighbor in neighbors:
                neighbor_hash = self._solution_hash(neighbor)

                if neighbor_hash not in tabu_list or neighbor.total_cost < best_solution.total_cost:
                    if neighbor.total_cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = neighbor.total_cost

            if best_neighbor is None:
                break

            current_solution = best_neighbor

            if current_solution.total_cost < best_solution.total_cost:
                best_solution = current_solution

            tabu_list.add(self._solution_hash(current_solution))

            if len(tabu_list) > tabu_tenure:
                tabu_list.pop()

        return best_solution

    def _generate_initial_solution(self, graph: nx.Graph, license_types: List[LicenseType]) -> Solution:
        nodes = list(graph.nodes())
        uncovered = set(nodes)
        groups = []

        while uncovered:
            node = random.choice(list(uncovered))
            neighbors = set(graph.neighbors(node)) | {node}
            available = neighbors & uncovered

            license_type = random.choice(license_types)

            max_size = min(len(available), license_type.max_capacity)
            min_size = max(1, license_type.min_capacity)

            if max_size < min_size:
                license_type = min(license_types, key=lambda lt: lt.min_capacity)
                min_size = license_type.min_capacity
                max_size = min(len(available), license_type.max_capacity)

            if max_size >= min_size:
                group_size = random.randint(min_size, max_size)
                members = random.sample(list(available), group_size)

                additional_members = set(members) - {node}
                group = LicenseGroup(license_type, node, additional_members)
                groups.append(group)
                uncovered -= set(members)

        total_cost = sum(g.license_type.cost for g in groups)
        covered = set()
        for g in groups:
            covered.update(g.all_members)

        return Solution(groups, total_cost, covered)

    def _generate_neighbors(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> List[Solution]:
        neighbors = []

        neighbors.extend(self._merge_groups(solution, graph, license_types))
        neighbors.extend(self._split_groups(solution, graph, license_types))
        neighbors.extend(self._change_license_type(solution, license_types))
        neighbors.extend(self._reassign_members(solution, graph))

        return [n for n in neighbors if self._is_valid_solution(n, graph)]

    def _merge_groups(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> List[Solution]:
        neighbors = []

        for i in range(len(solution.groups)):
            for j in range(i + 1, len(solution.groups)):
                group1, group2 = solution.groups[i], solution.groups[j]

                combined_members = group1.all_members | group2.all_members

                for license_type in license_types:
                    if license_type.min_capacity <= len(combined_members) <= license_type.max_capacity:
                        for owner in [group1.owner, group2.owner]:
                            owner_neighbors = set(graph.neighbors(owner)) | {owner}
                            if combined_members.issubset(owner_neighbors):
                                new_groups = [g for k, g in enumerate(solution.groups) if k not in [i, j]]

                                additional_members = combined_members - {owner}
                                new_group = LicenseGroup(license_type, owner, additional_members)
                                new_groups.append(new_group)

                                total_cost = sum(g.license_type.cost for g in new_groups)
                                covered = set()
                                for g in new_groups:
                                    covered.update(g.all_members)

                                neighbors.append(Solution(new_groups, total_cost, covered))

        return neighbors

    def _split_groups(self, solution: Solution, graph: nx.Graph, license_types: List[LicenseType]) -> List[Solution]:
        neighbors = []

        for i, group in enumerate(solution.groups):
            if group.size <= 2:
                continue

            members_list = list(group.all_members)

            for split_size in range(1, len(members_list)):
                for _ in range(3):
                    subset1 = set(random.sample(members_list, split_size))
                    subset2 = set(members_list) - subset1

                    if group.owner not in subset1:
                        subset1, subset2 = subset2, subset1

                    valid_split = False
                    new_groups = [g for j, g in enumerate(solution.groups) if j != i]

                    for lt1 in license_types:
                        if not (lt1.min_capacity <= len(subset1) <= lt1.max_capacity):
                            continue

                        owner1_neighbors = set(graph.neighbors(group.owner)) | {group.owner}
                        if not subset1.issubset(owner1_neighbors):
                            continue

                        for member2 in subset2:
                            for lt2 in license_types:
                                if not (lt2.min_capacity <= len(subset2) <= lt2.max_capacity):
                                    continue

                                owner2_neighbors = set(graph.neighbors(member2)) | {member2}
                                if not subset2.issubset(owner2_neighbors):
                                    continue

                                group1 = LicenseGroup(lt1, group.owner, subset1 - {group.owner})
                                group2 = LicenseGroup(lt2, member2, subset2 - {member2})

                                split_groups = new_groups + [group1, group2]
                                total_cost = sum(g.license_type.cost for g in split_groups)
                                covered = set()
                                for g in split_groups:
                                    covered.update(g.all_members)

                                neighbors.append(Solution(split_groups, total_cost, covered))
                                valid_split = True
                                break
                            if valid_split:
                                break
                        if valid_split:
                            break
                    if valid_split:
                        break

        return neighbors

    def _change_license_type(self, solution: Solution, license_types: List[LicenseType]) -> List[Solution]:
        neighbors = []

        for i, group in enumerate(solution.groups):
            for license_type in license_types:
                if license_type != group.license_type and license_type.min_capacity <= group.size <= license_type.max_capacity:
                    new_groups = solution.groups.copy()
                    new_group = LicenseGroup(license_type, group.owner, group.additional_members)
                    new_groups[i] = new_group

                    total_cost = sum(g.license_type.cost for g in new_groups)
                    covered = set()
                    for g in new_groups:
                        covered.update(g.all_members)

                    neighbors.append(Solution(new_groups, total_cost, covered))

        return neighbors

    def _reassign_members(self, solution: Solution, graph: nx.Graph) -> List[Solution]:
        neighbors = []

        for i, group in enumerate(solution.groups):
            if len(group.additional_members) == 0:
                continue

            member = random.choice(list(group.additional_members))

            for j, other_group in enumerate(solution.groups):
                if i == j:
                    continue

                other_owner_neighbors = set(graph.neighbors(other_group.owner)) | {other_group.owner}
                if member in other_owner_neighbors:
                    new_group1_members = group.additional_members - {member}
                    new_group2_members = other_group.additional_members | {member}

                    new_size1 = len(new_group1_members) + 1
                    new_size2 = len(new_group2_members) + 1

                    if (
                        group.license_type.min_capacity <= new_size1 <= group.license_type.max_capacity
                        and other_group.license_type.min_capacity <= new_size2 <= other_group.license_type.max_capacity
                    ):
                        new_groups = solution.groups.copy()
                        new_groups[i] = LicenseGroup(group.license_type, group.owner, new_group1_members)
                        new_groups[j] = LicenseGroup(other_group.license_type, other_group.owner, new_group2_members)

                        total_cost = sum(g.license_type.cost for g in new_groups)
                        covered = set()
                        for g in new_groups:
                            covered.update(g.all_members)

                        neighbors.append(Solution(new_groups, total_cost, covered))

        return neighbors

    def _is_valid_solution(self, solution: Solution, graph: nx.Graph) -> bool:
        all_nodes = set(graph.nodes())
        return solution.is_valid(graph, all_nodes)

    def _solution_hash(self, solution: Solution) -> str:
        groups_repr = []
        for group in sorted(solution.groups, key=lambda g: (g.owner, g.license_type.name)):
            members_str = ",".join(map(str, sorted(group.all_members)))
            groups_repr.append(f"{group.license_type.name}:{group.owner}:{members_str}")
        return "|".join(groups_repr)
