from src.core import LicenseType, Solution, Algorithm, LicenseGroup
from src.utils import SolutionBuilder
from typing import Any, List, Set, Tuple, Optional
import networkx as nx
import random


class RandomizedAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "randomized_algorithm"

    def __init__(self, greedy_probability: float = 0.7, seed: Optional[int] = None):
        """
        Algorytm losowy łączący strategię zachłanną z losową.

        Args:
            greedy_probability: Prawdopodobieństwo p wyboru strategii zachłannej (0.0 - 1.0)
            seed: Ziarno dla generatora liczb losowych (dla reprodukowalności)
        """
        self.greedy_probability = max(0.0, min(1.0, greedy_probability))
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def solve(self, graph: nx.Graph, license_types: List[LicenseType], **kwargs: Any) -> Solution:
        if len(graph.nodes()) == 0:
            return Solution(groups=[], total_cost=0.0, covered_nodes=set())

        # Ustawienie ziarna jeśli podane w kwargs
        runtime_seed = kwargs.get("seed", self.seed)
        if runtime_seed is not None:
            random.seed(runtime_seed)

        nodes = list(graph.nodes())
        uncovered_nodes = set(nodes)
        groups = []

        # Losowa permutacja węzłów dla eliminacji bias kolejności
        random.shuffle(nodes)

        for node in nodes:
            if node not in uncovered_nodes:
                continue

            # Decyzja: strategia zachłanna czy losowa
            use_greedy = random.random() < self.greedy_probability

            if use_greedy:
                assignment = self._greedy_assignment(node, uncovered_nodes, graph, license_types)
            else:
                assignment = self._random_assignment(node, uncovered_nodes, graph, license_types)

            if assignment:
                license_type, group_members = assignment
                additional_members = group_members - {node}
                group = LicenseGroup(license_type, node, additional_members)
                groups.append(group)
                uncovered_nodes -= group_members

        # Pokrycie pozostałych węzłów (fallback)
        while uncovered_nodes:
            node = uncovered_nodes.pop()
            cheapest_single = self._find_cheapest_single_license(license_types)
            group = LicenseGroup(cheapest_single, node, set())
            groups.append(group)

        return SolutionBuilder.create_solution_from_groups(groups)

    def _greedy_assignment(
        self, node: Any, uncovered_nodes: Set[Any], graph: nx.Graph, license_types: List[LicenseType]
    ) -> Optional[Tuple[LicenseType, Set[Any]]]:
        """
        Strategia zachłanna: wybiera przypisanie minimalizujące koszt na osobę.
        """
        neighbors = set(graph.neighbors(node)) & uncovered_nodes
        available_nodes = neighbors | {node}

        best_assignment = None
        best_efficiency = float("inf")

        for license_type in license_types:
            max_possible_size = min(len(available_nodes), license_type.max_capacity)

            for group_size in range(license_type.min_capacity, max_possible_size + 1):
                if group_size > len(available_nodes):
                    break

                # Wybierz najlepszych sąsiadów (według stopnia w grafie)
                group_members = self._select_greedy_group_members(node, available_nodes, group_size, graph)

                if len(group_members) == group_size:
                    cost_per_node = license_type.cost / group_size

                    if cost_per_node < best_efficiency:
                        best_efficiency = cost_per_node
                        best_assignment = (license_type, group_members)

        return best_assignment

    def _random_assignment(
        self, node: Any, uncovered_nodes: Set[Any], graph: nx.Graph, license_types: List[LicenseType]
    ) -> Optional[Tuple[LicenseType, Set[Any]]]:
        """
        Strategia losowa: losowy wybór typu licencji i członków grupy.
        """
        neighbors = set(graph.neighbors(node)) & uncovered_nodes
        available_nodes = neighbors | {node}

        # Losowo wybierz typ licencji
        compatible_licenses = [lt for lt in license_types if lt.min_capacity <= len(available_nodes)]

        if not compatible_licenses:
            # Fallback: najlepsze dostępne
            return self._greedy_assignment(node, uncovered_nodes, graph, license_types)

        # Przetasuj typy licencji i wybierz pierwszy pasujący
        random.shuffle(compatible_licenses)

        for license_type in compatible_licenses:
            max_possible_size = min(len(available_nodes), license_type.max_capacity)

            if max_possible_size < license_type.min_capacity:
                continue

            # Losowo wybierz rozmiar grupy w dozwolonym zakresie
            group_size = random.randint(license_type.min_capacity, max_possible_size)

            # Losowo wybierz członków grupy
            group_members = self._select_random_group_members(node, available_nodes, group_size)

            if len(group_members) >= license_type.min_capacity:
                return (license_type, group_members)

        # Fallback jeśli losowy wybór nie powiódł się
        return self._greedy_assignment(node, uncovered_nodes, graph, license_types)

    def _select_greedy_group_members(self, owner: Any, available_nodes: Set[Any], target_size: int, graph: nx.Graph) -> Set[Any]:
        """
        Wybiera członków grupy zachłannie (według stopnia w grafie).
        """
        if target_size <= 0:
            return set()

        group_members = {owner}
        remaining_slots = target_size - 1

        if remaining_slots <= 0:
            return group_members

        # Sortuj kandydatów według stopnia (malejąco)
        candidates = list(available_nodes - {owner})
        candidates.sort(key=lambda n: graph.degree(n), reverse=True)

        # Dodaj najlepszych kandydatów
        for candidate in candidates[:remaining_slots]:
            group_members.add(candidate)

        return group_members

    def _select_random_group_members(self, owner: Any, available_nodes: Set[Any], target_size: int) -> Set[Any]:
        """
        Losowo wybiera członków grupy.
        """
        if target_size <= 0:
            return set()

        group_members = {owner}
        remaining_slots = target_size - 1

        if remaining_slots <= 0:
            return group_members

        # Losowo wybierz kandydatów
        candidates = list(available_nodes - {owner})

        if len(candidates) >= remaining_slots:
            selected_candidates = random.sample(candidates, remaining_slots)
            group_members.update(selected_candidates)
        else:
            group_members.update(candidates)

        return group_members

    def _find_cheapest_single_license(self, license_types: List[LicenseType]) -> LicenseType:
        """Znajduje najtańszą licencję dla pojedynczego użytkownika."""
        single_licenses = [lt for lt in license_types if lt.min_capacity <= 1 <= lt.max_capacity]

        if not single_licenses:
            return min(license_types, key=lambda lt: lt.cost)

        return min(single_licenses, key=lambda lt: lt.cost)
