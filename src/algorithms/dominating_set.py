"""Moduł implementuje algorytm dominating set dla dystrybucji licencji.

Wejście zwykle obejmuje obiekt `networkx.Graph` oraz konfiguracje licencji (`LicenseType`, `LicenseGroup`).
"""

from src.core import LicenseType, Solution, Algorithm, LicenseGroup
from src.utils import SolutionBuilder
from typing import Any, List, Set, Tuple
import networkx as nx


class DominatingSetAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "dominating_set_algorithm"

    def solve(
        self, graph: nx.Graph, license_types: List[LicenseType], **kwargs: Any
    ) -> Solution:
        if len(graph.nodes()) == 0:
            return Solution(groups=[], total_cost=0.0, covered_nodes=set())

        # Krok 1: Wyznaczenie zbioru dominującego z uwzględnieniem kosztu
        dominating_set = self._find_cost_effective_dominating_set(graph, license_types)

        # Krok 2: Przydział licencji dla węzłów w zbiorze dominującym
        remaining_nodes = set(graph.nodes())
        groups = []

        # Sortujemy dominatory według stopnia (malejąco) dla lepszej efektywności
        sorted_dominators = sorted(
            dominating_set, key=lambda n: graph.degree(n), reverse=True
        )

        for dominator in sorted_dominators:
            if dominator not in remaining_nodes:
                continue

            # Znajdź najlepsze przypisanie licencji dla tego dominatora
            neighbors = set(graph.neighbors(dominator)) & remaining_nodes
            available_nodes = neighbors | {dominator}

            best_assignment = self._find_best_cost_assignment(
                dominator, available_nodes, license_types
            )

            if best_assignment:
                license_type, group_members = best_assignment
                additional_members = group_members - {dominator}
                group = LicenseGroup(license_type, dominator, additional_members)
                groups.append(group)
                remaining_nodes -= group_members

        # Krok 3: Pokrycie pozostałych węzłów
        remaining_sorted = sorted(
            remaining_nodes, key=lambda n: graph.degree(n), reverse=True
        )

        for node in remaining_sorted:
            if node not in remaining_nodes:
                continue

            neighbors = set(graph.neighbors(node)) & remaining_nodes
            available_nodes = neighbors | {node}

            best_assignment = self._find_best_cost_assignment(
                node, available_nodes, license_types
            )

            if best_assignment:
                license_type, group_members = best_assignment
                additional_members = group_members - {node}
                group = LicenseGroup(license_type, node, additional_members)
                groups.append(group)
                remaining_nodes -= group_members
            else:
                # Fallback: pojedyncza licencja
                cheapest_single = self._find_cheapest_single_license(license_types)
                group = LicenseGroup(cheapest_single, node, set())
                groups.append(group)
                remaining_nodes.remove(node)

        return SolutionBuilder.create_solution_from_groups(groups)

    def _find_cost_effective_dominating_set(
        self, graph: nx.Graph, license_types: List[LicenseType]
    ) -> Set[Any]:
        """
        Znajduje zbiór dominujący uwzględniając efektywność kosztową.
        Używa heurystyki zachłannej wybierającej węzły o najlepszym współczynniku pokrycia/koszt.
        """
        nodes = set(graph.nodes())
        uncovered = nodes.copy()
        dominating_set = set()

        while uncovered:
            best_node = None
            best_score = -1

            for node in nodes:
                if node in dominating_set:
                    continue

                # Oblicz ile niepokriytych węzłów może pokryć ten węzeł
                neighbors = set(graph.neighbors(node))
                coverage = (neighbors | {node}) & uncovered

                if len(coverage) == 0:
                    continue

                # Oblicz minimalny koszt na osobę dla tego węzła
                min_cost_per_node = self._calculate_min_cost_per_node(
                    len(coverage), license_types
                )

                # Współczynnik efektywności: pokrycie / koszt
                if min_cost_per_node > 0:
                    score = len(coverage) / min_cost_per_node
                else:
                    score = len(
                        coverage
                    )  # Jeśli koszt = 0, priorytet dla większego pokrycia

                if score > best_score:
                    best_score = score
                    best_node = node

            if best_node is None:
                # Jeśli nie można znaleźć dobrego węzła, weź pierwszy niepokryty
                best_node = next(iter(uncovered))

            dominating_set.add(best_node)
            neighbors = set(graph.neighbors(best_node))
            covered_by_node = (neighbors | {best_node}) & uncovered
            uncovered -= covered_by_node

        return dominating_set

    def _calculate_min_cost_per_node(
        self, group_size: int, license_types: List[LicenseType]
    ) -> float:
        """Oblicza minimalny koszt na osobę dla grupy danego rozmiaru."""
        min_cost = float("inf")

        for license_type in license_types:
            if license_type.min_capacity <= group_size <= license_type.max_capacity:
                cost_per_node = license_type.cost / group_size
                min_cost = min(min_cost, cost_per_node)

        return min_cost if min_cost != float("inf") else 0

    def _find_best_cost_assignment(
        self, owner: Any, available_nodes: Set[Any], license_types: List[LicenseType]
    ) -> Tuple[LicenseType, Set[Any]]:
        """
        Znajduje najlepsze przypisanie licencji dla danego właściciela i dostępnych węzłów.
        Zwraca (license_type, group_members) lub None jeśli nie można przypisać.
        """
        best_assignment = None
        best_efficiency = float("inf")

        for license_type in license_types:
            # Wypróbuj różne rozmiary grup w ramach ograniczeń licencji
            max_possible_size = min(len(available_nodes), license_type.max_capacity)

            for group_size in range(license_type.min_capacity, max_possible_size + 1):
                if group_size > len(available_nodes):
                    break

                # Wybierz węzły do grupy (właściciel + najlepsi sąsiedzi)
                group_members = self._select_best_group_members(
                    owner, available_nodes, group_size
                )

                if len(group_members) == group_size:
                    cost_per_node = license_type.cost / group_size

                    if cost_per_node < best_efficiency:
                        best_efficiency = cost_per_node
                        best_assignment = (license_type, group_members)

        return best_assignment

    def _select_best_group_members(
        self, owner: Any, available_nodes: Set[Any], target_size: int
    ) -> Set[Any]:
        """
        Wybiera najlepszych członków grupy dla danego właściciela.
        Priorytet dla węzłów o wysokim stopniu (więcej połączeń).
        """
        if target_size <= 0:
            return set()

        group_members = {owner}
        remaining_slots = target_size - 1

        if remaining_slots <= 0:
            return group_members

        # Wybierz pozostałe węzły z dostępnych (wykluczając właściciela)
        candidates = list(available_nodes - {owner})

        # Sortuj kandydatów według użyteczności (możemy użyć stopnia jako heurystyki)
        # W rzeczywistej implementacji można użyć bardziej zaawansowanych metryk
        candidates.sort(key=lambda n: len(available_nodes), reverse=True)

        # Dodaj najlepszych kandydatów
        for candidate in candidates[:remaining_slots]:
            group_members.add(candidate)

        return group_members

    def _find_cheapest_single_license(
        self, license_types: List[LicenseType]
    ) -> LicenseType:
        """Znajduje najtańszą licencję dla pojedynczego użytkownika."""
        single_licenses = [
            lt for lt in license_types if lt.min_capacity <= 1 <= lt.max_capacity
        ]

        if not single_licenses:
            # Jeśli nie ma licencji dla pojedynczych użytkowników, weź najtańszą ogólnie
            return min(license_types, key=lambda lt: lt.cost)

        return min(single_licenses, key=lambda lt: lt.cost)
