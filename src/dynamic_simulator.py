from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
import random
from dataclasses import dataclass
from src import LicenseType, Solution, LicenseGroup, Algorithm
from src.solution_builder import SolutionBuilder
from src.algorithms import GreedyAlgorithm


@dataclass
class MutationParams:
    """Parametry mutacji grafu."""

    add_nodes_prob: float = 0.1
    remove_nodes_prob: float = 0.05
    add_edges_prob: float = 0.15
    remove_edges_prob: float = 0.1
    max_nodes_add: int = 3
    max_nodes_remove: int = 2
    max_edges_add: int = 5
    max_edges_remove: int = 3


@dataclass
class DynamicStep:
    """Pojedynczy krok w symulacji dynamicznej."""

    step_number: int
    graph: nx.Graph
    solution: Solution
    mutations_applied: List[str]
    rebalance_cost_change: float


class DynamicNetworkSimulator:
    """
    Symulator scenariuszy dynamicznych dla sieci społecznościowych.

    Proces:
    1. Utworzenie grafu początkowego
    2. Przypisanie licencji
    3. Mutacje grafu (dodanie/usunięcie węzłów i krawędzi)
    4. Rebalansowanie licencji
    """

    def __init__(self, rebalance_algorithm: Optional[Algorithm] = None, mutation_params: Optional[MutationParams] = None, seed: Optional[int] = None):
        """
        Args:
            rebalance_algorithm: Algorytm do rebalansowania (domyślnie GreedyAlgorithm)
            mutation_params: Parametry mutacji grafu
            seed: Ziarno dla generatora liczb losowych
        """
        self.rebalance_algorithm = rebalance_algorithm or GreedyAlgorithm()
        self.mutation_params = mutation_params or MutationParams()
        self.seed = seed
        self.history: List[DynamicStep] = []
        self.next_node_id = 0

        if seed is not None:
            random.seed(seed)

    def simulate(
        self, initial_graph: nx.Graph, license_types: List[LicenseType], num_steps: int = 10, initial_algorithm: Optional[Algorithm] = None
    ) -> List[DynamicStep]:
        """
        Przeprowadza symulację dynamiczną.

        Args:
            initial_graph: Graf początkowy
            license_types: Dostępne typy licencji
            num_steps: Liczba kroków symulacji
            initial_algorithm: Algorytm do pierwszego przypisania

        Returns:
            Historia kroków symulacji
        """
        self.history.clear()
        current_graph = initial_graph.copy()
        self.next_node_id = max(current_graph.nodes()) + 1 if current_graph.nodes() else 0

        if initial_algorithm is None:
            initial_algorithm = self.rebalance_algorithm

        current_solution = initial_algorithm.solve(current_graph, license_types)

        self.history.append(DynamicStep(step_number=0, graph=current_graph.copy(), solution=current_solution, mutations_applied=[], rebalance_cost_change=0.0))

        for step in range(1, num_steps + 1):
            mutations_applied = []

            current_graph, step_mutations = self._apply_mutations(current_graph)
            mutations_applied.extend(step_mutations)

            old_cost = current_solution.total_cost
            current_solution = self._rebalance_licenses(current_graph, license_types, current_solution)
            new_cost = current_solution.total_cost
            cost_change = new_cost - old_cost

            self.history.append(
                DynamicStep(
                    step_number=step,
                    graph=current_graph.copy(),
                    solution=current_solution,
                    mutations_applied=mutations_applied,
                    rebalance_cost_change=cost_change,
                )
            )

        return self.history

    def _apply_mutations(self, graph: nx.Graph) -> Tuple[nx.Graph, List[str]]:
        """Zastosuj losowe mutacje do grafu."""
        mutations = []

        if random.random() < self.mutation_params.add_nodes_prob:
            num_add = random.randint(1, self.mutation_params.max_nodes_add)
            new_nodes = self._add_nodes(graph, num_add)
            mutations.append(f"Added nodes: {new_nodes}")

        if random.random() < self.mutation_params.remove_nodes_prob and len(graph.nodes()) > 5:
            num_remove = random.randint(1, min(self.mutation_params.max_nodes_remove, len(graph.nodes()) - 5))
            removed_nodes = self._remove_nodes(graph, num_remove)
            mutations.append(f"Removed nodes: {removed_nodes}")

        if random.random() < self.mutation_params.add_edges_prob:
            num_add = random.randint(1, self.mutation_params.max_edges_add)
            added_edges = self._add_edges(graph, num_add)
            mutations.append(f"Added {len(added_edges)} edges")

        if random.random() < self.mutation_params.remove_edges_prob and len(graph.edges()) > 0:
            num_remove = random.randint(1, min(self.mutation_params.max_edges_remove, len(graph.edges())))
            removed_edges = self._remove_edges(graph, num_remove)
            mutations.append(f"Removed {len(removed_edges)} edges")

        return graph, mutations

    def _add_nodes(self, graph: nx.Graph, num_nodes: int) -> List[int]:
        """Dodaj nowe węzły z losowymi połączeniami."""
        new_nodes = []
        existing_nodes = list(graph.nodes())

        for _ in range(num_nodes):
            new_node = self.next_node_id
            self.next_node_id += 1
            graph.add_node(new_node)
            new_nodes.append(new_node)

            if existing_nodes:
                num_connections = random.randint(1, min(3, len(existing_nodes)))
                neighbors = random.sample(existing_nodes, num_connections)
                for neighbor in neighbors:
                    graph.add_edge(new_node, neighbor)

        return new_nodes

    def _remove_nodes(self, graph: nx.Graph, num_nodes: int) -> List[int]:
        """Usuń losowe węzły z grafu."""
        nodes_to_remove = random.sample(list(graph.nodes()), num_nodes)

        for node in nodes_to_remove:
            graph.remove_node(node)

        return nodes_to_remove

    def _add_edges(self, graph: nx.Graph, num_edges: int) -> List[Tuple[int, int]]:
        """Dodaj losowe krawędzie między istniejącymi węzłami."""
        nodes = list(graph.nodes())
        added_edges = []

        if len(nodes) < 2:
            return added_edges

        attempts = 0
        while len(added_edges) < num_edges and attempts < num_edges * 10:
            node1, node2 = random.sample(nodes, 2)
            if not graph.has_edge(node1, node2):
                graph.add_edge(node1, node2)
                added_edges.append((node1, node2))
            attempts += 1

        return added_edges

    def _remove_edges(self, graph: nx.Graph, num_edges: int) -> List[Tuple[int, int]]:
        """Usuń losowe krawędzie z grafu."""
        edges_to_remove = random.sample(list(graph.edges()), num_edges)

        for edge in edges_to_remove:
            graph.remove_edge(*edge)

        return edges_to_remove

    def _rebalance_licenses(self, graph: nx.Graph, license_types: List[LicenseType], old_solution: Solution) -> Solution:
        """
        Rebalansowanie licencji po mutacjach grafu.

        Strategia:
        1. Sprawdź które grupy są nadal valid
        2. Zachowaj valid grupy
        3. Dla invalid grup i niepokrytych węzłów znajdź nowe przypisania
        """

        existing_nodes = set(graph.nodes())

        valid_groups = []
        uncovered_nodes = set(existing_nodes)

        for group in old_solution.groups:
            group_nodes = group.all_members

            if not group_nodes.issubset(existing_nodes):
                continue

            if self._is_group_valid(graph, group):
                valid_groups.append(group)
                uncovered_nodes -= group_nodes

        if uncovered_nodes:
            subgraph = graph.subgraph(uncovered_nodes).copy()

            if len(subgraph.nodes()) > 0:
                new_solution = self.rebalance_algorithm.solve(subgraph, license_types)
                valid_groups.extend(new_solution.groups)

        return SolutionBuilder.create_solution_from_groups(valid_groups)

    def _is_group_valid(self, graph: nx.Graph, group: LicenseGroup) -> bool:
        """Sprawdź czy grupa jest nadal valid w kontekście grafu."""
        owner = group.owner
        additional_members = group.additional_members

        if owner not in graph.nodes():
            return False

        for member in additional_members:
            if member not in graph.nodes():
                return False
            if not graph.has_edge(owner, member):
                return False

        group_size = group.size
        license_type = group.license_type

        return license_type.min_capacity <= group_size <= license_type.max_capacity

    def get_simulation_summary(self) -> Dict[str, Any]:
        """Zwróć podsumowanie symulacji."""
        if not self.history:
            return {}

        total_cost_changes = sum(step.rebalance_cost_change for step in self.history[1:])
        initial_cost = self.history[0].solution.total_cost
        final_cost = self.history[-1].solution.total_cost

        node_changes = []
        edge_changes = []

        for step in self.history[1:]:
            for mutation in step.mutations_applied:
                if "nodes" in mutation:
                    node_changes.append(mutation)
                elif "edges" in mutation:
                    edge_changes.append(mutation)

        return {
            "initial_cost": initial_cost,
            "final_cost": final_cost,
            "total_cost_change": total_cost_changes,
            "cost_efficiency": final_cost / initial_cost if initial_cost > 0 else 1.0,
            "num_steps": len(self.history) - 1,
            "node_mutations": len(node_changes),
            "edge_mutations": len(edge_changes),
            "avg_cost_change_per_step": total_cost_changes / max(1, len(self.history) - 1),
        }

    def export_history_to_csv(self, filename: str) -> None:
        """Eksportuj historię symulacji do pliku CSV."""
        import csv

        with open(filename, "w", newline="") as csvfile:
            fieldnames = ["step", "nodes", "edges", "cost", "groups", "cost_change", "mutations"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for step in self.history:
                writer.writerow(
                    {
                        "step": step.step_number,
                        "nodes": len(step.graph.nodes()),
                        "edges": len(step.graph.edges()),
                        "cost": step.solution.total_cost,
                        "groups": len(step.solution.groups),
                        "cost_change": step.rebalance_cost_change,
                        "mutations": "; ".join(step.mutations_applied),
                    }
                )


class DynamicScenarioFactory:
    """Factory do tworzenia różnych scenariuszy dynamicznych."""

    @staticmethod
    def create_growth_scenario(seed: Optional[int] = None) -> MutationParams:
        """Scenariusz wzrostu sieci - głównie dodawanie węzłów i krawędzi."""
        return MutationParams(
            add_nodes_prob=0.3,
            remove_nodes_prob=0.05,
            add_edges_prob=0.4,
            remove_edges_prob=0.1,
            max_nodes_add=5,
            max_nodes_remove=1,
            max_edges_add=8,
            max_edges_remove=2,
        )

    @staticmethod
    def create_churn_scenario(seed: Optional[int] = None) -> MutationParams:
        """Scenariusz churn - użytkownicy odchodzą i przychodzą."""
        return MutationParams(
            add_nodes_prob=0.2,
            remove_nodes_prob=0.25,
            add_edges_prob=0.2,
            remove_edges_prob=0.3,
            max_nodes_add=3,
            max_nodes_remove=4,
            max_edges_add=4,
            max_edges_remove=6,
        )

    @staticmethod
    def create_stable_scenario(seed: Optional[int] = None) -> MutationParams:
        """Scenariusz stabilny - małe zmiany."""
        return MutationParams(
            add_nodes_prob=0.1,
            remove_nodes_prob=0.05,
            add_edges_prob=0.15,
            remove_edges_prob=0.1,
            max_nodes_add=2,
            max_nodes_remove=1,
            max_edges_add=3,
            max_edges_remove=2,
        )
