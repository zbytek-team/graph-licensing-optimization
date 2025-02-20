import random
import numpy as np
from app.models.solve import License, LicenseAssignment, Assignments
from app.models.graph import Graph, Node

# Tylko to na razie działa tak, że:
# 1. Mrówki rozpylają feromon na węzły, które pokrywają
# 2. Feromon wpływa TYLKO na wybór węzła ( nie na wybór licencji )
# 3. Wybór nalepiej pokrywającej licencji dla węzla


class Ant:
    def __init__(self, graph: Graph, licenses: list[License], pheromones: dict[Node, float], alpha: float, beta: float):
        self.graph = graph
        self.licenses = licenses
        self.pheromones = pheromones
        self.alpha = alpha
        self.beta = beta
        self.solution: Assignments = {}
        self.covered_nodes = set()
        self.cost = 0

    def choose_node(self, available_nodes: set[Node]) -> Node:
        """Wybiera węzeł probabilistycznie na podstawie feromonów i heurystyki"""
        nodes = list(available_nodes)
        pheromone_values = np.array([self.pheromones[n] for n in nodes])
        heuristic_values = np.array([len(self.graph.get_neighbors(n)) for n in nodes])

        probabilities = (pheromone_values ** self.alpha) * (heuristic_values ** self.beta)
        probabilities /= probabilities.sum()

        return np.random.choice(nodes, p=probabilities)

    def select_license(self, node: Node) -> tuple[License, list[Node]]:
        """Wybiera licencję, biorąc pod uwagę pokrycie oraz feromony"""
        best_license = None
        best_score = -1
        best_covered_nodes = []

        for license in self.licenses:
            candidate_set = {node} | set(self.graph.get_neighbors(node))
            covered_nodes = list(candidate_set)[: license.limit]
            effective_coverage = len(set(covered_nodes) - self.covered_nodes)

            score = effective_coverage + self.pheromones[node]

            if score > best_score:
                best_license = license
                best_score = score
                best_covered_nodes = covered_nodes

        return best_license, best_covered_nodes

    def construct_solution(self) -> None:
        """Tworzy rozwiązanie, wybierając licencje dla węzłów"""
        available_nodes = set(self.graph.nodes)

        while available_nodes:
            node = self.choose_node(available_nodes)  # Wybór węzła na podstawie feromonów
            license, covered_nodes = self.select_license(node)

            if license and covered_nodes:
                self.covered_nodes.update(covered_nodes)
                assignment = LicenseAssignment(license_holder=node, covered_nodes=covered_nodes)

                if license.name not in self.solution:
                    self.solution[license.name] = []
                self.solution[license.name].append(assignment)

                self.cost += license.cost

            available_nodes -= self.covered_nodes


def ant_solver(graph: Graph, licenses: list[License], ants: int, iterations: int, alpha: float = 1.0, beta: float = 2.0, evaporation: float = 0.9) -> Assignments:
    """
    Rozwiązanie problemu pokrycia węzłów grafu licencjami przy użyciu algorytmu mrówkowego.

    :param graph: graf, którego węzły mają być pokryte licencjami
    :param licenses: lista dostępnych licencji
    :param ants: liczba mrówek
    :param iterations: liczba iteracji
    :param alpha: wpływ feromonów na wybór
    :param beta: wpływ heurystyki (liczba sąsiadów) na wybór
    :param evaporation: stopień parowania feromonów

    :return: przypisanie licencji do węzłów
    """
    pheromones = {node: 1.0 for node in graph.nodes}
    best_solution = None
    best_cost = float("inf")

    for _ in range(iterations):
        solutions = []
        for _ in range(ants):
            ant = Ant(graph, licenses, pheromones, alpha, beta)
            ant.construct_solution()
            solutions.append(ant)

        solutions.sort(key=lambda ant: ant.cost)
        best_ant = solutions[0]

        if best_ant.cost < best_cost:
            best_solution = best_ant.solution
            best_cost = best_ant.cost

        for node in pheromones:
            pheromones[node] *= evaporation
        for _, assignments in best_ant.solution.items():
            for assignment in assignments:
                pheromones[assignment.license_holder] += 1.0 / best_ant.cost # :)

    return best_solution if best_solution else {}
