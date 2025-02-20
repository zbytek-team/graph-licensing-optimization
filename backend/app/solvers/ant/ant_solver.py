import random
from app.models.solve import License, LicenseAssignment, Assignments
from app.models.graph import Graph, Node


class Ant:
    def __init__(self, graph: Graph, licenses: list[License]):
        self.graph = graph
        self.licenses = licenses
        self.solution: Assignments = {}
        self.covered_nodes = set()
        self.cost = 0

    def select_license(self, node: Node):
        """Wybiera licencję do zakupu na danym węźle"""
        best_license = None
        best_coverage = 0
        best_covered_nodes = []

        for license in self.licenses:
            candidate_set = {node} | set(self.graph.get_neighbors(node))
            covered_nodes = list(candidate_set)[: license.limit]
            effective_coverage = len(set(covered_nodes) - self.covered_nodes)

            if effective_coverage > best_coverage:
                best_license = license
                best_coverage = effective_coverage
                best_covered_nodes = covered_nodes

        return best_license, best_covered_nodes

    def construct_solution(self):
        """Tworzy rozwiązanie, wybierając licencje dla węzłów"""
        available_nodes = set(self.graph.nodes)

        while available_nodes:
            node = random.choice(list(available_nodes))
            license, covered_nodes = self.select_license(node)

            if license and covered_nodes:
                self.covered_nodes.update(covered_nodes)
                assignment = LicenseAssignment(license_holder=node, covered_nodes=covered_nodes)

                if license.name not in self.solution:
                    self.solution[license.name] = []
                self.solution[license.name].append(assignment)

                self.cost += license.cost

            available_nodes -= self.covered_nodes


def ant_solver(graph: Graph, licenses: list[License], ants: int, iterations: int) -> Assignments:
    '''
    Rozwiązanie problemu pokrycia węzłów grafu licencjami przy użyciu algorytmu mrówkowego.

    :param graph: graf, którego węzły mają być pokryte licencjami
    :param licenses: lista dostępnych licencji
    :param ants: liczba mrówek
    :param iterations: liczba iteracji

    :return: przypisanie licencji do węzłów
    '''
    pheromones = {node: 1.0 for node in graph.nodes}
    best_solution = None
    best_cost = float("inf")

    for _ in range(iterations):
        solutions = []
        for _ in range(ants):
            ant = Ant(graph, licenses)
            ant.construct_solution()
            solutions.append(ant)

        solutions.sort(key=lambda ant: ant.cost)

        if solutions[0].cost < best_cost:
            best_solution = solutions[0].solution
            best_cost = solutions[0].cost

        for node in pheromones:
            pheromones[node] *= 0.9
        for _, assignments in solutions[0].solution.items():
            for assignment in assignments:
                pheromones[assignment.license_holder] += 1.0 / solutions[0].cost # nie wiem czy to jest najlepsze rozwiązanie ale to się zobaczy jak będziemy porównywać wyniki :)

    return best_solution if best_solution else {}
