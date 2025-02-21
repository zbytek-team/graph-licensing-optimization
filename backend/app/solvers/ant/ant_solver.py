import random
import numpy as np
from app.models.solve import License, LicenseAssignment, Assignments, AntSolverType
from app.models.graph import Graph, Node
import concurrent.futures

class Ant:
    def __init__(
        self,
        graph: Graph,
        licenses: list[License],
        pheromones: dict[Node, np.ndarray],
        alpha: float,
        beta: float,
        solution_type: AntSolverType = AntSolverType.PATH,
    ):
        self.graph = graph
        self.licenses = licenses
        self.pheromones = pheromones
        self.alpha = alpha
        self.beta = beta
        self.solution: Assignments = {}
        self.covered_nodes = set()
        self.cost = 0
        self.solution_type = solution_type

    def choose_node(self, available_nodes: set[Node]) -> Node:
        """Losowy wybór węzła"""
        return random.choice(list(available_nodes))

    def choose_node_pheromones(self, available_nodes: set[Node]) -> Node:
        """Wybór węzła probabilistycznie na podstawie feromonów i heurystyki"""
        nodes = list(available_nodes)
        pheromone_values = np.array([self.pheromones[n].max() for n in nodes])
        heuristic_values = np.array([len(self.graph.get_neighbors(n)) for n in nodes])

        probabilities = (pheromone_values**self.alpha) * (heuristic_values**self.beta)
        probabilities /= probabilities.sum()

        return np.random.choice(nodes, p=probabilities)

    def select_license(self, node: Node) -> tuple[License, list[Node]]:
        """Wybiera licencję, biorąc pod uwagę pokrycie"""
        best_license = None
        best_score = -1
        best_covered_nodes = []

        for license in self.licenses:
            candidate_set = {node} | set(self.graph.get_neighbors(node))
            covered_nodes = list(candidate_set)[: license.limit]
            effective_coverage = len(set(covered_nodes) - self.covered_nodes)

            score = effective_coverage + self.pheromones[node].sum()

            if score > best_score:
                best_license = license
                best_score = score
                best_covered_nodes = covered_nodes

        return best_license, best_covered_nodes

    def select_license_pheromones(self, node: Node) -> tuple[License, list[Node]]:
        """Wybór licencji probabilistycznie na podstawie tablicy feromonów węzła"""

        pheromone_values = np.array(self.pheromones[node])
        uncovered_neighbors = [
            n for n in self.graph.get_neighbors(node) if n not in self.covered_nodes
        ]

        heuristic_values = np.array(
            [
                max(1, len(uncovered_neighbors[: license.limit - 1])) / license.cost
                for license in self.licenses
            ]
        )

        probabilities = (pheromone_values**self.alpha) * (heuristic_values**self.beta)
        probability_sum = probabilities.sum()

        if probability_sum == 0 or np.isnan(probability_sum):
            chosen_license = random.choice(self.licenses)
        else:
            probabilities /= probability_sum
            chosen_index = np.random.choice(len(self.licenses), p=probabilities)
            chosen_license = self.licenses[chosen_index]

        covered_nodes = [node] + uncovered_neighbors[: chosen_license.limit - 1]

        return chosen_license, covered_nodes

    def construct_solution(self) -> None:
        """Tworzenie rozwiązania poprzez wybór węzłów i licencji"""
        available_nodes = set(self.graph.nodes)

        while available_nodes:
            match self.solution_type:
                case AntSolverType.PATH:
                    node = self.choose_node_pheromones(available_nodes)
                    license, covered_nodes = self.select_license(node)
                case AntSolverType.LICENCES:
                    node = self.choose_node(available_nodes)
                    license, covered_nodes = self.select_license_pheromones(node)
                case AntSolverType.PATH_AND_LICENCES:
                    node = self.choose_node_pheromones(available_nodes)
                    license, covered_nodes = self.select_license_pheromones(node)
                case _:
                    raise ValueError("Invalid solution type")

            if license and covered_nodes:
                self.covered_nodes.update(covered_nodes)
                assignment = LicenseAssignment(
                    license_holder=node, covered_nodes=covered_nodes
                )

                if license.name not in self.solution:
                    self.solution[license.name] = []
                self.solution[license.name].append(assignment)

                self.cost += license.cost

            available_nodes -= self.covered_nodes
        
        return None

def ant_worker(graph, licenses, pheromones, alpha, beta, solution_type):
    """Funckja pomocnicza dla algorytmu mrówkowego reprezentująca proces"""
    ant = Ant(graph, licenses, pheromones, alpha, beta, solution_type)
    ant.construct_solution()
    return ant.solution, ant.cost

def ant_solver(
    graph: Graph,
    licenses: list[License],
    ants: int,
    iterations: int = 0,
    alpha: float = 1.0,
    beta: float = 8.0,
    evaporation: float = 0.75,
    stagnation_limit: int = 30,
    solution_type: AntSolverType = AntSolverType.PATH,
) -> Assignments:
    """Algorytm mrówkowy z wieloprocesowością."""

    pheromones = {node: np.ones(len(licenses)) for node in graph.nodes}
    best_solution = None
    best_cost = float("inf")
    stagnation_counter = 0
    iteration = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=ants) as executor:
        while iterations == 0 or iteration < iterations:
            futures = [
                executor.submit(ant_worker, graph, licenses, pheromones, alpha, beta, solution_type)
                for _ in range(ants)
            ]

            results = [future.result() for future in concurrent.futures.as_completed(futures)]
            results.sort(key=lambda x: x[1])

            best_ant_solution, best_ant_cost = results[0]

            if best_ant_cost < best_cost:
                best_solution = best_ant_solution
                best_cost = best_ant_cost
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            for node in pheromones:
                pheromones[node] *= evaporation

            for license, assignments in best_ant_solution.items():
                for assignment in assignments:
                    node = assignment.license_holder
                    license_index = licenses.index(
                        next(lic for lic in licenses if lic.name == license)
                    )
                    pheromones[node][license_index] += 1.0 / best_ant_cost

            iteration += 1
            if iterations == 0 and stagnation_counter >= stagnation_limit:
                break

    print(iteration)
    return best_solution if best_solution else {}