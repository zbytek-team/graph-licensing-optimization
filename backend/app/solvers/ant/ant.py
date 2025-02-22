import random
import numpy as np
from app.models.solve import License, LicenseAssignment, Assignments, AntSolverType
from app.models.graph import Graph, Node


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
        self.covered_nodes = set()
        self.solution_type = solution_type

    def choose_random_node(self, available_nodes: set[Node]) -> Node:
        return random.choice(list(available_nodes))

    def choose_node_by_pheromones(self, available_nodes: set[Node]) -> Node:
        nodes = list(available_nodes)
        pheromone_values = np.array([self.pheromones[n].max() for n in nodes])
        heuristic_values = np.array([len(self.graph.get_neighbors(n)) for n in nodes])

        probabilities = (pheromone_values**self.alpha) * (heuristic_values**self.beta)
        probabilities /= probabilities.sum()

        return np.random.choice(nodes, p=probabilities)

    def select_license_by_coverage(self, node: Node) -> tuple[License, list[Node]]:
        best_license = random.choice(self.licenses)
        best_score = -1
        best_covered_nodes = []

        for license in self.licenses:
            candidate_set = {node} | set(self.graph.get_neighbors(node))
            covered_nodes = list(candidate_set)[: license.limit]
            effective_coverage = len(set(covered_nodes) - self.covered_nodes)

            score = effective_coverage

            if score > best_score:
                best_license = license
                best_score = score
                best_covered_nodes = covered_nodes

        return best_license, best_covered_nodes

    def select_license_by_pheromones(self, node: Node) -> tuple[License, list[Node]]:
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
        probabilities /= probabilities.sum()

        chosen_index = np.random.choice(len(self.licenses), p=probabilities)
        chosen_license = self.licenses[chosen_index]

        covered_nodes = [node] + uncovered_neighbors[: chosen_license.limit - 1]

        return chosen_license, covered_nodes

    def run(self) -> tuple[Assignments, float]:
        available_nodes = set(self.graph.nodes)

        match self.solution_type:
            case AntSolverType.LICENSES:
                choose_node_fn = self.choose_random_node
                select_license_fn = self.select_license_by_pheromones
            case AntSolverType.PATH:
                choose_node_fn = self.choose_node_by_pheromones
                select_license_fn = self.select_license_by_coverage
            case AntSolverType.PATH_AND_LICENSES:
                choose_node_fn = self.choose_node_by_pheromones
                select_license_fn = self.select_license_by_pheromones
            case _:
                raise ValueError("Invalid solution type")

        assignments: Assignments = {}
        cost = 0.0

        while available_nodes:
            license_holder = choose_node_fn(available_nodes)
            license, covered_nodes = select_license_fn(license_holder)

            self.covered_nodes.update(covered_nodes)

            assignment = LicenseAssignment(
                license_holder=license_holder, covered_nodes=covered_nodes
            )

            assignments.setdefault(license.name, []).append(assignment)

            cost += license.cost

            available_nodes -= self.covered_nodes

        return assignments, cost
