import random
from src.solvers.base import Solver, SolverResult
import networkx as nx

from src.logger import get_logger

logger = get_logger(__name__)


class AntColonySolver(Solver):
    def __init__(
        self,
        individual_cost: float,
        group_cost: float,
        group_size: int,
        ant_count: int,
        alpha: float,
        beta: float,
        evaporation_rate: float,
        iterations: int,
    ):
        super().__init__(individual_cost, group_cost, group_size)
        self.ant_count = ant_count
        self.alpha = alpha  # alpha is the weight of the pheromone
        self.beta = beta  # beta is the weight of the heuristic information
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = {}
        self.iterations = iterations

    def _make_solution(self, graph: nx.Graph) -> SolverResult:
        solution = {"individual": set(), "group": {}}
        unvisited_nodes = list(graph.nodes)
        random.shuffle(unvisited_nodes)

        for node in unvisited_nodes:
            if node in solution["individual"] or any(
                node in group for group in solution["group"].values()
            ):
                continue

            pheromones = self.pheromone_deposit[node]

            heuristic = {
                "individual": 1 / (10),
                "group": 1 / (10),
            }

            probabilities = {
                "individual": (pheromones["individual"] ** self.alpha)
                * (heuristic["individual"] ** self.beta),
                "group": (pheromones["group"] ** self.alpha)
                * (heuristic["group"] ** self.beta),
            }

            # Normalizacja do sumy 1
            total = sum(probabilities.values())
            probabilities = {k: v / total for k, v in probabilities.items()}
            choice = random.choices(
                list(probabilities.keys()), weights=probabilities.values()
            )[0]

            if choice == "individual":
                solution["individual"].add(node)
            elif choice == "group":
                neighbors = list(graph.neighbors(node))
                random.shuffle(neighbors)
                group_members = {node}

                pheromones_O = []
                probabilities_O = []
                for neighbor in neighbors:
                    pheromones_O.append(self.pheromone_deposit[neighbor]["O"])
                    probabilities_O.append(
                        (pheromones_O[-1] ** self.alpha)
                        * (1 / (1 + self.group_cost)) ** self.beta
                    )

                total_O = sum(probabilities_O)
                probabilities_O = [p / total_O for p in probabilities_O]

                while len(group_members) < self.group_size and len(neighbors) > 0:
                    neighbor = random.choices(neighbors, weights=probabilities_O)[0]
                    group_members.add(neighbor)
                    probabilities_O = [
                        p
                        for i, p in enumerate(probabilities_O)
                        if i != neighbors.index(neighbor)
                    ]
                    neighbors = [n for n in neighbors if n != neighbor]

                if len(group_members) > 1:
                    solution["group"][node] = group_members
                else:
                    solution["individual"].add(node)

        return solution

    def _update_pheromones(self, solutions: list[SolverResult, float], graph: nx.Graph):
        for node in graph.nodes:
            for node_type in ["O", "individual", "group"]:
                self.pheromone_deposit[node][node_type] *= 1 - self.evaporation_rate

        for solution, cost in solutions:
            for node in solution["individual"]:
                self.pheromone_deposit[node]["individual"] += 1 / cost

            for node in solution["group"].keys():
                self.pheromone_deposit[node]["group"] += 1 / cost

            for group in solution["group"].values():
                for node in group:
                    self.pheromone_deposit[node]["O"] += 1 / cost

    def _solve(self, graph: nx.Graph) -> SolverResult:
        node_types = ["O", "individual", "group"]
        for node in graph.nodes:
            self.pheromone_deposit[node] = {}
            for node_type in node_types:
                self.pheromone_deposit[node][node_type] = 1

        for _ in range(self.iterations):
            solutions = [self._make_solution(graph) for _ in range(self.ant_count)]
            results = [
                (solution, self.calculate_total_cost(solution))
                for solution in solutions
            ]
            self._update_pheromones(results, graph)

            best_solution = min(results, key=lambda x: x[1])[0]

        return best_solution
