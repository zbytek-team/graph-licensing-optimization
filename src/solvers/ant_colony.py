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
        assigned_nodes = set()  # Zbiór trzymający przypisane węzły

        unvisited_nodes = list(graph.nodes)
        random.shuffle(unvisited_nodes)

        for node in unvisited_nodes:
            if node in assigned_nodes:
                continue

            pheromones = self.pheromone_deposit[node]

            heuristic = {
                "individual": 1 / (1 + self.individual_cost),
                "group": 1 / self.group_cost if self.group_cost > 0 else 1,
            }

            probabilities = {
                "individual": (pheromones["individual"] ** self.alpha) * (heuristic["individual"] ** self.beta),
                "group": (pheromones["group"] ** self.alpha) * (heuristic["group"] ** self.beta),
            }

            total = sum(probabilities.values())
            probabilities = (
                {k: v / total for k, v in probabilities.items()}
                if total > 0
                else {"individual": 1}
            )

            choice = random.choices(
                list(probabilities.keys()), weights=probabilities.values()
            )[0]

            if choice == "individual":
                solution["individual"].add(node)
                assigned_nodes.add(node)
            elif choice == "group":
                neighbors = [
                    neighbor
                    for neighbor in graph.neighbors(node)
                    if neighbor not in assigned_nodes
                ]
                random.shuffle(neighbors)
                group_members = {node}

                if neighbors:
                    probabilities_O = [
                        (self.pheromone_deposit[node]["O"][neighbor] ** self.alpha)
                        * (1 / self.group_cost) ** self.beta
                        for neighbor in neighbors
                    ]
                    total_O = sum(probabilities_O)
                    probabilities_O = [p / total_O for p in probabilities_O]

                    while len(group_members) < self.group_size and probabilities_O:
                        neighbor = random.choices(neighbors, weights=probabilities_O)[0]
                        group_members.add(neighbor)
                        assigned_nodes.add(neighbor)
                        idx = neighbors.index(neighbor)
                        neighbors.pop(idx)
                        probabilities_O.pop(idx)

                if len(group_members) > 1:
                    solution["group"][node] = group_members
                    assigned_nodes.update(group_members)
                else:
                    solution["individual"].add(node)
                    assigned_nodes.add(node)

        return solution

    def _update_pheromones(self, solutions: list[SolverResult, float], graph: nx.Graph):
        for node in graph.nodes:
            for node_type in ["O", "individual", "group"]:
                if node_type == "O":
                    for neighbor in graph.neighbors(node):
                        self.pheromone_deposit[node][node_type][neighbor] *= (
                            1 - self.evaporation_rate
                        )
                else:
                    self.pheromone_deposit[node][node_type] *= 1 - self.evaporation_rate

        for solution, cost in solutions:
            for node in solution["individual"]:
                self.pheromone_deposit[node]["individual"] += 1 / cost

            for node in solution["group"].keys():
                self.pheromone_deposit[node]["group"] += 1 / cost

            for group in solution["group"].values():
                for node in group:
                    for neighbor in graph.neighbors(node):
                        self.pheromone_deposit[node]["O"][neighbor] += 1 / cost

    def _solve(self, graph: nx.Graph) -> SolverResult:
        best_solution = None
        best_cost = float("inf")
        node_types = ["O", "individual", "group"]
        for node in graph.nodes:
            self.pheromone_deposit[node] = {}
            for node_type in node_types:
                if node_type == "O":
                    self.pheromone_deposit[node][node_type] = {}
                    for neighbor in graph.neighbors(node):
                        self.pheromone_deposit[node][node_type][neighbor] = 1
                else:
                    self.pheromone_deposit[node][node_type] = 1

        for _ in range(self.iterations):
            solutions = [self._make_solution(graph) for _ in range(self.ant_count)]
            results = [(solution, self.calculate_total_cost(solution)) for solution in solutions]
            self._update_pheromones(results, graph)

            for solution, cost in results:
                if cost < best_cost:
                    best_solution = solution
                    best_cost = cost

        return best_solution
