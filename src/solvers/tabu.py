import random
import networkx as nx
from collections import deque
from src.solvers.base import Solver, SolverResult
from src.logger import get_logger
from src.solvers.greedy import GreedySolver

logger = get_logger(__name__)

class TabuSolver(Solver):
    def __init__(self, individual_cost: float, group_cost: float, group_size: int, tabu_size: int, iterations: int):
        super().__init__(individual_cost, group_cost, group_size)
        self.tabu_size = tabu_size
        self.iterations = iterations 
        self.tabu_list = deque(maxlen=tabu_size)

    def _generate_initial_solution(self, graph: nx.Graph) -> SolverResult:
        solution = {"individual": set(), "group": {}}
        nodes = list(graph.nodes)
        random.shuffle(nodes)

        unassigned_nodes = set(nodes)

        for node in nodes:
            if node in solution["individual"] or any(node in group for group in solution["group"].values()):
                continue

            neighbors = list(graph.neighbors(node))

            if random.random() < 0.5 and len(neighbors) > 0:
                group_members = {node} | set(random.sample(neighbors, min(len(neighbors), self.group_size - 1)))
                solution["group"][node] = group_members
                unassigned_nodes -= group_members
            else:
                solution["individual"].add(node)
                unassigned_nodes.discard(node)

        return solution
    
    def _generate_neighbors(self, solution: SolverResult, graph: nx.Graph):
        neighbors = []

        for node in graph.nodes:
            new_solution = {"individual": solution["individual"].copy(), "group": {k: v.copy() for k, v in solution["group"].items()}}

            if node in new_solution["individual"]:
                new_solution["individual"].remove(node)
                neighbors.append(new_solution)
            else:
                for holder, group in new_solution["group"].items():
                    if node in group:
                        new_solution["group"][holder].remove(node)
                        if len(new_solution["group"][holder]) < 2:
                            del new_solution["group"][holder]
                        neighbors.append(new_solution)
                        break

            if node not in new_solution["individual"] and node not in [n for g in new_solution["group"].values() for n in g]:
                new_solution["individual"].add(node)
                neighbors.append(new_solution)

        for neighbor in neighbors:
            all_nodes = set(graph.nodes)
            covered_nodes = neighbor["individual"].union(*neighbor["group"].values())
            if all_nodes != covered_nodes:
                missing_nodes = all_nodes - covered_nodes
                neighbor["individual"].update(missing_nodes)

        return neighbors

    def _best_valid_neighbor(self, neighbors, best_cost):
        best_neighbor = None
        best_neighbor_cost = float("inf")

        for neighbor in neighbors:
            cost = self.calculate_total_cost(neighbor)

            if cost < best_neighbor_cost and neighbor not in self.tabu_list:
                best_neighbor = neighbor
                best_neighbor_cost = cost

        if best_neighbor and best_neighbor_cost < best_cost:
            return best_neighbor, best_neighbor_cost
        return None, None

    def _solve(self, graph: nx.Graph) -> SolverResult:
        current_solution = self._generate_initial_solution(graph)
        best_solution = current_solution
        best_cost = self.calculate_total_cost(best_solution)

        for _ in range(self.iterations):
            neighbors = self._generate_neighbors(current_solution, graph)
            best_neighbor, best_neighbor_cost = self._best_valid_neighbor(neighbors, best_cost)

            if best_neighbor:
                current_solution = best_neighbor
                if best_neighbor_cost < best_cost:
                    best_solution = best_neighbor
                    best_cost = best_neighbor_cost

            self.tabu_list.append(current_solution)

        return best_solution