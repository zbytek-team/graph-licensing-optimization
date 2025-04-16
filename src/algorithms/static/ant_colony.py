import random
from concurrent.futures import ProcessPoolExecutor
import networkx as nx
from src.utils.calculate_cost import calculate_cost
from ..base import BaseSolver, Solution


class AntColonySolver(BaseSolver):
    def __init__(
        self,
        ant_count: int,
        alpha: float,
        beta: float,
        evaporation_rate: float,
        iterations: int,
    ):
        self.ant_count = ant_count
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.iterations = iterations
        self.pheromone_deposit = {}
        self.pheromone_deposit_move = {}
        self._neighbors = {}
        self.neighbors_cache = {}

    def _make_solution(self, graph: nx.Graph, group_size: int) -> tuple[Solution, list]:
        alpha = self.alpha
        beta = self.beta
        _neighbors = self._neighbors
        neighbors_cache = self.neighbors_cache

        solution: Solution = {"singles": [], "groups": []}
        unvisited_nodes = set(graph.nodes)
        visiting_order = []
        degree = {node: len(_neighbors[node]) for node in unvisited_nodes}

        while unvisited_nodes:
            nodes_list = list(unvisited_nodes)
            weights = [
                max(
                    (self.pheromone_deposit_move[node] ** alpha) * ((1 + degree[node]) ** beta),
                    1e-10,
                )
                for node in nodes_list
            ]
            chosen_node = random.choices(nodes_list, weights=weights, k=1)[0]
            visiting_order.append(chosen_node)
            unvisited_nodes.remove(chosen_node)

            pheromones = self.pheromone_deposit[chosen_node]
            node_neighbors = neighbors_cache[chosen_node]
            n_neighbors = len(node_neighbors)

            h_singles = beta / (n_neighbors + 1e-10)
            h_group = n_neighbors

            p_singles = (pheromones["singles"] ** alpha) * (h_singles**beta)
            p_group = (pheromones["group"] ** alpha) * (h_group**beta)
            total = p_singles + p_group
            if total > 0:
                p_singles /= total
                p_group /= total
            else:
                p_singles = 1
                p_group = 0

            choice = random.choices(["singles", "group"], weights=[p_singles, p_group], k=1)[0]

            if choice == "singles":
                solution["singles"].append(chosen_node)
            elif choice == "group":
                available_neighbors = [n for n in _neighbors[chosen_node] if n in unvisited_nodes]
                group_members = {chosen_node}
                if available_neighbors:
                    denom = len(_neighbors[chosen_node])
                    prob_O = [
                        (self.pheromone_deposit[chosen_node]["O"][neighbor] ** alpha) * ((1 / denom) ** beta)
                        for neighbor in available_neighbors
                    ]
                    total_O = sum(prob_O)
                    if total_O > 0:
                        prob_O = [p / total_O for p in prob_O]
                    else:
                        n_avail = len(available_neighbors)
                        prob_O = [1 / n_avail] * n_avail

                    while len(group_members) < group_size and available_neighbors:
                        selected_neighbor = random.choices(available_neighbors, weights=prob_O, k=1)[0]
                        group_members.add(selected_neighbor)
                        unvisited_nodes.remove(selected_neighbor)
                        idx = available_neighbors.index(selected_neighbor)
                        available_neighbors.pop(idx)
                        prob_O.pop(idx)

                if len(group_members) > 1:
                    solution["groups"].append({"license_holder": chosen_node, "members": list(group_members)})
                else:
                    solution["singles"].append(chosen_node)

        return solution, visiting_order

    def _evaporate_pheromones(self) -> None:
        evap = 1 - self.evaporation_rate
        for node, nbrs in self._neighbors.items():
            self.pheromone_deposit_move[node] *= evap
            for p_type in ["O", "singles", "group"]:
                if p_type == "O":
                    for neighbor in nbrs:
                        self.pheromone_deposit[node]["O"][neighbor] *= evap
                else:
                    self.pheromone_deposit[node][p_type] *= evap

    def _deposit_pheromones(self, solutions: list[tuple[Solution, float]], node_orders: list[list]) -> None:
        for (solution, cost), order in zip(solutions, node_orders):
            delta = 1 / (cost + 1e-10)
            for node in order:
                self.pheromone_deposit_move[node] += delta
            for node in solution["singles"]:
                self.pheromone_deposit[node]["singles"] += delta
            for group_data in solution["groups"]:
                leader = group_data["license_holder"]
                members = group_data["members"]
                self.pheromone_deposit[leader]["group"] += delta
                for member in members:
                    for neighbor in self._neighbors[member]:
                        self.pheromone_deposit[member]["O"][neighbor] += delta

    def _update_pheromones(self, solutions: list[tuple[Solution, float]], node_orders: list[list]) -> None:
        self._evaporate_pheromones()
        self._deposit_pheromones(solutions, node_orders)

    def solve(self, graph: nx.Graph, c_single: float, c_group: float, group_size: int) -> Solution:
        best_solution = None
        best_cost = float("inf")
        node_types = ["O", "singles", "group"]
        self._neighbors = {node: list(graph.neighbors(node)) for node in graph.nodes}
        self.neighbors_cache = {node: list(graph.adj[node]) for node in graph.nodes}

        for node in graph.nodes:
            self.pheromone_deposit[node] = {}
            for p_type in node_types:
                if p_type == "O":
                    self.pheromone_deposit[node][p_type] = {nbr: 1 for nbr in self._neighbors[node]}
                else:
                    self.pheromone_deposit[node][p_type] = 1
            self.pheromone_deposit_move[node] = 1

        with ProcessPoolExecutor() as executor:
            for _ in range(self.iterations):
                generated = list(
                    executor.map(self._make_solution, [graph] * self.ant_count, [group_size] * self.ant_count)
                )
                solutions, orders = zip(*generated)
                results = [(sol, calculate_cost(sol, c_single, c_group)) for sol in solutions]
                self._update_pheromones(results, list(orders))
                for sol, cost in results:
                    if cost < best_cost:
                        best_solution = sol
                        best_cost = cost

        if best_solution is None:
            raise ValueError("No valid solution found.")
        return best_solution
