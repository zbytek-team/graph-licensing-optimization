import random
import copy
import networkx as nx
from src.solvers.base import Solver, SolverResult
from src.logger import get_logger
from src.solvers.greedy import GreedySolver

logger = get_logger(__name__)

class TabuSolver(Solver):
    def __init__(self, individual_cost: float, group_cost: float, group_size: int, tabu_size: int, iterations: int):
        super().__init__(individual_cost, group_cost, group_size)
        self.tabu_size = tabu_size
        self.iterations = iterations 

    def _generate_initial_solution(self, graph: nx.Graph) -> SolverResult:
        solution = {"individual": set(), "group": {}}
        nodes = list(graph.nodes)
        random.shuffle(nodes)        
        
        solution["individual"].update(nodes)

        return solution
    
    def _get_neighbors(self, solution: SolverResult, graph: nx.Graph) -> list:
        neighbors = []
        nodes = list(graph.nodes)

        possible_moves = [
            'individual_to_group',
            'group_to_individual'
        ]

        for _ in range(len(nodes)):  
            move = random.choice(possible_moves)
            neighbor_solution = copy.deepcopy(solution)

            match move:
                case 'individual_to_group':
                    if not neighbor_solution['individual']:
                        continue  

                    individual = random.choice(list(neighbor_solution['individual']))
                    node_neighbors = list(graph.neighbors(individual))

                    license_holders_candidates = [group for group, members in neighbor_solution['group'].items()
                                        if len(members) < self.group_size and group in node_neighbors]
                    if len(license_holders_candidates) > 0:
                        group = random.choice(license_holders_candidates)
                        neighbor_solution['individual'].remove(individual)
                        neighbor_solution['group'][group].add(individual)
                    else:
                        group_members_candidates = [ node for node in node_neighbors if node in neighbor_solution['individual']]
                        if len(group_members_candidates) == 0:
                            continue
                        if group_members_candidates:
                            neighbor_solution['individual'].remove(individual)
                            neighbor_solution['group'][individual] = {individual} | set(group_members_candidates)
                case 'group_to_individual':
                    if not neighbor_solution['group']:
                        continue

                    group_nodes = {node for group in neighbor_solution['group'].values() for node in group} 

                    node = random.choice(list(group_nodes))

                    if node in neighbor_solution['group']:
                        for member in neighbor_solution['group'][node]:
                            neighbor_solution['individual'].add(member)
                        del neighbor_solution['group'][node]  
                    else:
                        for group, members in neighbor_solution['group'].items():
                            if node in members:
                                neighbor_solution['individual'].add(node)
                                members.remove(node)
                                if len(members) == 1:
                                    remaining_node = members.pop()
                                    neighbor_solution['individual'].add(remaining_node)
                                    del neighbor_solution['group'][group]
                                break

            neighbors.append(neighbor_solution)

        return neighbors
    
    def _solve(self, graph: nx.Graph) -> SolverResult:
        best_solution = self._generate_initial_solution(graph)
        best_cost = self.calculate_total_cost(best_solution)
        
        current_solution = best_solution
        tabu_list = []

        for _ in range(self.iterations):
            neighbors = self._get_neighbors(current_solution, graph)
            best_neighbor = None
            best_neighbor_cost = float("inf")

            for neighbor in neighbors:
                if neighbor not in tabu_list:
                    neighbor_cost = self.calculate_total_cost(neighbor)
                    if neighbor_cost < best_neighbor_cost:
                        best_neighbor_cost = neighbor_cost
                        best_neighbor = neighbor
                    
            if best_neighbor is None:
                break  
            
            current_solution = best_neighbor
            tabu_list.append(best_neighbor)
            if len(tabu_list) > self.tabu_size:
                tabu_list.pop(0)

            if best_neighbor_cost < best_cost:
                best_solution = best_neighbor
                best_cost = best_neighbor_cost

        return best_solution
