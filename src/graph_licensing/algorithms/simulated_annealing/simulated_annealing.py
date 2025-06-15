import math
import random
from typing import TYPE_CHECKING, Optional

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx

    from ...models.license import LicenseConfig, LicenseSolution


class SimulatedAnnealingAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        initial_temp: float = 200.0,  # Increased from 100.0
        final_temp: float = 0.01,  # Decreased from 0.1 for more thorough search
        cooling_rate: float = 0.98,  # Increased from 0.95 for slower cooling
        max_iterations: int = 2000,  # Increased from 1000
        seed: int | None = None,
    ) -> None:
        super().__init__("SimulatedAnnealing")
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.seed = seed

    def supports_warm_start(self) -> bool:
        return True

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None,
        **kwargs,
    ) -> "LicenseSolution":
        from ...algorithms.greedy import GreedyAlgorithm
        from ...models.license import LicenseSolution

        if self.seed is not None:
            random.seed(self.seed)
        nodes = list(graph.nodes())
        if not nodes:
            return LicenseSolution.create_empty()
        if warm_start is not None:
            current_solution = self._adapt_solution_to_graph(warm_start, graph, config)
        else:
            greedy = GreedyAlgorithm()
            current_solution = greedy.solve(graph, config)
        current_cost = current_solution.calculate_cost(config)
        best_solution = current_solution
        best_cost = current_cost
        temperature = self.initial_temp
        for _iteration in range(self.max_iterations):
            if temperature < self.final_temp:
                break
            neighbor_solution = self._generate_neighbor(current_solution, graph, config)
            neighbor_cost = neighbor_solution.calculate_cost(config)
            cost_diff = neighbor_cost - current_cost
            if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                if current_cost < best_cost:
                    best_solution = current_solution
                    best_cost = current_cost
            temperature *= self.cooling_rate

        return best_solution

    def _adapt_solution_to_graph(
        self, solution: "LicenseSolution", graph: "nx.Graph", config: "LicenseConfig"
    ) -> "LicenseSolution":
        from ...models.license import LicenseSolution

        current_nodes = set(graph.nodes())
        new_licenses = {}
        assigned_nodes = set()
        for license_type, groups in solution.licenses.items():
            if license_type not in config.license_types:
                continue
            license_config = config.license_types[license_type]
            new_groups = {}
            for owner, members in groups.items():
                if owner in current_nodes:
                    valid_members = []
                    for member in members:
                        if member in current_nodes:
                            if member == owner or graph.has_edge(owner, member):
                                valid_members.append(member)
                    if valid_members and license_config.is_valid_size(len(valid_members)):
                        new_groups[owner] = valid_members
                        assigned_nodes.update(valid_members)
            if new_groups:
                new_licenses[license_type] = new_groups
        unassigned = current_nodes - assigned_nodes
        if unassigned:
            best_solo = config.get_best_license_for_size(1)
            if best_solo:
                license_type, _ = best_solo
                if license_type not in new_licenses:
                    new_licenses[license_type] = {}
                for node in unassigned:
                    new_licenses[license_type][node] = [node]
        return LicenseSolution(licenses=new_licenses)

    def _generate_neighbor(
        self,
        solution: "LicenseSolution",
        graph: "nx.Graph",
        config: "LicenseConfig",
    ) -> "LicenseSolution":
        import copy

        from ...models.license import LicenseSolution

        nodes = list(graph.nodes())
        if not nodes:
            return solution
            
        # Choose neighbor generation strategy
        strategies = ["move_node", "merge_solos", "split_group", "swap_nodes"]
        strategy = random.choice(strategies)
        
        if strategy == "move_node":
            return self._move_node_neighbor(solution, graph, config, nodes)
        elif strategy == "merge_solos":
            return self._merge_solos_neighbor(solution, graph, config)
        elif strategy == "split_group":
            return self._split_group_neighbor(solution, graph, config)
        elif strategy == "swap_nodes":
            return self._swap_nodes_neighbor(solution, graph, config, nodes)
        
        return solution

    def _move_node_neighbor(self, solution: "LicenseSolution", graph: "nx.Graph", 
                           config: "LicenseConfig", nodes: list) -> "LicenseSolution":
        """Original node movement strategy."""
        import copy
        from ...models.license import LicenseSolution

        new_licenses = copy.deepcopy(solution.licenses)
        node = random.choice(nodes)
        
        # Remove node from current assignment
        current_license_type = None
        current_owner = None
        current_members = None
        for license_type, groups in new_licenses.items():
            for owner, members in groups.items():
                if node in members:
                    current_license_type = license_type
                    current_owner = owner
                    current_members = members
                    break
            if current_license_type:
                break
                
        if current_license_type is None:
            return solution
            
        current_members.remove(node)
        if len(current_members) == 0:
            del new_licenses[current_license_type][current_owner]
            if len(new_licenses[current_license_type]) == 0:
                del new_licenses[current_license_type]
        elif current_owner == node:
            if current_members:
                new_owner = current_members[0]
                new_licenses[current_license_type][new_owner] = current_members
            del new_licenses[current_license_type][current_owner]
        
        # Reassign node intelligently
        modification_types = ["best_solo", "join_best_group", "create_group"]
        modification = random.choice(modification_types)
        
        if modification == "best_solo":
            best_license = config.get_best_license_for_size(1)
            if best_license:
                license_type, _ = best_license
                if license_type not in new_licenses:
                    new_licenses[license_type] = {}
                new_licenses[license_type][node] = [node]
                
        elif modification == "join_best_group":
            neighbors = list(graph.neighbors(node))
            potential_groups = []
            for license_type, groups in new_licenses.items():
                license_config = config.license_types[license_type]
                for owner, members in groups.items():
                    if owner in neighbors and len(members) < license_config.max_size:
                        cost_per_person = license_config.cost_per_person(len(members) + 1)
                        potential_groups.append((cost_per_person, license_type, owner))
            
            if potential_groups:
                # Choose best cost option
                potential_groups.sort()
                _, license_type, owner = potential_groups[0]
                new_licenses[license_type][owner].append(node)
            else:
                # Fallback to solo
                best_license = config.get_best_license_for_size(1)
                if best_license:
                    license_type, _ = best_license
                    if license_type not in new_licenses:
                        new_licenses[license_type] = {}
                    new_licenses[license_type][node] = [node]
                    
        elif modification == "create_group":
            neighbors = [n for n in graph.neighbors(node)]
            if neighbors:
                neighbor = random.choice(neighbors)
                # Find best license for 2-person group
                best_group_license = None
                best_cost = float('inf')
                for license_type, license_config in config.license_types.items():
                    if license_config.is_valid_size(2):
                        cost = license_config.price
                        if cost < best_cost:
                            best_cost = cost
                            best_group_license = license_type
                
                if best_group_license:
                    # Remove neighbor from current assignment
                    self._remove_node_from_licenses(new_licenses, neighbor)
                    # Create new group
                    if best_group_license not in new_licenses:
                        new_licenses[best_group_license] = {}
                    new_licenses[best_group_license][node] = [node, neighbor]
        
        return LicenseSolution(licenses=new_licenses)

    def _merge_solos_neighbor(self, solution: "LicenseSolution", graph: "nx.Graph", 
                             config: "LicenseConfig") -> "LicenseSolution":
        """Try to merge two connected solo licenses into a group."""
        import copy
        from ...models.license import LicenseSolution
        
        new_licenses = copy.deepcopy(solution.licenses)
        
        # Find solo licenses
        solo_nodes = []
        for license_type, groups in new_licenses.items():
            for owner, members in groups.items():
                if len(members) == 1:
                    solo_nodes.append(owner)
        
        if len(solo_nodes) < 2:
            return solution
            
        # Find connected solo nodes
        connected_pairs = []
        for i in range(len(solo_nodes)):
            for j in range(i + 1, len(solo_nodes)):
                if graph.has_edge(solo_nodes[i], solo_nodes[j]):
                    connected_pairs.append((solo_nodes[i], solo_nodes[j]))
        
        if not connected_pairs:
            return solution
            
        node1, node2 = random.choice(connected_pairs)
        
        # Remove both from current assignments
        self._remove_node_from_licenses(new_licenses, node1)
        self._remove_node_from_licenses(new_licenses, node2)
        
        # Find best group license
        best_license = None
        best_cost = float('inf')
        for license_type, license_config in config.license_types.items():
            if license_config.is_valid_size(2):
                cost = license_config.price
                if cost < best_cost:
                    best_cost = cost
                    best_license = license_type
        
        if best_license:
            if best_license not in new_licenses:
                new_licenses[best_license] = {}
            new_licenses[best_license][node1] = [node1, node2]
        
        return LicenseSolution(licenses=new_licenses)

    def _split_group_neighbor(self, solution: "LicenseSolution", graph: "nx.Graph", 
                             config: "LicenseConfig") -> "LicenseSolution":
        """Split a group by removing one member."""
        import copy
        from ...models.license import LicenseSolution
        
        new_licenses = copy.deepcopy(solution.licenses)
        
        # Find groups with more than 2 members
        large_groups = []
        for license_type, groups in new_licenses.items():
            for owner, members in groups.items():
                if len(members) > 2:
                    large_groups.append((license_type, owner, members))
        
        if not large_groups:
            return solution
            
        license_type, owner, members = random.choice(large_groups)
        
        # Remove a non-owner member
        non_owners = [m for m in members if m != owner]
        if not non_owners:
            return solution
            
        member_to_remove = random.choice(non_owners)
        
        # Remove from group
        new_licenses[license_type][owner] = [m for m in members if m != member_to_remove]
        
        # Assign to best solo license
        best_solo = config.get_best_license_for_size(1)
        if best_solo:
            solo_license_type, _ = best_solo
            if solo_license_type not in new_licenses:
                new_licenses[solo_license_type] = {}
            new_licenses[solo_license_type][member_to_remove] = [member_to_remove]
        
        return LicenseSolution(licenses=new_licenses)

    def _swap_nodes_neighbor(self, solution: "LicenseSolution", graph: "nx.Graph", 
                           config: "LicenseConfig", nodes: list) -> "LicenseSolution":
        """Swap assignments of two nodes."""
        import copy
        from ...models.license import LicenseSolution
        
        if len(nodes) < 2:
            return solution
            
        new_licenses = copy.deepcopy(solution.licenses)
        node1, node2 = random.sample(nodes, 2)
        
        # Find current assignments
        assignment1 = self._find_node_assignment(new_licenses, node1)
        assignment2 = self._find_node_assignment(new_licenses, node2)
        
        if assignment1 is None or assignment2 is None:
            return solution
            
        # Remove both nodes
        self._remove_node_from_licenses(new_licenses, node1)
        self._remove_node_from_licenses(new_licenses, node2)
        
        # Swap assignments (if valid)
        license_type1, owner1 = assignment1
        license_type2, owner2 = assignment2
        
        # Check if swaps are valid (connectivity constraints)
        valid_swap1 = (owner2 == node2 or graph.has_edge(node1, owner2))
        valid_swap2 = (owner1 == node1 or graph.has_edge(node2, owner1))
        
        if valid_swap1:
            if license_type2 not in new_licenses:
                new_licenses[license_type2] = {}
            if owner2 not in new_licenses[license_type2]:
                new_licenses[license_type2][owner2] = []
            new_licenses[license_type2][owner2].append(node1)
        else:
            # Fallback to solo
            best_solo = config.get_best_license_for_size(1)
            if best_solo:
                lt, _ = best_solo
                if lt not in new_licenses:
                    new_licenses[lt] = {}
                new_licenses[lt][node1] = [node1]
        
        if valid_swap2:
            if license_type1 not in new_licenses:
                new_licenses[license_type1] = {}
            if owner1 not in new_licenses[license_type1]:
                new_licenses[license_type1][owner1] = []
            new_licenses[license_type1][owner1].append(node2)
        else:
            # Fallback to solo
            best_solo = config.get_best_license_for_size(1)
            if best_solo:
                lt, _ = best_solo
                if lt not in new_licenses:
                    new_licenses[lt] = {}
                new_licenses[lt][node2] = [node2]
        
        return LicenseSolution(licenses=new_licenses)

    def _remove_node_from_licenses(self, licenses: dict, node: int) -> None:
        """Remove a node from all licenses."""
        to_delete = []
        for license_type, groups in licenses.items():
            for owner, members in list(groups.items()):
                if node in members:
                    members.remove(node)
                    if len(members) == 0:
                        to_delete.append((license_type, owner))
                    elif owner == node and len(members) > 0:
                        new_owner = members[0]
                        groups[new_owner] = members
                        to_delete.append((license_type, owner))
        
        for license_type, owner in to_delete:
            if owner in licenses[license_type]:
                del licenses[license_type][owner]
            if not licenses[license_type]:
                del licenses[license_type]

    def _find_node_assignment(self, licenses: dict, node: int) -> tuple | None:
        """Find the license assignment for a node."""
        for license_type, groups in licenses.items():
            for owner, members in groups.items():
                if node in members:
                    return (license_type, owner)
        return None
