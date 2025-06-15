"""Solution manipulation operators."""

import random
from typing import TYPE_CHECKING, List, Set, Tuple

if TYPE_CHECKING:
    import networkx as nx
    from ...models.license import LicenseConfig, LicenseSolution


class SolutionOperators:
    """Common operators for manipulating solutions."""

    @staticmethod
    def get_solution_neighbors(
        solution: "LicenseSolution", 
        graph: "nx.Graph", 
        config: "LicenseConfig",
        max_neighbors: int = 50
    ) -> List["LicenseSolution"]:
        """Generate neighboring solutions for local search algorithms."""
        neighbors = []
        
        # Move operations
        neighbors.extend(
            SolutionOperators._generate_move_neighbors(solution, graph, config, max_neighbors // 3)
        )
        
        # Swap operations  
        neighbors.extend(
            SolutionOperators._generate_swap_neighbors(solution, graph, config, max_neighbors // 3)
        )
        
        # License type change operations
        neighbors.extend(
            SolutionOperators._generate_license_change_neighbors(solution, graph, config, max_neighbors // 3)
        )
        
        return neighbors[:max_neighbors]

    @staticmethod
    def _generate_move_neighbors(
        solution: "LicenseSolution", 
        graph: "nx.Graph", 
        config: "LicenseConfig",
        max_count: int
    ) -> List["LicenseSolution"]:
        """Generate neighbors by moving nodes between groups."""
        from ...models.license import LicenseSolution
        
        neighbors = []
        all_nodes = solution.get_all_nodes()
        
        for _ in range(min(max_count, len(all_nodes) * 2)):
            if not all_nodes:
                break
                
            # Pick a random node to move
            node = random.choice(all_nodes)
            node_info = solution.get_node_license_info(node)
            if not node_info:
                continue
                
            current_license_type, current_owner = node_info
            
            # Try to move to a different group of the same license type
            current_groups = solution.licenses.get(current_license_type, {})
            other_owners = [owner for owner in current_groups.keys() if owner != current_owner]
            
            if other_owners:
                new_owner = random.choice(other_owners)
                new_solution = SolutionOperators._move_node_to_group(
                    solution, node, current_license_type, new_owner, config
                )
                if new_solution:
                    neighbors.append(new_solution)
        
        return neighbors

    @staticmethod
    def _generate_swap_neighbors(
        solution: "LicenseSolution", 
        graph: "nx.Graph", 
        config: "LicenseConfig",
        max_count: int
    ) -> List["LicenseSolution"]:
        """Generate neighbors by swapping nodes between groups."""
        from ...models.license import LicenseSolution
        
        neighbors = []
        all_nodes = solution.get_all_nodes()
        
        for _ in range(min(max_count, len(all_nodes))):
            if len(all_nodes) < 2:
                break
                
            # Pick two random nodes
            node1, node2 = random.sample(all_nodes, 2)
            
            new_solution = SolutionOperators._swap_nodes(solution, node1, node2, config)
            if new_solution:
                neighbors.append(new_solution)
        
        return neighbors

    @staticmethod
    def _generate_license_change_neighbors(
        solution: "LicenseSolution", 
        graph: "nx.Graph", 
        config: "LicenseConfig",
        max_count: int
    ) -> List["LicenseSolution"]:
        """Generate neighbors by changing license types of groups."""
        neighbors = []
        
        # Get all groups
        all_groups = []
        for license_type, groups in solution.licenses.items():
            for owner, members in groups.items():
                all_groups.append((license_type, owner, members))
        
        for _ in range(min(max_count, len(all_groups))):
            if not all_groups:
                break
                
            # Pick a random group
            current_license_type, owner, members = random.choice(all_groups)
            
            # Try a different license type
            other_license_types = [
                lt for lt in config.license_types.keys() 
                if lt != current_license_type
            ]
            
            if other_license_types:
                new_license_type = random.choice(other_license_types)
                new_solution = SolutionOperators._change_group_license_type(
                    solution, current_license_type, owner, new_license_type, config
                )
                if new_solution:
                    neighbors.append(new_solution)
        
        return neighbors

    @staticmethod
    def _move_node_to_group(
        solution: "LicenseSolution",
        node: int,
        license_type: str,
        new_owner: int,
        config: "LicenseConfig"
    ) -> "LicenseSolution":
        """Move a node to a different group."""
        from ...models.license import LicenseSolution
        from copy import deepcopy
        
        new_licenses = deepcopy(solution.licenses)
        
        # Remove node from current group
        current_info = solution.get_node_license_info(node)
        if not current_info:
            return None
            
        current_license_type, current_owner = current_info
        current_members = new_licenses[current_license_type][current_owner]
        
        if len(current_members) <= 1:
            # Can't remove the only member
            return None
            
        current_members.remove(node)
        
        # If removed node was the owner, pick a new owner
        if node == current_owner:
            new_current_owner = current_members[0]
            new_licenses[current_license_type][new_current_owner] = current_members
            del new_licenses[current_license_type][current_owner]
        
        # Add to new group
        if license_type not in new_licenses:
            new_licenses[license_type] = {}
        if new_owner not in new_licenses[license_type]:
            new_licenses[license_type][new_owner] = [new_owner]
            
        new_licenses[license_type][new_owner].append(node)
        
        # Validate the new solution
        new_solution = LicenseSolution(licenses=new_licenses)
        license_config = config.license_types[license_type]
        
        new_group_size = len(new_licenses[license_type][new_owner])
        if not license_config.is_valid_size(new_group_size):
            return None
            
        return new_solution

    @staticmethod
    def _swap_nodes(
        solution: "LicenseSolution",
        node1: int,
        node2: int,
        config: "LicenseConfig"
    ) -> "LicenseSolution":
        """Swap two nodes between their groups."""
        from ...models.license import LicenseSolution
        from copy import deepcopy
        
        info1 = solution.get_node_license_info(node1)
        info2 = solution.get_node_license_info(node2)
        
        if not info1 or not info2:
            return None
            
        license_type1, owner1 = info1
        license_type2, owner2 = info2
        
        # Don't swap if they're in the same group
        if license_type1 == license_type2 and owner1 == owner2:
            return None
        
        new_licenses = deepcopy(solution.licenses)
        
        # Remove both nodes from their current groups
        members1 = new_licenses[license_type1][owner1]
        members2 = new_licenses[license_type2][owner2]
        
        members1.remove(node1)
        members2.remove(node2)
        
        # Add nodes to their new groups
        members1.append(node2)
        members2.append(node1)
        
        # Handle owner changes if necessary
        if node1 == owner1:
            if members2:
                new_owner1 = members2[0] if node2 != owner2 else members2[1] if len(members2) > 1 else members2[0]
                new_licenses[license_type2][new_owner1] = members2
                if new_owner1 != owner2:
                    del new_licenses[license_type2][owner2]
        
        if node2 == owner2:
            if members1:
                new_owner2 = members1[0] if node1 != owner1 else members1[1] if len(members1) > 1 else members1[0]
                new_licenses[license_type1][new_owner2] = members1
                if new_owner2 != owner1:
                    del new_licenses[license_type1][owner1]
        
        return LicenseSolution(licenses=new_licenses)

    @staticmethod  
    def _change_group_license_type(
        solution: "LicenseSolution",
        old_license_type: str,
        owner: int,
        new_license_type: str,
        config: "LicenseConfig"
    ) -> "LicenseSolution":
        """Change the license type of a group."""
        from ...models.license import LicenseSolution
        from copy import deepcopy
        
        old_license_config = config.license_types[old_license_type]
        new_license_config = config.license_types[new_license_type]
        
        members = solution.licenses[old_license_type][owner]
        group_size = len(members)
        
        # Check if group size is valid for new license type
        if not new_license_config.is_valid_size(group_size):
            return None
        
        new_licenses = deepcopy(solution.licenses)
        
        # Remove from old license type
        del new_licenses[old_license_type][owner]
        if not new_licenses[old_license_type]:
            del new_licenses[old_license_type]
        
        # Add to new license type
        if new_license_type not in new_licenses:
            new_licenses[new_license_type] = {}
        new_licenses[new_license_type][owner] = members
        
        return LicenseSolution(licenses=new_licenses)

    @staticmethod
    def mutate_solution(
        solution: "LicenseSolution",
        graph: "nx.Graph", 
        config: "LicenseConfig",
        mutation_rate: float = 0.1
    ) -> "LicenseSolution":
        """Apply random mutations to a solution."""
        if random.random() > mutation_rate:
            return solution
            
        neighbors = SolutionOperators.get_solution_neighbors(
            solution, graph, config, max_neighbors=10
        )
        
        if neighbors:
            return random.choice(neighbors)
        return solution
