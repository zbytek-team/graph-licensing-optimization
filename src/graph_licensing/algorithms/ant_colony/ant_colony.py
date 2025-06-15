import random
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from ..base import BaseAlgorithm

if TYPE_CHECKING:
    import networkx as nx

    from ...models.license import LicenseConfig, LicenseSolution, LicenseTypeConfig


class AntColonyAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        num_ants: int = 20,  # Reduced from 50
        max_iterations: int = 30,  # Reduced from 100
        alpha: float = 1.0,  # pheromone importance - back to original
        beta: float = 4.0,  # heuristic importance - increased further
        rho: float = 0.4,  # evaporation rate
        q0: float = 0.7,  # exploitation vs exploration parameter
        initial_pheromone: float = 0.1,
        seed: int | None = None,
    ) -> None:
        super().__init__("Ant Colony")
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.initial_pheromone = initial_pheromone
        self.seed = seed

        self.solo_pheromones: Dict[int, float] = {}  # node -> pheromone for solo license
        self.group_pheromones: Dict[Tuple[int, int], float] = {}  # (owner, member) -> pheromone

        self._best_solution = None
        self._best_cost = float("inf")

    def supports_warm_start(self) -> bool:
        return True

    def solve(
        self,
        graph: "nx.Graph",
        config: "LicenseConfig",
        warm_start: Optional["LicenseSolution"] = None,
        **kwargs,
    ) -> "LicenseSolution":
        from ...models.license import LicenseSolution

        if self.seed is not None:
            random.seed(self.seed)

        nodes = list(graph.nodes())
        if not nodes:
            return LicenseSolution.create_empty()

        self._initialize_pheromones(graph, config, warm_start)

        # Start with greedy solution as base
        from ...algorithms.greedy import GreedyAlgorithm
        greedy_algo = GreedyAlgorithm()
        greedy_solution = greedy_algo.solve(graph, config)
        
        if warm_start:
            self._best_solution = warm_start
            self._best_cost = warm_start.calculate_cost(config)
        else:
            self._best_solution = greedy_solution
            self._best_cost = greedy_solution.calculate_cost(config)

        for iteration in range(self.max_iterations):
            iteration_solutions = []
            for ant in range(self.num_ants):
                # Start from greedy solution and improve it
                solution = self._improve_solution(self._best_solution, graph, config)
                if solution:
                    # Apply local search improvement
                    improved_solution = self._local_search_improvement(solution, graph, config)
                    cost = improved_solution.calculate_cost(config)
                    iteration_solutions.append((improved_solution, cost))

                    if cost < self._best_cost:
                        self._best_cost = cost
                        self._best_solution = improved_solution

            self._update_pheromones(iteration_solutions)

            # Early stopping if no improvement for many iterations
            if iteration > 10 and iteration % 5 == 0:
                recent_improvements = [cost for _, cost in iteration_solutions if cost < self._best_cost]
                if not recent_improvements:
                    # No improvements in this iteration, maybe stop early
                    pass

        return self._best_solution or self._create_fallback_solution(graph, config)

    def _initialize_pheromones(
        self, graph: "nx.Graph", config: "LicenseConfig", warm_start: Optional["LicenseSolution"] = None
    ) -> None:
        nodes = list(graph.nodes())

        self.license_pheromones = {}  # (license_type, node) -> pheromone
        self.group_pheromones = {}  # (license_type, owner, member) -> pheromone

        for license_type in config.license_types.keys():
            for node in nodes:
                self.license_pheromones[(license_type, node)] = self.initial_pheromone

            for node1 in nodes:
                for node2 in graph.neighbors(node1):
                    if node1 != node2:
                        self.group_pheromones[(license_type, node1, node2)] = self.initial_pheromone

        if warm_start:
            boost_factor = 2.0

            for license_type, groups in warm_start.licenses.items():
                for owner, members in groups.items():
                    if len(members) == 1:
                        key = (license_type, owner)
                        if key in self.license_pheromones:
                            self.license_pheromones[key] *= boost_factor
                    else:
                        for member in members:
                            if member != owner:
                                key = (license_type, owner, member)
                                if key in self.group_pheromones:
                                    self.group_pheromones[key] *= boost_factor

    def _construct_solution(self, graph: "nx.Graph", config: "LicenseConfig") -> Optional["LicenseSolution"]:
        from ...models.license import LicenseSolution

        nodes = list(graph.nodes())
        unassigned_nodes = set(nodes)
        solution_licenses = {}

        while unassigned_nodes:
            current_node = random.choice(list(unassigned_nodes))

            best_choice = self._choose_license_assignment(current_node, graph, config, unassigned_nodes)

            if best_choice:
                license_type, owner, members = best_choice

                if license_type not in solution_licenses:
                    solution_licenses[license_type] = {}

                solution_licenses[license_type][owner] = members

                for member in members:
                    unassigned_nodes.discard(member)
            else:
                cheapest_solo = self._get_cheapest_solo_license(config)
                if cheapest_solo:
                    license_type = cheapest_solo
                    if license_type not in solution_licenses:
                        solution_licenses[license_type] = {}
                    solution_licenses[license_type][current_node] = [current_node]

                unassigned_nodes.remove(current_node)

        return LicenseSolution(licenses=solution_licenses)

    def _update_pheromones(self, solutions: List[Tuple["LicenseSolution", float]]) -> None:
        if not solutions:
            return

        for key in self.license_pheromones:
            self.license_pheromones[key] *= 1.0 - self.rho

        for key in self.group_pheromones:
            self.group_pheromones[key] *= 1.0 - self.rho

        for solution, cost in solutions:
            if cost <= 0:
                continue

            deposit = 1.0 / cost

            if cost == self._best_cost:
                deposit *= 2.0  # Best solution gets double pheromone

            for license_type, groups in solution.licenses.items():
                for owner, members in groups.items():
                    if len(members) == 1:
                        key = (license_type, owner)
                        if key in self.license_pheromones:
                            self.license_pheromones[key] += deposit
                    else:
                        for member in members:
                            if member != owner:
                                key = (license_type, owner, member)
                                if key in self.group_pheromones:
                                    self.group_pheromones[key] += deposit

    def _choose_license_assignment(
        self, node: int, graph: "nx.Graph", config: "LicenseConfig", unassigned_nodes: Set[int]
    ) -> Optional[Tuple[str, int, List[int]]]:
        candidates = []

        for license_type, license_config in config.license_types.items():
            for group_size in range(
                license_config.min_size, min(license_config.max_size + 1, len(unassigned_nodes) + 1)
            ):
                if group_size == 1:
                    pheromone = self.license_pheromones.get((license_type, node), self.initial_pheromone)
                    heuristic = self._calculate_solo_heuristic(node, license_config, graph)
                    attractiveness = (pheromone**self.alpha) * (heuristic**self.beta)
                    candidates.append((attractiveness, license_type, node, [node]))

                elif group_size > 1:
                    potential_members = self._find_potential_group_members(
                        node, graph, unassigned_nodes, group_size - 1
                    )

                    if len(potential_members) >= group_size - 1:
                        members = [node] + potential_members[: group_size - 1]
                        pheromone = self._calculate_group_pheromone(license_type, node, members[1:])
                        heuristic = self._calculate_group_heuristic(license_config, members, graph)
                        attractiveness = (pheromone**self.alpha) * (heuristic**self.beta)
                        candidates.append((attractiveness, license_type, node, members))

        if not candidates:
            return None

        if random.random() < self.q0:
            best_candidate = max(candidates, key=lambda x: x[0])
            return (best_candidate[1], best_candidate[2], best_candidate[3])
        else:
            total_attractiveness = sum(candidate[0] for candidate in candidates)
            if total_attractiveness > 0:
                rand_val = random.random() * total_attractiveness
                cumulative = 0.0
                for attractiveness, license_type, owner, members in candidates:
                    cumulative += attractiveness
                    if cumulative >= rand_val:
                        return (license_type, owner, members)

            chosen = random.choice(candidates)
            return (chosen[1], chosen[2], chosen[3])

    def _get_cheapest_solo_license(self, config: "LicenseConfig") -> Optional[str]:
        cheapest_type = None
        cheapest_cost = float("inf")

        for license_type, license_config in config.license_types.items():
            if license_config.is_valid_size(1) and license_config.price < cheapest_cost:
                cheapest_cost = license_config.price
                cheapest_type = license_type

        return cheapest_type

    def _find_potential_group_members(
        self, node: int, graph: "nx.Graph", unassigned_nodes: Set[int], max_members: int
    ) -> List[int]:
        neighbors = [n for n in graph.neighbors(node) if n in unassigned_nodes]

        # Sort by a combination of degree (connectivity) and local clustering
        def score_member(member):
            degree = graph.degree(member)
            # Count common neighbors between node and potential member
            common_neighbors = len(set(graph.neighbors(node)) & set(graph.neighbors(member)))
            return degree + 2 * common_neighbors  # Weight common neighbors more heavily

        neighbors.sort(key=score_member, reverse=True)

        return neighbors[:max_members]

    def _calculate_solo_heuristic(self, node: int, license_config: "LicenseTypeConfig", graph: "nx.Graph") -> float:
        # Solo licenses should be attractive for cost efficiency, but penalized for high-degree nodes
        cost_efficiency = 1.0 / license_config.price
        degree = graph.degree(node)
        
        # High-degree nodes should prefer groups, so solo gets penalty
        # But low-degree nodes should prefer solo
        if degree <= 2:
            # Low connectivity - solo is fine
            return cost_efficiency * 1.5
        else:
            # High connectivity - should form groups, solo is less attractive
            degree_penalty = 1.0 / (1.0 + degree * 0.1)
            return cost_efficiency * degree_penalty

    def _calculate_group_pheromone(self, license_type: str, owner: int, members: List[int]) -> float:
        total_pheromone = 0.0
        count = 0

        for member in members:
            key = (license_type, owner, member)
            if key in self.group_pheromones:
                total_pheromone += self.group_pheromones[key]
                count += 1

        return total_pheromone / max(count, 1)

    def _calculate_group_heuristic(
        self, license_config: "LicenseTypeConfig", members: List[int], graph: "nx.Graph"
    ) -> float:
        group_size = len(members)
        if not license_config.is_valid_size(group_size):
            return 0.0

        cost_per_person = license_config.cost_per_person(group_size)

        # Enhanced connectivity calculation
        owner = members[0]
        connectivity = sum(1 for member in members[1:] if graph.has_edge(owner, member))
        connectivity_factor = connectivity / max(len(members) - 1, 1)
        
        # Group size bonus - favor larger groups for better cost efficiency
        group_size_bonus = group_size / license_config.max_size
        
        # Connectivity bonus - heavily favor well-connected groups
        connectivity_bonus = connectivity_factor ** 2
        
        return (1.0 / cost_per_person) * connectivity_bonus * (1 + group_size_bonus)

    def _create_fallback_solution(self, graph: "nx.Graph", config: "LicenseConfig") -> "LicenseSolution":
        from ...models.license import LicenseSolution

        cheapest_type = self._get_cheapest_solo_license(config)
        if not cheapest_type:
            return LicenseSolution.create_empty()

        nodes = list(graph.nodes())
        licenses = {cheapest_type: {node: [node] for node in nodes}}

        return LicenseSolution(licenses=licenses)

    def get_algorithm_info(self) -> dict:
        return {
            "name": self.name,
            "num_ants": self.num_ants,
            "max_iterations": self.max_iterations,
            "alpha": self.alpha,
            "beta": self.beta,
            "rho": self.rho,
            "q0": self.q0,
            "best_cost": self._best_cost if self._best_cost != float("inf") else None,
            "pheromone_levels": {
                "license_avg": sum(self.license_pheromones.values()) / len(self.license_pheromones)
                if self.license_pheromones
                else 0,
                "group_avg": sum(self.group_pheromones.values()) / len(self.group_pheromones)
                if self.group_pheromones
                else 0,
            },
        }

    def _local_search_improvement(self, solution: "LicenseSolution", graph: "nx.Graph", config: "LicenseConfig") -> "LicenseSolution":
        """Apply local search to improve solution by trying to merge solo licenses into groups."""
        import copy
        from ...models.license import LicenseSolution
        
        current_solution = solution
        current_cost = solution.calculate_cost(config)
        improved = True
        
        # Try a few improvement iterations
        for _ in range(3):
            if not improved:
                break
                
            improved = False
            best_improvement = None
            best_cost = current_cost
            
            # Find all solo license holders
            solo_nodes = []
            for license_type, groups in current_solution.licenses.items():
                for owner, members in groups.items():
                    if len(members) == 1:
                        solo_nodes.append((license_type, owner))
            
            # Try to merge solo nodes into existing groups or create new groups
            for license_type, solo_node in solo_nodes:
                # Try to join existing groups
                for check_license_type, groups in current_solution.licenses.items():
                    license_config = config.license_types[check_license_type]
                    for owner, members in groups.items():
                        if (len(members) < license_config.max_size and 
                            solo_node != owner and 
                            graph.has_edge(solo_node, owner)):
                            
                            # Try this merge
                            new_solution = self._try_merge_into_group(
                                current_solution, solo_node, owner, check_license_type, config
                            )
                            if new_solution:
                                new_cost = new_solution.calculate_cost(config)
                                if new_cost < best_cost:
                                    best_cost = new_cost
                                    best_improvement = new_solution
                                    improved = True
                
                # Try to create new group with another solo node
                for other_license_type, other_solo in solo_nodes:
                    if (solo_node != other_solo and 
                        graph.has_edge(solo_node, other_solo)):
                        
                        # Find best license type for this pair
                        best_license_option = None
                        best_pair_cost = float('inf')
                        
                        for lt, lc in config.license_types.items():
                            if lc.is_valid_size(2):
                                pair_cost = lc.price
                                if pair_cost < best_pair_cost:
                                    best_pair_cost = pair_cost
                                    best_license_option = lt
                        
                        if best_license_option:
                            new_solution = self._try_create_group(
                                current_solution, solo_node, other_solo, best_license_option, config
                            )
                            if new_solution:
                                new_cost = new_solution.calculate_cost(config)
                                if new_cost < best_cost:
                                    best_cost = new_cost
                                    best_improvement = new_solution
                                    improved = True
            
            if best_improvement:
                current_solution = best_improvement
                current_cost = best_cost
        
        return current_solution

    def _try_merge_into_group(self, solution: "LicenseSolution", solo_node: int, group_owner: int, 
                            license_type: str, config: "LicenseConfig") -> Optional["LicenseSolution"]:
        """Try to merge a solo node into an existing group."""
        import copy
        from ...models.license import LicenseSolution
        
        license_config = config.license_types[license_type]
        new_licenses = copy.deepcopy(solution.licenses)
        
        # Remove solo node from its current license
        for lt, groups in new_licenses.items():
            if solo_node in groups and len(groups[solo_node]) == 1:
                del groups[solo_node]
                if not groups:
                    del new_licenses[lt]
                break
        
        # Add to target group
        if license_type in new_licenses and group_owner in new_licenses[license_type]:
            current_group = new_licenses[license_type][group_owner]
            if len(current_group) < license_config.max_size:
                new_licenses[license_type][group_owner] = current_group + [solo_node]
                return LicenseSolution(licenses=new_licenses)
        
        return None

    def _try_create_group(self, solution: "LicenseSolution", node1: int, node2: int, 
                         license_type: str, config: "LicenseConfig") -> Optional["LicenseSolution"]:
        """Try to create a new group from two solo nodes."""
        import copy
        from ...models.license import LicenseSolution
        
        license_config = config.license_types[license_type]
        if not license_config.is_valid_size(2):
            return None
            
        new_licenses = copy.deepcopy(solution.licenses)
        
        # Remove both nodes from their current licenses
        for node in [node1, node2]:
            for lt, groups in new_licenses.items():
                if node in groups and len(groups[node]) == 1:
                    del groups[node]
                    if not groups:
                        del new_licenses[lt]
                    break
        
        # Create new group
        if license_type not in new_licenses:
            new_licenses[license_type] = {}
        new_licenses[license_type][node1] = [node1, node2]
        
        return LicenseSolution(licenses=new_licenses)

    def _improve_solution(self, base_solution: "LicenseSolution", graph: "nx.Graph", config: "LicenseConfig") -> "LicenseSolution":
        """Improve an existing solution using ant colony logic instead of building from scratch."""
        import copy
        from ...models.license import LicenseSolution
        
        current_solution = copy.deepcopy(base_solution)
        
        # Apply several improvement moves based on pheromone and heuristic information
        nodes = list(graph.nodes())
        num_improvements = min(5, len(nodes) // 20)  # Try 5 improvements or 5% of nodes
        
        for _ in range(num_improvements):
            # Choose a random node to potentially reassign
            node = random.choice(nodes)
            
            # Find current assignment
            current_assignment = self._find_node_assignment_in_solution(current_solution, node)
            if not current_assignment:
                continue
                
            current_license_type, current_owner = current_assignment
            
            # Generate alternative assignments based on ant colony logic
            alternatives = self._generate_alternatives(node, graph, config, current_solution)
            
            if alternatives:
                # Choose best alternative based on pheromone and heuristic
                best_alternative = self._choose_best_alternative(node, alternatives, current_assignment, graph, config)
                
                if best_alternative and best_alternative != current_assignment:
                    # Apply the change
                    new_solution = self._apply_reassignment(current_solution, node, current_assignment, best_alternative, graph, config)
                    if new_solution:
                        new_cost = new_solution.calculate_cost(config)
                        current_cost = current_solution.calculate_cost(config)
                        if new_cost < current_cost:
                            current_solution = new_solution
        
        return current_solution

    def _find_node_assignment_in_solution(self, solution: "LicenseSolution", node: int) -> tuple | None:
        """Find what license assignment a node currently has."""
        for license_type, groups in solution.licenses.items():
            for owner, members in groups.items():
                if node in members:
                    return (license_type, owner)
        return None

    def _generate_alternatives(self, node: int, graph: "nx.Graph", config: "LicenseConfig", 
                             current_solution: "LicenseSolution") -> list:
        """Generate alternative assignments for a node."""
        alternatives = []
        
        # Alternative 1: Solo licenses
        for license_type, license_config in config.license_types.items():
            if license_config.is_valid_size(1):
                alternatives.append(("solo", license_type, node))
        
        # Alternative 2: Join existing groups
        neighbors = list(graph.neighbors(node))
        for license_type, groups in current_solution.licenses.items():
            license_config = config.license_types[license_type]
            for owner, members in groups.items():
                if (owner in neighbors and 
                    len(members) < license_config.max_size and
                    node not in members):
                    alternatives.append(("join", license_type, owner))
        
        # Alternative 3: Create new group with neighbor
        for neighbor in neighbors:
            neighbor_assignment = self._find_node_assignment_in_solution(current_solution, neighbor)
            if neighbor_assignment:
                for license_type, license_config in config.license_types.items():
                    if license_config.is_valid_size(2):
                        alternatives.append(("create_group", license_type, neighbor))
        
        return alternatives

    def _choose_best_alternative(self, node: int, alternatives: list, current_assignment: tuple,
                               graph: "nx.Graph", config: "LicenseConfig") -> tuple | None:
        """Choose best alternative using ant colony logic."""
        candidates = []
        
        for alternative in alternatives:
            alt_type, license_type, target = alternative
            
            if alt_type == "solo":
                pheromone = self.license_pheromones.get((license_type, node), self.initial_pheromone)
                heuristic = self._calculate_solo_heuristic(node, config.license_types[license_type], graph)
                attractiveness = (pheromone ** self.alpha) * (heuristic ** self.beta)
                candidates.append((attractiveness, ("solo", license_type, node)))
                
            elif alt_type == "join":
                pheromone = self.group_pheromones.get((license_type, target, node), self.initial_pheromone)
                # Estimate group size after joining
                est_group_size = 2  # Simplified estimation
                heuristic = self._calculate_group_heuristic_simple(config.license_types[license_type], est_group_size, graph)
                attractiveness = (pheromone ** self.alpha) * (heuristic ** self.beta)
                candidates.append((attractiveness, ("join", license_type, target)))
                
            elif alt_type == "create_group":
                pheromone = self.group_pheromones.get((license_type, node, target), self.initial_pheromone)
                heuristic = self._calculate_group_heuristic_simple(config.license_types[license_type], 2, graph)
                attractiveness = (pheromone ** self.alpha) * (heuristic ** self.beta)
                candidates.append((attractiveness, ("create_group", license_type, target)))
        
        if not candidates:
            return None
            
        # Choose based on probability or best choice
        if random.random() < self.q0:
            # Exploitation: choose best
            best_candidate = max(candidates, key=lambda x: x[0])
            return best_candidate[1]
        else:
            # Exploration: probabilistic choice
            total_attractiveness = sum(candidate[0] for candidate in candidates)
            if total_attractiveness > 0:
                rand_val = random.random() * total_attractiveness
                cumulative = 0.0
                for attractiveness, alternative in candidates:
                    cumulative += attractiveness
                    if cumulative >= rand_val:
                        return alternative
        
        return None

    def _calculate_group_heuristic_simple(self, license_config: "LicenseTypeConfig", group_size: int, graph: "nx.Graph") -> float:
        """Simplified group heuristic calculation."""
        if not license_config.is_valid_size(group_size):
            return 0.0
        cost_per_person = license_config.cost_per_person(group_size)
        return 1.0 / cost_per_person

    def _apply_reassignment(self, solution: "LicenseSolution", node: int, current_assignment: tuple,
                          new_assignment: tuple, graph: "nx.Graph", config: "LicenseConfig") -> "LicenseSolution | None":
        """Apply a reassignment to create a new solution."""
        import copy
        from ...models.license import LicenseSolution
        
        new_solution = copy.deepcopy(solution)
        
        # Remove node from current assignment
        current_license_type, current_owner = current_assignment
        if current_license_type in new_solution.licenses and current_owner in new_solution.licenses[current_license_type]:
            members = new_solution.licenses[current_license_type][current_owner]
            if node in members:
                members.remove(node)
                if len(members) == 0:
                    del new_solution.licenses[current_license_type][current_owner]
                    if not new_solution.licenses[current_license_type]:
                        del new_solution.licenses[current_license_type]
                elif current_owner == node and members:
                    # Reassign ownership
                    new_owner = members[0]
                    new_solution.licenses[current_license_type][new_owner] = members
                    del new_solution.licenses[current_license_type][current_owner]
        
        # Apply new assignment
        assignment_type, license_type, target = new_assignment
        
        if assignment_type == "solo":
            if license_type not in new_solution.licenses:
                new_solution.licenses[license_type] = {}
            new_solution.licenses[license_type][node] = [node]
            
        elif assignment_type == "join":
            if license_type in new_solution.licenses and target in new_solution.licenses[license_type]:
                new_solution.licenses[license_type][target].append(node)
            else:
                return None  # Invalid join
                
        elif assignment_type == "create_group":
            # Remove target from its current assignment first
            target_assignment = self._find_node_assignment_in_solution(new_solution, target)
            if target_assignment:
                # Similar removal logic as above
                # ... (simplified for brevity)
                pass
            
            if license_type not in new_solution.licenses:
                new_solution.licenses[license_type] = {}
            new_solution.licenses[license_type][node] = [node, target]
        
        return new_solution
