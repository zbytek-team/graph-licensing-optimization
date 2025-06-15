"""Refactored Tabu Search algorithm using common components."""

from typing import TYPE_CHECKING, Optional, Set, Tuple, Any
from collections import deque

from ..common.local_search_base import LocalSearchAlgorithm
from ..common.validation import SolutionValidator

if TYPE_CHECKING:
    import networkx as nx
    from ...models.license import LicenseConfig, LicenseSolution
    from ..common.config import TabuSearchConfig


class TabuSearchAlgorithm(LocalSearchAlgorithm):
    """Tabu Search algorithm for license optimization."""

    def __init__(
        self,
        max_iterations: int = 100,
        tabu_tenure: int = 7,
        max_no_improvement: int = 20,
        seed: int | None = None,
    ) -> None:
        super().__init__("TabuSearch", max_iterations, seed)
        self.tabu_tenure = tabu_tenure
        self.max_no_improvement = max_no_improvement
        
        # Algorithm state
        self.tabu_list: deque = deque()
        self.no_improvement_count = 0
        self.aspiration_threshold = float('inf')

    @classmethod
    def from_config(cls, config: "TabuSearchConfig"):
        """Create instance from configuration object."""
        return cls(
            max_iterations=config.max_iterations,
            tabu_tenure=config.tabu_tenure,
            max_no_improvement=config.max_no_improvement,
            seed=config.seed
        )

    def _initialize_algorithm_state(
        self, graph: "nx.Graph", config: "LicenseConfig", initial_solution: "LicenseSolution"
    ) -> None:
        """Initialize tabu search state."""
        self.tabu_list = deque(maxlen=self.tabu_tenure)
        self.no_improvement_count = 0
        self.aspiration_threshold = SolutionValidator.calculate_solution_fitness(initial_solution, config)

    def _select_next_solution(
        self,
        current_solution: "LicenseSolution",
        neighbors: list["LicenseSolution"],
        config: "LicenseConfig",
        iteration: int,
    ) -> Optional["LicenseSolution"]:
        """Select next solution using tabu search criteria."""
        if not neighbors:
            return None

        best_neighbor = None
        best_neighbor_cost = float('inf')
        
        for neighbor in neighbors:
            neighbor_cost = SolutionValidator.calculate_solution_fitness(neighbor, config)
            move_key = self._get_move_key(current_solution, neighbor)
            
            # Check if move is tabu
            is_tabu = move_key in self.tabu_list
            
            # Apply aspiration criteria (accept tabu move if it's better than best known)
            is_aspired = neighbor_cost < self.aspiration_threshold
            
            if not is_tabu or is_aspired:
                if neighbor_cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost

        if best_neighbor is not None:
            # Add move to tabu list
            move_key = self._get_move_key(current_solution, best_neighbor)
            self.tabu_list.append(move_key)
            
            return best_neighbor
            
        return None

    def _update_algorithm_state(
        self, current_solution: "LicenseSolution", current_cost: float, iteration: int
    ) -> None:
        """Update tabu search state."""
        if current_cost < self.aspiration_threshold:
            self.aspiration_threshold = current_cost
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

    def _should_terminate(self, iteration: int, current_cost: float, best_cost: float) -> bool:
        """Terminate if no improvement for too long."""
        return self.no_improvement_count >= self.max_no_improvement

    def _get_move_key(
        self, 
        from_solution: "LicenseSolution", 
        to_solution: "LicenseSolution"
    ) -> Tuple[Any, ...]:
        """Generate a key representing the move between two solutions."""
        # Simple approach: hash the difference in license assignments
        from_nodes = {}
        to_nodes = {}
        
        # Get node assignments for both solutions
        for license_type, groups in from_solution.licenses.items():
            for owner, members in groups.items():
                for member in members:
                    from_nodes[member] = (license_type, owner)
        
        for license_type, groups in to_solution.licenses.items():
            for owner, members in groups.items():
                for member in members:
                    to_nodes[member] = (license_type, owner) 
        
        # Find differences
        all_nodes = set(from_nodes.keys()) | set(to_nodes.keys())
        changes = []
        
        for node in all_nodes:
            from_assignment = from_nodes.get(node)
            to_assignment = to_nodes.get(node)
            
            if from_assignment != to_assignment:
                changes.append((node, from_assignment, to_assignment))
        
        # Create a stable hash of the changes
        changes.sort()
        return tuple(changes)

    def _get_neighborhood_size(self) -> int:
        """Use moderate neighborhood size for tabu search."""
        return 30
