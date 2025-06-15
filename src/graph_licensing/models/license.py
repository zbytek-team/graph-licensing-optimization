"""License models and data structures."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Any

if TYPE_CHECKING:
    import networkx as nx


@dataclass
class LicenseTypeConfig:
    """Configuration for a single license type."""
    
    price: float
    min_size: int
    max_size: int
    
    def is_valid_size(self, size: int) -> bool:
        """Check if the group size is valid for this license type."""
        return self.min_size <= size <= self.max_size
    
    def cost_per_person(self, size: int) -> float:
        """Calculate cost per person for this license type."""
        if not self.is_valid_size(size):
            return float('inf')
        return self.price / size


@dataclass
class LicenseConfig:
    """Configuration for all available license types."""

    license_types: Dict[str, LicenseTypeConfig]
    
    @classmethod
    def create_flexible(cls, license_configs: Dict[str, Dict[str, Any]]):
        """Create flexible license configuration from dictionary.
        
        Args:
            license_configs: Dictionary like {
                "solo": {"price": 1.0, "min_size": 1, "max_size": 1},
                "duo": {"price": 1.6, "min_size": 2, "max_size": 2},
                "family": {"price": 2.08, "min_size": 2, "max_size": 6}
            }
        """
        license_types = {}
        for name, config in license_configs.items():
            license_types[name] = LicenseTypeConfig(
                price=config["price"],
                min_size=config["min_size"],
                max_size=config["max_size"]
            )
        return cls(license_types=license_types)
    
    def get_best_license_for_size(self, size: int) -> tuple[str, LicenseTypeConfig] | None:
        """Find the most cost-effective license type for a given group size."""
        best_license = None
        best_cost_per_person = float('inf')
        
        for name, license_type in self.license_types.items():
            if license_type.is_valid_size(size):
                cost_per_person = license_type.cost_per_person(size)
                if cost_per_person < best_cost_per_person:
                    best_cost_per_person = cost_per_person
                    best_license = (name, license_type)
        
        return best_license
    
    def is_size_beneficial(self, license_name: str, size: int) -> bool:
        """Check if a specific license type is beneficial for given size."""
        if license_name not in self.license_types:
            return False
            
        license_type = self.license_types[license_name]
        if not license_type.is_valid_size(size):
            return False
        
        # Compare with best alternative
        best_alternative = self.get_best_license_for_size(size)
        if best_alternative is None:
            return True
        
        return license_type.cost_per_person(size) <= best_alternative[1].cost_per_person(size)


@dataclass
class LicenseSolution:
    """Solution for the licensing optimization problem.
    
    Structure: {
        "license_type_name": {
            owner_id: [owner_id, member1, member2, ...]
        }
    }
    """

    licenses: Dict[str, Dict[int, List[int]]]  # license_type -> {owner_id -> [members]}

    def __post_init__(self) -> None:
        """Validate the solution after initialization."""
        # Ensure owners include themselves in their member lists
        for license_type, groups in self.licenses.items():
            for owner_id, members in groups.items():
                if owner_id not in members:
                    members.insert(0, owner_id)

    def get_node_license_info(self, node_id: int) -> tuple[str, int] | None:
        """Get the license type and owner for a specific node.

        Args:
            node_id: The node identifier.

        Returns:
            Tuple of (license_type, owner_id) or None if not found.
        """
        for license_type, groups in self.licenses.items():
            for owner_id, members in groups.items():
                if node_id in members:
                    return (license_type, owner_id)
        return None

    def get_all_nodes(self) -> List[int]:
        """Get all nodes in the solution."""
        all_nodes = []
        for license_type, groups in self.licenses.items():
            for owner_id, members in groups.items():
                all_nodes.extend(members)
        return list(set(all_nodes))  # Remove duplicates

    def calculate_cost(self, config: LicenseConfig) -> float:
        """Calculate total cost of the solution.

        Args:
            config: License configuration.

        Returns:
            Total cost of all licenses.
        """
        total_cost = 0.0
        
        for license_type, groups in self.licenses.items():
            if license_type not in config.license_types:
                continue
                
            license_config = config.license_types[license_type]
            
            for owner_id, members in groups.items():
                group_size = len(members)
                if license_config.is_valid_size(group_size):
                    total_cost += license_config.price
                else:
                    # Invalid group size - return high penalty
                    total_cost += float('inf')
        
        return total_cost

    def is_valid(self, graph: "nx.Graph", config: "LicenseConfig") -> bool:
        """Check if the solution is valid.

        Args:
            graph: The social network graph.
            config: License configuration.

        Returns:
            True if solution is valid.
        """
        try:
            # Check if all nodes are covered
            graph_nodes = set(graph.nodes())
            solution_nodes = set(self.get_all_nodes())
            
            if graph_nodes != solution_nodes:
                return False
            
            # Check if all groups are valid
            for license_type, groups in self.licenses.items():
                if license_type not in config.license_types:
                    return False
                    
                license_config = config.license_types[license_type]
                
                for owner_id, members in groups.items():
                    # Check group size constraints
                    if not license_config.is_valid_size(len(members)):
                        return False
                    
                    # Check if owner is in the graph
                    if owner_id not in graph_nodes:
                        return False
                    
                    # Check if all members are connected to owner (for groups > 1)
                    if len(members) > 1:
                        for member in members:
                            if member != owner_id and not graph.has_edge(owner_id, member):
                                return False
            
            return True
            
        except Exception:
            return False

    @classmethod
    def create_empty(cls) -> "LicenseSolution":
        """Create an empty solution."""
        return cls(licenses={})
