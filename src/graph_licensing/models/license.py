"""License models and data structures."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import networkx as nx


class LicenseType(Enum):
    """Types of licenses available."""

    SOLO = "solo"
    GROUP_OWNER = "group_owner"
    GROUP_MEMBER = "group_member"


@dataclass
class LicenseConfig:
    """Configuration for license pricing and constraints."""

    solo_price: float
    group_price: float
    group_size: int

    @property
    def price_ratio(self) -> float:
        """Calculate the ratio of group price to solo price."""
        return self.group_price / self.solo_price

    def is_group_beneficial(self, group_members: int) -> bool:
        """Check if group license is more cost-effective than solo licenses.

        Args:
            group_members: Number of members in the potential group.

        Returns:
            True if group license is more cost-effective.
        """
        if group_members > self.group_size:
            return False
        return self.group_price < group_members * self.solo_price


@dataclass
class LicenseSolution:
    """Solution for the licensing optimization problem."""

    solo_nodes: list[int]
    group_owners: dict[int, list[int]]  # owner_id -> [owner_id, member1, member2, ...]

    def __post_init__(self) -> None:
        """Validate the solution after initialization."""
        # Ensure group owners include themselves in their member lists
        for owner_id, members in self.group_owners.items():
            if owner_id not in members:
                members.insert(0, owner_id)

    @property
    def total_cost(self) -> float:
        """Calculate total cost of the solution."""
        return 0.0  # This will be calculated by the optimization algorithms

    def get_node_license_type(self, node_id: int) -> LicenseType:
        """Get the license type for a specific node.

        Args:
            node_id: The node identifier.

        Returns:
            The license type for the node.
        """
        if node_id in self.solo_nodes:
            return LicenseType.SOLO

        for owner_id, members in self.group_owners.items():
            if node_id == owner_id:
                return LicenseType.GROUP_OWNER
            if node_id in members:
                return LicenseType.GROUP_MEMBER

        msg = f"Node {node_id} not found in any license assignment"
        raise ValueError(msg)

    def get_group_owner(self, node_id: int) -> int | None:
        """Get the group owner for a node, if it's a group member.

        Args:
            node_id: The node identifier.

        Returns:
            The owner node ID if the node is in a group, None otherwise.
        """
        for owner_id, members in self.group_owners.items():
            if node_id in members:
                return owner_id
        return None

    def calculate_cost(self, config: LicenseConfig) -> float:
        """Calculate the total cost of this solution.

        Args:
            config: License configuration with pricing information.

        Returns:
            Total cost of the solution.
        """
        solo_cost = len(self.solo_nodes) * config.solo_price
        group_cost = len(self.group_owners) * config.group_price
        return solo_cost + group_cost

    def is_valid(self, graph: "nx.Graph", config: "LicenseConfig") -> bool:
        """Validate that the solution is feasible.

        Args:
            graph: NetworkX graph representing the social network.
            config: License configuration.

        Returns:
            True if the solution is valid.
        """
        # Check all nodes are assigned
        all_nodes = set(graph.nodes())
        assigned_nodes = set(self.solo_nodes)

        for members in self.group_owners.values():
            assigned_nodes.update(members)

        if assigned_nodes != all_nodes:
            return False

        # Check group size constraints and connectivity
        for owner_id, members in self.group_owners.items():
            if len(members) > config.group_size:
                return False

            # Check that all members are connected to the owner
            for member_id in members:
                if member_id != owner_id and not graph.has_edge(owner_id, member_id):
                    return False

        return True
