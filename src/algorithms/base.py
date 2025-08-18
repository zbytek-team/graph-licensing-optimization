"""Base classes for optimization algorithms."""

from abc import ABC, abstractmethod
from typing import Any, List
import networkx as nx
from src.core import LicenseType, Solution


class BaseAlgorithm(ABC):
    """Abstract base for all licensing optimization algorithms."""

    @abstractmethod
    def solve(
        self, graph: nx.Graph, license_types: List[LicenseType], **kwargs: Any
    ) -> Solution:
        """Solve the optimization problem for a given graph and license types."""
        pass
