"""Graph generators for testing various network topologies."""

import random
from collections.abc import Callable

import networkx as nx

from .facebook_loader import FacebookDataLoader


class GraphGenerator:
    """Generator for various types of graphs used in licensing optimization testing."""

    @staticmethod
    def random_graph(n: int, p: float, seed: int | None = None) -> nx.Graph:
        """Generate a random Erdős-Rényi graph.

        Args:
            n: Number of nodes.
            p: Probability of edge creation.
            seed: Random seed for reproducibility.

        Returns:
            Random graph with n nodes.
        """
        return nx.erdos_renyi_graph(n, p, seed=seed)

    @staticmethod
    def scale_free_graph(n: int, m: int = 2, seed: int | None = None) -> nx.Graph:
        """Generate a scale-free Barabási-Albert graph.

        Args:
            n: Number of nodes.
            m: Number of edges to attach from a new node to existing nodes.
            seed: Random seed for reproducibility.

        Returns:
            Scale-free graph with n nodes.
        """
        return nx.barabasi_albert_graph(n, m, seed=seed)

    @staticmethod
    def small_world_graph(
        n: int,
        k: int = 4,
        p: float = 0.3,
        seed: int | None = None,
    ) -> nx.Graph:
        """Generate a small-world Watts-Strogatz graph.

        Args:
            n: Number of nodes.
            k: Each node is joined with its k nearest neighbors in a ring topology.
            p: Probability of rewiring each edge.
            seed: Random seed for reproducibility.

        Returns:
            Small-world graph with n nodes.
        """
        return nx.watts_strogatz_graph(n, k, p, seed=seed)

    @staticmethod
    def complete_graph(n: int) -> nx.Graph:
        """Generate a complete graph.

        Args:
            n: Number of nodes.

        Returns:
            Complete graph with n nodes.
        """
        return nx.complete_graph(n)

    @staticmethod
    def grid_graph(rows: int, cols: int) -> nx.Graph:
        """Generate a 2D grid graph.

        Args:
            rows: Number of rows in the grid.
            cols: Number of columns in the grid.

        Returns:
            Grid graph with rows * cols nodes.
        """
        return nx.grid_2d_graph(rows, cols)

    @staticmethod
    def star_graph(n: int) -> nx.Graph:
        """Generate a star graph.

        Args:
            n: Number of nodes (including the central node).

        Returns:
            Star graph with n nodes.
        """
        return nx.star_graph(n - 1)  # NetworkX star_graph takes number of outer nodes

    @staticmethod
    def path_graph(n: int) -> nx.Graph:
        """Generate a path graph.

        Args:
            n: Number of nodes.

        Returns:
            Path graph with n nodes.
        """
        return nx.path_graph(n)

    @staticmethod
    def cycle_graph(n: int) -> nx.Graph:
        """Generate a cycle graph.

        Args:
            n: Number of nodes.

        Returns:
            Cycle graph with n nodes.
        """
        return nx.cycle_graph(n)

    @staticmethod
    def random_regular_graph(n: int, d: int, seed: int | None = None) -> nx.Graph:
        """Generate a random d-regular graph.

        Args:
            n: Number of nodes.
            d: Degree of each node.
            seed: Random seed for reproducibility.

        Returns:
            Random regular graph with n nodes and degree d.
        """
        return nx.random_regular_graph(d, n, seed=seed)

    @staticmethod
    def powerlaw_cluster_graph(
        n: int,
        m: int,
        p: float,
        seed: int | None = None,
    ) -> nx.Graph:
        """Generate a graph with powerlaw degree distribution and triangle clustering.

        Args:
            n: Number of nodes.
            m: Number of random edges to add for each new node.
            p: Probability of adding a triangle after adding a random edge.
            seed: Random seed for reproducibility.

        Returns:
            Power-law cluster graph with n nodes.
        """
        return nx.powerlaw_cluster_graph(n, m, p, seed=seed)

    @staticmethod
    def facebook_graph(size: int = None, ego_id: str = None, seed: int | None = None) -> nx.Graph:
        """Load a Facebook ego network.

        Args:
            size: Ignored for Facebook graphs (included for compatibility).
            ego_id: Specific ego network ID to load. If None, loads random network.
            seed: Random seed for reproducibility when selecting random network.

        Returns:
            Facebook ego network graph.
        """
        loader = FacebookDataLoader()
        
        if ego_id is not None:
            return loader.load_ego_network(ego_id)
        else:
            graph, _ = loader.load_random_ego_network(seed=seed)
            return graph

    @classmethod
    def get_generator_function(cls, graph_type: str) -> Callable[..., nx.Graph]:
        """Get the generator function for a specific graph type.

        Args:
            graph_type: Type of graph to generate.

        Returns:
            Generator function for the specified graph type.

        Raises:
            ValueError: If graph_type is not supported.
        """
        generators = {
            "random": cls.random_graph,
            "scale_free": cls.scale_free_graph,
            "small_world": cls.small_world_graph,
            "complete": cls.complete_graph,
            "grid": cls.grid_graph,
            "star": cls.star_graph,
            "path": cls.path_graph,
            "cycle": cls.cycle_graph,
            "random_regular": cls.random_regular_graph,
            "powerlaw_cluster": cls.powerlaw_cluster_graph,
            "facebook": cls.facebook_graph,
        }

        if graph_type not in generators:
            supported = ", ".join(generators.keys())
            msg = f"Graph type '{graph_type}' not supported. Supported types: {supported}"
            raise ValueError(msg)

        return generators[graph_type]

    @classmethod
    def generate_graph(cls, graph_type: str, size: int, seed: int | None = None, **kwargs) -> nx.Graph:
        """Generate a graph of the specified type and size.

        Args:
            graph_type: Type of graph to generate.
            size: Number of nodes in the graph.
            seed: Random seed for reproducibility.
            **kwargs: Additional parameters for specific graph types.

        Returns:
            Generated graph.

        Raises:
            ValueError: If graph_type is not supported or invalid parameters.
        """
        generator = cls.get_generator_function(graph_type)

        if graph_type == "random":
            p = kwargs.get("p", 0.3)
            return generator(size, p, seed=seed)
        if graph_type == "scale_free":
            m = kwargs.get("m", min(2, size - 1)) if size > 1 else 0
            if m > 0:
                return generator(size, m, seed=seed)
            return nx.empty_graph(size)
        if graph_type == "small_world":
            k = kwargs.get("k", min(4, size - 1)) if size > 1 else 0
            if k % 2 == 1:  # k must be even for watts_strogatz_graph
                k -= 1
            p = kwargs.get("p", 0.3)
            if size >= 4 and k > 0:
                return generator(size, k, p, seed=seed)
            return nx.path_graph(size)
        if graph_type == "grid":
            rows = kwargs.get("rows", int(size**0.5))
            cols = kwargs.get("cols", size // rows if rows > 0 else 1)
            return generator(rows, cols)
        if graph_type == "random_regular":
            degree = kwargs.get("degree", min(3, size - 1)) if size > 1 else 0
            if size > 1 and degree > 0:
                return generator(size, degree, seed=seed)
            return nx.empty_graph(size)
        if graph_type == "powerlaw_cluster":
            m = kwargs.get("m", min(2, size - 1)) if size > 1 else 0
            p = kwargs.get("p", 0.3)
            if size >= 4 and m > 0:
                return generator(size, m, p, seed=seed)
            return nx.path_graph(size)
        if graph_type == "facebook":
            ego_id = kwargs.get("ego_id", None)
            return generator(size, ego_id=ego_id, seed=seed)
        # For simple graphs (complete, star, path, cycle)
        return generator(size)
