from typing import Any
import networkx as nx


class GraphGeneratorFactory:
    @classmethod
    def get_generator(cls, name: str):
        generators = {
            "random": cls.random,
            "scale_free": cls.scale_free,
            "small_world": cls.small_world,
            "complete": cls.complete,
            "star": cls.star,
            "path": cls.path,
            "cycle": cls.cycle,
            "tree": cls.tree,
        }
        if name not in generators:
            available = ", ".join(generators.keys())
            raise ValueError(f"Unknown graph generator '{name}'. Available options: {available}")
        return generators[name]

    @staticmethod
    def random(n_nodes: int, **kwargs: Any) -> nx.Graph:
        p = kwargs.get("p", 0.1)
        seed = kwargs.get("seed", None)
        return nx.erdos_renyi_graph(n=n_nodes, p=p, seed=seed)

    @staticmethod
    def scale_free(n_nodes: int, **kwargs: Any) -> nx.Graph:
        m = kwargs.get("m", 2)
        seed = kwargs.get("seed", None)
        return nx.barabasi_albert_graph(n=n_nodes, m=m, seed=seed)

    @staticmethod
    def small_world(n_nodes: int, **kwargs: Any) -> nx.Graph:
        k = kwargs.get("k", 4)
        p = kwargs.get("p", 0.1)
        seed = kwargs.get("seed", None)
        return nx.watts_strogatz_graph(n=n_nodes, k=k, p=p, seed=seed)

    @staticmethod
    def complete(n_nodes: int, **kwargs: Any) -> nx.Graph:
        return nx.complete_graph(n=n_nodes)

    @staticmethod
    def star(n_nodes: int, **kwargs: Any) -> nx.Graph:
        return nx.star_graph(n=n_nodes - 1)

    @staticmethod
    def path(n_nodes: int, **kwargs: Any) -> nx.Graph:
        return nx.path_graph(n=n_nodes)

    @staticmethod
    def cycle(n_nodes: int, **kwargs: Any) -> nx.Graph:
        return nx.cycle_graph(n=n_nodes)

    @staticmethod
    def tree(n_nodes: int, **kwargs: Any) -> nx.Graph:
        seed = kwargs.get("seed", None)
        if n_nodes <= 0:
            return nx.Graph()
        if n_nodes == 1:
            G = nx.Graph()
            G.add_node(0)
            return G
        while True:
            p = min(0.3, 4.0 / n_nodes)
            G = nx.erdos_renyi_graph(n=n_nodes, p=p, seed=seed)
            if nx.is_connected(G):
                return nx.minimum_spanning_tree(G)
            if seed is not None:
                seed += 1
