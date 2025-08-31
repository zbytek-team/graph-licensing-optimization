from collections.abc import Callable

import networkx as nx

GeneratorFn = Callable[..., nx.Graph]


class GraphGeneratorFactory:

    _GENERATORS: dict[str, GeneratorFn] = {
        "random": lambda *, n_nodes, **p: GraphGeneratorFactory._random(n_nodes, **p),
        "scale_free": lambda *, n_nodes, **p: GraphGeneratorFactory._scale_free(n_nodes, **p),
        "small_world": lambda *, n_nodes, **p: GraphGeneratorFactory._small_world(n_nodes, **p),
        "complete": lambda *, n_nodes, **p: GraphGeneratorFactory._complete(n_nodes, **p),
        "star": lambda *, n_nodes, **p: GraphGeneratorFactory._star(n_nodes, **p),
        "path": lambda *, n_nodes, **p: GraphGeneratorFactory._path(n_nodes, **p),
        "cycle": lambda *, n_nodes, **p: GraphGeneratorFactory._cycle(n_nodes, **p),
        "tree": lambda *, n_nodes, **p: GraphGeneratorFactory._tree(n_nodes, **p),
    }

    @classmethod
    def get(cls, name: str) -> GeneratorFn:
        try:
            return cls._GENERATORS[name]
        except KeyError:
            available = ", ".join(cls._GENERATORS.keys())
            raise ValueError(f"unknown graph generator '{name}'. available: {available}")

    @staticmethod
    def _random(n_nodes: int, *, p: float = 0.1, seed: int | None = None) -> nx.Graph:
        return nx.gnp_random_graph(n=n_nodes, p=p, seed=seed)

    @staticmethod
    def _scale_free(n_nodes: int, *, m: int = 2, seed: int | None = None) -> nx.Graph:
        return nx.barabasi_albert_graph(n=n_nodes, m=m, seed=seed)

    @staticmethod
    def _small_world(n_nodes: int, *, k: int = 4, p: float = 0.1, seed: int | None = None) -> nx.Graph:
        return nx.watts_strogatz_graph(n=n_nodes, k=k, p=p, seed=seed)

    @staticmethod
    def _complete(n_nodes: int) -> nx.Graph:
        return nx.complete_graph(n=n_nodes)

    @staticmethod
    def _star(n_nodes: int) -> nx.Graph:
        return nx.star_graph(n=n_nodes - 1)

    @staticmethod
    def _path(n_nodes: int) -> nx.Graph:
        return nx.path_graph(n=n_nodes)

    @staticmethod
    def _cycle(n_nodes: int) -> nx.Graph:
        return nx.cycle_graph(n=n_nodes)

    @staticmethod
    def _tree(n_nodes: int, *, seed: int | None = None) -> nx.Graph:
        import networkx as nx

        if n_nodes == 1:
            G = nx.Graph()
            G.add_node(0)
            return G

        base = nx.complete_graph(n_nodes)
        return nx.random_spanning_tree(base, weight=None, seed=seed)
