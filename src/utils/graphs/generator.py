from typing import Any
from ...core.types import GraphGenerator

import networkx as nx


class RandomGraphGenerator(GraphGenerator):
    @property
    def name(self) -> str:
        return "random"

    def generate(self, n_nodes: int, **kwargs: Any) -> nx.Graph:
        p = kwargs.get("p", 0.1)
        seed = kwargs.get("seed", None)

        return nx.erdos_renyi_graph(n=n_nodes, p=p, seed=seed)


class ScaleFreeGraphGenerator(GraphGenerator):
    @property
    def name(self) -> str:
        return "scale_free"

    def generate(self, n_nodes: int, **kwargs: Any) -> nx.Graph:
        m = kwargs.get("m", 2)
        seed = kwargs.get("seed", None)

        return nx.barabasi_albert_graph(n=n_nodes, m=m, seed=seed)


class SmallWorldGraphGenerator(GraphGenerator):
    @property
    def name(self) -> str:
        return "small_world"

    def generate(self, n_nodes: int, **kwargs: Any) -> nx.Graph:
        k = kwargs.get("k", 4)
        p = kwargs.get("p", 0.1)
        seed = kwargs.get("seed", None)

        return nx.watts_strogatz_graph(n=n_nodes, k=k, p=p, seed=seed)


class CompleteGraphGenerator(GraphGenerator):
    @property
    def name(self) -> str:
        return "complete"

    def generate(self, n_nodes: int, **kwargs: Any) -> nx.Graph:
        return nx.complete_graph(n=n_nodes)


class StarGraphGenerator(GraphGenerator):
    @property
    def name(self) -> str:
        return "star"

    def generate(self, n_nodes: int, **kwargs: Any) -> nx.Graph:
        return nx.star_graph(n=n_nodes - 1)


class PathGraphGenerator(GraphGenerator):
    @property
    def name(self) -> str:
        return "path"

    def generate(self, n_nodes: int, **kwargs: Any) -> nx.Graph:
        return nx.path_graph(n=n_nodes)


class CycleGraphGenerator(GraphGenerator):
    @property
    def name(self) -> str:
        return "cycle"

    def generate(self, n_nodes: int, **kwargs: Any) -> nx.Graph:
        return nx.cycle_graph(n=n_nodes)


class GraphGeneratorFactory:
    _generators = {
        "random": RandomGraphGenerator(),
        "scale_free": ScaleFreeGraphGenerator(),
        "small_world": SmallWorldGraphGenerator(),
        "complete": CompleteGraphGenerator(),
        "star": StarGraphGenerator(),
        "path": PathGraphGenerator(),
        "cycle": CycleGraphGenerator(),
    }

    @classmethod
    def get_generator(cls, name: str) -> GraphGenerator:
        if name not in cls._generators:
            available = ", ".join(cls._generators.keys())
            raise ValueError(f"Unknown graph generator '{name}'. Available options: {available}")

        return cls._generators.get(name, None)

    @classmethod
    def list_generators(cls) -> list[str]:
        return list(cls._generators.keys())
