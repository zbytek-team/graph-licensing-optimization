import random
from dataclasses import dataclass

import networkx as nx

from .algorithms import GreedyAlgorithm
from .models import Algorithm


@dataclass
class MutationParams:
    add_nodes_prob: float = 0.1
    remove_nodes_prob: float = 0.05
    add_edges_prob: float = 0.15
    remove_edges_prob: float = 0.1
    max_nodes_add: int = 3
    max_nodes_remove: int = 2
    max_edges_add: int = 5
    max_edges_remove: int = 3
    mode_nodes: str = "random"
    mode_edges: str = "random"
    add_node_attach_m: int = 2
    triadic_trials: int = 20


class DynamicNetworkSimulator:
    def __init__(
        self,
        rebalance_algorithm: Algorithm | None = None,
        mutation_params: MutationParams | None = None,
        seed: int | None = None,
    ) -> None:
        self.rebalance_algorithm = rebalance_algorithm or GreedyAlgorithm()
        self.mutation_params = mutation_params or MutationParams()
        self.seed = seed
        self.next_node_id = 0
        if seed is not None:
            random.seed(seed)

    def _apply_mutations(self, graph: nx.Graph) -> tuple[nx.Graph, list[str]]:
        mutations = []
        if random.random() < self.mutation_params.add_nodes_prob:
            num_add = random.randint(1, self.mutation_params.max_nodes_add)
            new_nodes = self._add_nodes(graph, num_add)
            mutations.append(f"Added nodes: {new_nodes}")
        if (
            random.random() < self.mutation_params.remove_nodes_prob
            and graph.number_of_nodes() > 5
        ):
            num_remove = random.randint(
                1,
                min(
                    self.mutation_params.max_nodes_remove,
                    graph.number_of_nodes() - 5,
                ),
            )
            removed_nodes = self._remove_nodes(graph, num_remove)
            mutations.append(f"Removed nodes: {removed_nodes}")
        if random.random() < self.mutation_params.add_edges_prob:
            num_add = random.randint(1, self.mutation_params.max_edges_add)
            if self.mutation_params.mode_edges == "rewire_ws":
                addc, removec = self._rewire_edges(graph, num_add)
                if addc:
                    mutations.append(f"Added {addc} edges")
                if removec:
                    mutations.append(f"Removed {removec} edges")
            else:
                added_edges = self._add_edges(graph, num_add)
                mutations.append(f"Added {len(added_edges)} edges")
        if (
            random.random() < self.mutation_params.remove_edges_prob
            and graph.number_of_edges() > 0
        ):
            num_remove = random.randint(
                1,
                min(
                    self.mutation_params.max_edges_remove,
                    graph.number_of_edges(),
                ),
            )
            removed_edges = self._remove_edges(graph, num_remove)
            mutations.append(f"Removed {len(removed_edges)} edges")
        return (graph, mutations)

    def _add_nodes(self, graph: nx.Graph, num_nodes: int) -> list[int]:
        new_nodes = []
        existing_nodes = list(graph.nodes())
        for _ in range(num_nodes):
            new_node = self.next_node_id
            self.next_node_id += 1
            graph.add_node(new_node)
            new_nodes.append(new_node)
            if existing_nodes:
                m = max(
                    1,
                    min(
                        self.mutation_params.add_node_attach_m,
                        len(existing_nodes),
                    ),
                )
                if self.mutation_params.mode_nodes == "preferential":
                    deg = graph.degree
                    weights = [deg[v] + 1 for v in existing_nodes]
                    total = sum(weights)
                    chosen: set[int] = set()
                    for _k in range(m):
                        if not existing_nodes:
                            break
                        r = random.uniform(0, total)
                        acc = 0.0
                        pick_idx = 0
                        for idx, w in enumerate(weights):
                            acc += w
                            if acc >= r:
                                pick_idx = idx
                                break
                        v = existing_nodes[pick_idx]
                        if v not in chosen:
                            graph.add_edge(new_node, v)
                            chosen.add(v)
                        total -= weights[pick_idx]
                        existing_nodes.pop(pick_idx)
                        weights.pop(pick_idx)
                else:
                    num_connections = random.randint(1, m)
                    neighbors = random.sample(existing_nodes, num_connections)
                    for neighbor in neighbors:
                        graph.add_edge(new_node, neighbor)
        return new_nodes

    def _remove_nodes(self, graph: nx.Graph, num_nodes: int) -> list[int]:
        nodes_to_remove = random.sample(list(graph.nodes()), num_nodes)
        for node in nodes_to_remove:
            graph.remove_node(node)
        return nodes_to_remove

    def _add_edges(
        self, graph: nx.Graph, num_edges: int
    ) -> list[tuple[int, int]]:
        nodes = list(graph.nodes())
        added_edges = []
        if len(nodes) < 2:
            return added_edges
        mode = self.mutation_params.mode_edges
        if mode == "random":
            attempts = 0
            while len(added_edges) < num_edges and attempts < num_edges * 10:
                u, v = random.sample(nodes, 2)
                if not graph.has_edge(u, v):
                    graph.add_edge(u, v)
                    added_edges.append((u, v))
                attempts += 1
        elif mode == "preferential":
            attempts = 0
            while len(added_edges) < num_edges and attempts < num_edges * 20:
                u, v = random.sample(nodes, 2)
                if graph.has_edge(u, v):
                    attempts += 1
                    continue
                deg = graph.degree
                w = (deg[u] + 1) * (deg[v] + 1)
                max_deg = max((deg[x] for x in nodes), default=1)
                accept_p = min(1.0, w / float((max_deg + 1) ** 2))
                if random.random() < accept_p:
                    graph.add_edge(u, v)
                    added_edges.append((u, v))
                attempts += 1
        elif mode == "triadic":
            trials = max(1, self.mutation_params.triadic_trials)
            attempts = 0
            while (
                len(added_edges) < num_edges and attempts < num_edges * trials
            ):
                w = random.choice(nodes)
                neigh = list(graph.neighbors(w))
                if len(neigh) >= 2:
                    u, v = random.sample(neigh, 2)
                    if not graph.has_edge(u, v):
                        graph.add_edge(u, v)
                        added_edges.append((u, v))
                attempts += 1
        else:
            attempts = 0
            while len(added_edges) < num_edges and attempts < num_edges * 10:
                u, v = random.sample(nodes, 2)
                if not graph.has_edge(u, v):
                    graph.add_edge(u, v)
                    added_edges.append((u, v))
                attempts += 1
        return added_edges

    def _rewire_edges(self, graph: nx.Graph, num_ops: int) -> tuple[int, int]:
        added = 0
        removed = 0
        edges = list(graph.edges())
        nodes = list(graph.nodes())
        if not edges or len(nodes) < 3:
            return (0, 0)
        ops = 0
        while ops < num_ops and edges:
            u, v = random.choice(edges)
            if graph.has_edge(u, v):
                graph.remove_edge(u, v)
                removed += 1
            a = u if random.random() < 0.5 else v
            for _ in range(6):
                b = random.choice(nodes)
                if b != a and (not graph.has_edge(a, b)):
                    graph.add_edge(a, b)
                    added += 1
                    break
            ops += 1
            edges = list(graph.edges())
        return (added, removed)

    def _remove_edges(
        self, graph: nx.Graph, num_edges: int
    ) -> list[tuple[int, int]]:
        edges_to_remove = random.sample(list(graph.edges()), num_edges)
        for edge in edges_to_remove:
            graph.remove_edge(*edge)
        return edges_to_remove
