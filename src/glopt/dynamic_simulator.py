import csv
import random
from dataclasses import dataclass
from typing import Any

import networkx as nx

from .algorithms import GreedyAlgorithm
from .core import Algorithm, LicenseGroup, LicenseType, Solution
from .core.solution_builder import SolutionBuilder


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
    # Realistic dynamics modes
    # nodes: 'random' (uniform attachment) | 'preferential' (degree-proportional)
    mode_nodes: str = "random"
    # edges: 'random' | 'preferential' (by degree product) | 'triadic' (closure) | 'rewire_ws' (Watts--Strogatz style)
    mode_edges: str = "random"
    # parameters for modes
    add_node_attach_m: int = 2  # how many neighbors to attach per new node (avg)
    triadic_trials: int = 20    # attempts per added edge in triadic closure
    rewire_prob: float = 0.1    # probability per rewired edge in 'rewire_ws'


@dataclass
class DynamicStep:
    step_number: int
    graph: nx.Graph
    solution: Solution
    mutations_applied: list[str]
    rebalance_cost_change: float


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
        self.history: list[DynamicStep] = []
        self.next_node_id = 0

        if seed is not None:
            random.seed(seed)

    def simulate(
        self,
        initial_graph: nx.Graph,
        license_types: list[LicenseType],
        num_steps: int = 10,
        initial_algorithm: Algorithm | None = None,
    ) -> list[DynamicStep]:
        self.history.clear()
        current_graph = initial_graph.copy()
        self.next_node_id = max(current_graph.nodes()) + 1 if current_graph.nodes() else 0

        if initial_algorithm is None:
            initial_algorithm = self.rebalance_algorithm

        current_solution = initial_algorithm.solve(current_graph, license_types)

        self.history.append(
            DynamicStep(
                step_number=0,
                graph=current_graph.copy(),
                solution=current_solution,
                mutations_applied=[],
                rebalance_cost_change=0.0,
            )
        )

        for step in range(1, num_steps + 1):
            mutations_applied = []

            current_graph, step_mutations = self._apply_mutations(current_graph)
            mutations_applied.extend(step_mutations)

            old_cost = current_solution.total_cost
            current_solution = self._rebalance_licenses(current_graph, license_types, current_solution)
            new_cost = current_solution.total_cost
            cost_change = new_cost - old_cost

            self.history.append(
                DynamicStep(
                    step_number=step,
                    graph=current_graph.copy(),
                    solution=current_solution,
                    mutations_applied=mutations_applied,
                    rebalance_cost_change=cost_change,
                ),
            )

        return self.history

    def _apply_mutations(self, graph: nx.Graph) -> tuple[nx.Graph, list[str]]:
        mutations = []

        if random.random() < self.mutation_params.add_nodes_prob:
            num_add = random.randint(1, self.mutation_params.max_nodes_add)
            new_nodes = self._add_nodes(graph, num_add)
            mutations.append(f"Added nodes: {new_nodes}")

        if random.random() < self.mutation_params.remove_nodes_prob and graph.number_of_nodes() > 5:
            num_remove = random.randint(1, min(self.mutation_params.max_nodes_remove, graph.number_of_nodes() - 5))
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

        if random.random() < self.mutation_params.remove_edges_prob and graph.number_of_edges() > 0:
            num_remove = random.randint(1, min(self.mutation_params.max_edges_remove, graph.number_of_edges()))
            removed_edges = self._remove_edges(graph, num_remove)
            mutations.append(f"Removed {len(removed_edges)} edges")

        return graph, mutations

    def _add_nodes(self, graph: nx.Graph, num_nodes: int) -> list[int]:
        new_nodes = []
        existing_nodes = list(graph.nodes())

        for _ in range(num_nodes):
            new_node = self.next_node_id
            self.next_node_id += 1
            graph.add_node(new_node)
            new_nodes.append(new_node)
            if existing_nodes:
                m = max(1, min(self.mutation_params.add_node_attach_m, len(existing_nodes)))
                if self.mutation_params.mode_nodes == "preferential":
                    # choose neighbors with prob ~ degree+1
                    deg = graph.degree
                    weights = [deg[v] + 1 for v in existing_nodes]
                    total = sum(weights)
                    chosen: set[int] = set()
                    # sample without replacement
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
                        # remove picked to avoid duplicates
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

    def _add_edges(self, graph: nx.Graph, num_edges: int) -> list[tuple[int, int]]:
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
            # pick pairs with prob ~ (deg(u)+1)*(deg(v)+1)
            attempts = 0
            while len(added_edges) < num_edges and attempts < num_edges * 20:
                u, v = random.sample(nodes, 2)
                if graph.has_edge(u, v):
                    attempts += 1
                    continue
                deg = graph.degree
                w = (deg[u] + 1) * (deg[v] + 1)
                # normalize by an upper bound; accept with prob min(1, w/W)
                # choose W as (max_deg+1)^2
                max_deg = max((deg[x] for x in nodes), default=1)
                accept_p = min(1.0, w / float((max_deg + 1) ** 2))
                if random.random() < accept_p:
                    graph.add_edge(u, v)
                    added_edges.append((u, v))
                attempts += 1
        elif mode == "triadic":
            trials = max(1, self.mutation_params.triadic_trials)
            attempts = 0
            while len(added_edges) < num_edges and attempts < num_edges * trials:
                w = random.choice(nodes)
                neigh = list(graph.neighbors(w))
                if len(neigh) >= 2:
                    u, v = random.sample(neigh, 2)
                    if not graph.has_edge(u, v):
                        graph.add_edge(u, v)
                        added_edges.append((u, v))
                attempts += 1
        else:
            # fallback to random
            attempts = 0
            while len(added_edges) < num_edges and attempts < num_edges * 10:
                u, v = random.sample(nodes, 2)
                if not graph.has_edge(u, v):
                    graph.add_edge(u, v)
                    added_edges.append((u, v))
                attempts += 1

        return added_edges

    def _rewire_edges(self, graph: nx.Graph, num_ops: int) -> tuple[int, int]:
        # Perform up to num_ops rewirings: remove a random existing edge and add a new one from its endpoint to random other node
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
            # choose endpoint to rewire from
            a = u if random.random() < 0.5 else v
            # try a few attempts to connect to a non-neighbor
            for _ in range(6):
                b = random.choice(nodes)
                if b != a and not graph.has_edge(a, b):
                    graph.add_edge(a, b)
                    added += 1
                    break
            ops += 1
            edges = list(graph.edges())
        return (added, removed)

    def _remove_edges(self, graph: nx.Graph, num_edges: int) -> list[tuple[int, int]]:
        edges_to_remove = random.sample(list(graph.edges()), num_edges)

        for edge in edges_to_remove:
            graph.remove_edge(*edge)

        return edges_to_remove

    def _rebalance_licenses(self, graph: nx.Graph, license_types: list[LicenseType], old_solution: Solution) -> Solution:
        existing_nodes = set(graph.nodes())

        valid_groups = []
        uncovered_nodes = set(existing_nodes)

        for group in old_solution.groups:
            group_nodes = group.all_members

            if not group_nodes.issubset(existing_nodes):
                continue

            if self._is_group_valid(graph, group):
                valid_groups.append(group)
                uncovered_nodes -= group_nodes

        if uncovered_nodes:
            subgraph = graph.subgraph(uncovered_nodes).copy()

            if subgraph.number_of_nodes() > 0:
                new_solution = self.rebalance_algorithm.solve(subgraph, license_types)
                valid_groups.extend(new_solution.groups)

        return SolutionBuilder.create_solution_from_groups(valid_groups)

    def _is_group_valid(self, graph: nx.Graph, group: LicenseGroup) -> bool:
        owner = group.owner
        additional_members = group.additional_members

        if owner not in graph.nodes():
            return False

        for member in additional_members:
            if member not in graph.nodes():
                return False
            if not graph.has_edge(owner, member):
                return False

        group_size = group.size
        license_type = group.license_type

        return license_type.min_capacity <= group_size <= license_type.max_capacity

    def get_simulation_summary(self) -> dict[str, Any]:
        if not self.history:
            return {}

        total_cost_changes = sum(step.rebalance_cost_change for step in self.history[1:])
        initial_cost = self.history[0].solution.total_cost
        final_cost = self.history[-1].solution.total_cost

        node_changes = []
        edge_changes = []

        for step in self.history[1:]:
            for mutation in step.mutations_applied:
                if "nodes" in mutation:
                    node_changes.append(mutation)
                elif "edges" in mutation:
                    edge_changes.append(mutation)

        return {
            "initial_cost": initial_cost,
            "final_cost": final_cost,
            "total_cost_change": total_cost_changes,
            "cost_efficiency": final_cost / initial_cost if initial_cost > 0 else 1.0,
            "num_steps": len(self.history) - 1,
            "node_mutations": len(node_changes),
            "edge_mutations": len(edge_changes),
            "avg_cost_change_per_step": total_cost_changes / max(1, len(self.history) - 1),
        }

    def export_history_to_csv(self, filename: str) -> None:
        fieldnames = ["step", "nodes", "edges", "cost", "groups", "cost_change", "mutations"]
        from pathlib import Path

        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for step in self.history:
                writer.writerow(
                    {
                        "step": step.step_number,
                        "nodes": step.graph.number_of_nodes(),
                        "edges": step.graph.number_of_edges(),
                        "cost": step.solution.total_cost,
                        "groups": len(step.solution.groups),
                        "cost_change": step.rebalance_cost_change,
                        "mutations": "; ".join(step.mutations_applied),
                    }
                )

    # DynamicScenarioFactory was unused; removed
