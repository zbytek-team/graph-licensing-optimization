import random
from pathlib import Path
from typing import Optional

import networkx as nx


class FacebookDataLoader:
    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            data_dir = project_root / "data" / "facebook"
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Facebook data directory not found: {self.data_dir}")

    def get_available_ego_networks(self) -> list[str]:
        edge_files = list(self.data_dir.glob("*.edges"))
        ego_ids = [f.stem for f in edge_files]
        return sorted(ego_ids, key=int)

    def load_ego_network(self, ego_id: str) -> nx.Graph:
        edges_file = self.data_dir / f"{ego_id}.edges"
        if not edges_file.exists():
            available = self.get_available_ego_networks()
            raise FileNotFoundError(f"Ego network '{ego_id}' not found. Available networks: {available}")
        graph = nx.Graph()
        ego_node = int(ego_id)
        graph.add_node(ego_node)
        with open(edges_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        node1, node2 = map(int, line.split())
                        graph.add_edge(node1, node2)
                    except ValueError:
                        continue
        for node in graph.nodes():
            if node != ego_node:
                graph.add_edge(ego_node, node)
        components = list(nx.connected_components(graph))
        if len(components) > 1:
            ego_component = None
            for component in components:
                if ego_node in component:
                    ego_component = component
                    break
            for component in components:
                if component != ego_component and component:
                    random_node = next(iter(component))
                    graph.add_edge(ego_node, random_node)
        return graph

    def load_random_ego_network(self, seed: Optional[int] = None) -> tuple[nx.Graph, str]:
        if seed is not None:
            random.seed(seed)
        available_networks = self.get_available_ego_networks()
        if not available_networks:
            raise ValueError("No ego networks available")
        ego_id = random.choice(available_networks)
        graph = self.load_ego_network(ego_id)
        return graph, ego_id

    def load_combined_network(self, max_networks: Optional[int] = None, seed: Optional[int] = None) -> nx.Graph:
        if seed is not None:
            random.seed(seed)
        available_networks = self.get_available_ego_networks()
        if not available_networks:
            raise ValueError("No ego networks available")
        if max_networks is not None:
            available_networks = random.sample(available_networks, min(max_networks, len(available_networks)))
        combined_graph = nx.Graph()
        node_offset = 0
        for ego_id in available_networks:
            ego_graph = self.load_ego_network(ego_id)
            mapping = {node: node + node_offset for node in ego_graph.nodes()}
            relabeled_graph = nx.relabel_nodes(ego_graph, mapping)
            combined_graph = nx.compose(combined_graph, relabeled_graph)
            node_offset = max(combined_graph.nodes()) + 1 if combined_graph.nodes() else 0
        return combined_graph

    def get_network_info(self, ego_id: str) -> dict:
        graph = self.load_ego_network(ego_id)
        return {
            "ego_id": ego_id,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "average_clustering": nx.average_clustering(graph),
            "average_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes()
            if graph.number_of_nodes() > 0
            else 0,
        }

    def get_all_networks_info(self) -> list[dict]:
        networks_info = []
        for ego_id in self.get_available_ego_networks():
            try:
                info = self.get_network_info(ego_id)
                networks_info.append(info)
            except Exception as e:
                print(f"Warning: Could not load network {ego_id}: {e}")
                continue
        return networks_info
