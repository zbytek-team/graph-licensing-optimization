"""Facebook dataset loader for social network graphs."""

import random
from pathlib import Path
from typing import Optional

import networkx as nx


class FacebookDataLoader:
    """Loader for Facebook social network datasets."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the Facebook data loader.
        
        Args:
            data_dir: Path to the directory containing Facebook data files.
                     If None, uses default path.
        """
        if data_dir is None:
            # Default to data/facebook relative to project root
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            data_dir = project_root / "data" / "facebook"
        
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Facebook data directory not found: {self.data_dir}")
    
    def get_available_ego_networks(self) -> list[str]:
        """Get list of available ego network IDs.
        
        Returns:
            List of ego network IDs (e.g., ['0', '107', '1684', ...])
        """
        edge_files = list(self.data_dir.glob("*.edges"))
        ego_ids = [f.stem for f in edge_files]
        return sorted(ego_ids, key=int)
    
    def load_ego_network(self, ego_id: str) -> nx.Graph:
        """Load a specific ego network by ID.
        
        Args:
            ego_id: The ego network ID (e.g., '0', '107', etc.)
        
        Returns:
            NetworkX graph representing the ego network.
            
        Raises:
            FileNotFoundError: If the edge file for the ego network doesn't exist.
        """
        edges_file = self.data_dir / f"{ego_id}.edges"
        
        if not edges_file.exists():
            available = self.get_available_ego_networks()
            raise FileNotFoundError(
                f"Ego network '{ego_id}' not found. Available networks: {available}"
            )
        
        # Create graph
        graph = nx.Graph()
        
        # Add ego node (the central node)
        ego_node = int(ego_id)
        graph.add_node(ego_node)
        
        # Read edges
        with open(edges_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        node1, node2 = map(int, line.split())
                        graph.add_edge(node1, node2)
                    except ValueError:
                        continue  # Skip malformed lines
        
        # Add edges from ego to all other nodes (ego networks assumption)
        # In Facebook ego networks, the ego is connected to all other nodes
        for node in graph.nodes():
            if node != ego_node:
                graph.add_edge(ego_node, node)
        
        # Ensure the graph is connected (add ego as hub if isolated nodes exist)
        components = list(nx.connected_components(graph))
        if len(components) > 1:
            # Find the component containing ego
            ego_component = None
            for component in components:
                if ego_node in component:
                    ego_component = component
                    break
            
            # Connect ego to all isolated components
            for component in components:
                if component != ego_component and component:
                    # Connect ego to a random node in this component
                    random_node = next(iter(component))
                    graph.add_edge(ego_node, random_node)
        
        return graph
    
    def load_random_ego_network(self, seed: Optional[int] = None) -> tuple[nx.Graph, str]:
        """Load a random ego network.
        
        Args:
            seed: Random seed for reproducibility.
        
        Returns:
            Tuple of (graph, ego_id)
        """
        if seed is not None:
            random.seed(seed)
        
        available_networks = self.get_available_ego_networks()
        if not available_networks:
            raise ValueError("No ego networks available")
        
        ego_id = random.choice(available_networks)
        graph = self.load_ego_network(ego_id)
        
        return graph, ego_id
    
    def load_combined_network(self, max_networks: Optional[int] = None, seed: Optional[int] = None) -> nx.Graph:
        """Load and combine multiple ego networks into a single graph.
        
        Args:
            max_networks: Maximum number of networks to combine. If None, use all.
            seed: Random seed for reproducibility.
        
        Returns:
            Combined NetworkX graph.
        """
        if seed is not None:
            random.seed(seed)
        
        available_networks = self.get_available_ego_networks()
        if not available_networks:
            raise ValueError("No ego networks available")
        
        if max_networks is not None:
            available_networks = random.sample(
                available_networks, 
                min(max_networks, len(available_networks))
            )
        
        # Create combined graph
        combined_graph = nx.Graph()
        node_offset = 0
        
        for ego_id in available_networks:
            ego_graph = self.load_ego_network(ego_id)
            
            # Relabel nodes to avoid conflicts
            mapping = {node: node + node_offset for node in ego_graph.nodes()}
            relabeled_graph = nx.relabel_nodes(ego_graph, mapping)
            
            # Add to combined graph
            combined_graph = nx.compose(combined_graph, relabeled_graph)
            
            # Update offset for next network
            node_offset = max(combined_graph.nodes()) + 1 if combined_graph.nodes() else 0
        
        return combined_graph
    
    def get_network_info(self, ego_id: str) -> dict:
        """Get information about a specific ego network.
        
        Args:
            ego_id: The ego network ID.
        
        Returns:
            Dictionary with network statistics.
        """
        graph = self.load_ego_network(ego_id)
        
        return {
            "ego_id": ego_id,
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "average_clustering": nx.average_clustering(graph),
            "average_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes() if graph.number_of_nodes() > 0 else 0,
        }
    
    def get_all_networks_info(self) -> list[dict]:
        """Get information about all available ego networks.
        
        Returns:
            List of dictionaries with network statistics.
        """
        networks_info = []
        for ego_id in self.get_available_ego_networks():
            try:
                info = self.get_network_info(ego_id)
                networks_info.append(info)
            except Exception as e:
                print(f"Warning: Could not load network {ego_id}: {e}")
                continue
        
        return networks_info
