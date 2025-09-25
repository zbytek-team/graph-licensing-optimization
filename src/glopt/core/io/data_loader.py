from pathlib import Path

import networkx as nx


class RealWorldDataLoader:
    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)

    def load_facebook_ego_network(self, ego_id: str) -> nx.Graph:
        facebook_dir = self.data_dir / "facebook"
        edges_file = facebook_dir / f"{ego_id}.edges"
        if not edges_file.exists():
            msg = f"Plik edges nie istnieje: {edges_file}"
            raise FileNotFoundError(msg)
        graph = nx.Graph()
        ego_node = int(ego_id)
        graph.add_node(ego_node, is_ego=True)
        with edges_file.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        node1, node2 = (int(parts[0]), int(parts[1]))
                        graph.add_edge(node1, node2)
                        graph.add_edge(ego_node, node1)
                        graph.add_edge(ego_node, node2)
        self._load_node_features(graph, facebook_dir, ego_id)
        self._load_circles(graph, facebook_dir, ego_id)
        return graph

    def load_all_facebook_networks(self) -> dict[str, nx.Graph]:
        facebook_dir = self.data_dir / "facebook"
        networks = {}
        if not facebook_dir.exists():
            msg = f"Katalog Facebook nie istnieje: {facebook_dir}"
            raise FileNotFoundError(msg)
        edge_files = list(facebook_dir.glob("*.edges"))
        for edge_file in edge_files:
            ego_id = edge_file.stem
            try:
                network = self.load_facebook_ego_network(ego_id)
                networks[ego_id] = network
            except Exception:
                pass
        return networks

    def _load_node_features(
        self, graph: nx.Graph, data_dir: Path, ego_id: str
    ) -> None:
        feat_file = data_dir / f"{ego_id}.feat"
        egofeat_file = data_dir / f"{ego_id}.egofeat"
        featnames_file = data_dir / f"{ego_id}.featnames"
        feature_names = []
        if featnames_file.exists():
            with featnames_file.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(maxsplit=1)
                        if len(parts) >= 2:
                            feature_names.append(parts[1])
        if feat_file.exists():
            with feat_file.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            node_id = int(parts[0])
                            features = [int(x) for x in parts[1:]]
                            if node_id in graph.nodes():
                                graph.nodes[node_id]["features"] = features
                                graph.nodes[node_id]["feature_count"] = sum(
                                    features
                                )
        ego_node = int(ego_id)
        if egofeat_file.exists() and ego_node in graph.nodes():
            with egofeat_file.open() as f:
                line = f.readline().strip()
                if line:
                    features = [int(x) for x in line.split()]
                    graph.nodes[ego_node]["features"] = features
                    graph.nodes[ego_node]["feature_count"] = sum(features)

    def _load_circles(
        self, graph: nx.Graph, data_dir: Path, ego_id: str
    ) -> None:
        circles_file = data_dir / f"{ego_id}.circles"
        if not circles_file.exists():
            return
        circles = []
        with circles_file.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        circle_name = parts[0]
                        circle_members = [
                            int(x) for x in parts[1:] if x.isdigit()
                        ]
                        circles.append(
                            {
                                "name": circle_name,
                                "members": circle_members,
                                "size": len(circle_members),
                            }
                        )
        for node_id in graph.nodes():
            node_circles = []
            for i, circle in enumerate(circles):
                if node_id in circle["members"]:
                    node_circles.append(i)
            graph.nodes[node_id]["circles"] = node_circles
        graph.graph["circles"] = circles
