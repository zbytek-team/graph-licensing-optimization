from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    import networkx as nx


@dataclass
class LicenseTypeConfig:
    price: float
    min_size: int
    max_size: int

    def is_valid_size(self, size: int) -> bool:
        return self.min_size <= size <= self.max_size

    def cost_per_person(self, size: int) -> float:
        if not self.is_valid_size(size):
            return float("inf")
        return self.price / size


@dataclass
class LicenseConfig:
    license_types: Dict[str, LicenseTypeConfig]

    @classmethod
    def create_flexible(cls, license_configs: Dict[str, Dict[str, Any]]):
        license_types = {}
        for name, config in license_configs.items():
            license_types[name] = LicenseTypeConfig(
                price=config["price"], min_size=config["min_size"], max_size=config["max_size"]
            )
        return cls(license_types=license_types)

    def get_best_license_for_size(self, size: int) -> tuple[str, LicenseTypeConfig] | None:
        best_license = None
        best_cost_per_person = float("inf")
        for name, license_type in self.license_types.items():
            if license_type.is_valid_size(size):
                cost_per_person = license_type.cost_per_person(size)
                if cost_per_person < best_cost_per_person:
                    best_cost_per_person = cost_per_person
                    best_license = (name, license_type)
        return best_license

    def is_size_beneficial(self, license_name: str, size: int) -> bool:
        if license_name not in self.license_types:
            return False
        license_type = self.license_types[license_name]
        if not license_type.is_valid_size(size):
            return False
        best_alternative = self.get_best_license_for_size(size)
        if best_alternative is None:
            return True
        return license_type.cost_per_person(size) <= best_alternative[1].cost_per_person(size)


@dataclass
class LicenseSolution:
    licenses: Dict[str, Dict[int, List[int]]]

    def __post_init__(self) -> None:
        for license_type, groups in self.licenses.items():
            for owner_id, members in groups.items():
                if owner_id not in members:
                    members.insert(0, owner_id)

    def get_node_license_info(self, node_id: int) -> tuple[str, int] | None:
        for license_type, groups in self.licenses.items():
            for owner_id, members in groups.items():
                if node_id in members:
                    return (license_type, owner_id)
        return None

    def get_all_nodes(self) -> List[int]:
        all_nodes = []
        for license_type, groups in self.licenses.items():
            for owner_id, members in groups.items():
                all_nodes.extend(members)
        return list(set(all_nodes))

    def calculate_cost(self, config: LicenseConfig) -> float:
        total_cost = 0.0
        for license_type, groups in self.licenses.items():
            if license_type not in config.license_types:
                continue
            license_config = config.license_types[license_type]
            for owner_id, members in groups.items():
                group_size = len(members)
                if license_config.is_valid_size(group_size):
                    total_cost += license_config.price
                else:
                    total_cost += float("inf")
        return total_cost

    def is_valid(self, graph: "nx.Graph", config: "LicenseConfig") -> bool:
        try:
            graph_nodes = set(graph.nodes())
            solution_nodes = set(self.get_all_nodes())
            if graph_nodes != solution_nodes:
                return False
            for license_type, groups in self.licenses.items():
                if license_type not in config.license_types:
                    return False
                license_config = config.license_types[license_type]
                for owner_id, members in groups.items():
                    if not license_config.is_valid_size(len(members)):
                        return False
                    if owner_id not in graph_nodes:
                        return False
                    if len(members) > 1:
                        for member in members:
                            if member != owner_id and not graph.has_edge(owner_id, member):
                                return False
            return True
        except Exception:
            return False

    @classmethod
    def create_empty(cls) -> "LicenseSolution":
        return cls(licenses={})
