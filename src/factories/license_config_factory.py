from typing import List, Callable, Dict
from ..core.models import LicenseType


class LicenseConfigFactory:
    PURPLE = "#542f82"
    GOLD = "#cb8a35"
    GREEN = "#5d9f49"

    _CONFIGS: Dict[str, Callable[[], List[LicenseType]]] = {
        "duolingo_super": lambda: [
            LicenseType("Individual", 13.99, 1, 1, LicenseConfigFactory.PURPLE),
            LicenseType("Family", 29.17, 2, 6, LicenseConfigFactory.GOLD),
        ],
        "spotify": lambda: [
            LicenseType("Individual", 23.99, 1, 1, LicenseConfigFactory.PURPLE),
            LicenseType("Duo", 30.99, 2, 2, LicenseConfigFactory.GREEN),
            LicenseType("Family", 37.99, 2, 6, LicenseConfigFactory.GOLD),
        ],
        "roman_domination": lambda: [
            LicenseType("Solo", 1.0, 1, 1, LicenseConfigFactory.PURPLE),
            LicenseType("Group", 2.0, 2, 999, LicenseConfigFactory.GOLD),
        ],
    }

    @classmethod
    def get_config(cls, name: str) -> List[LicenseType]:
        try:
            return cls._CONFIGS[name]()
        except KeyError:
            available = ", ".join(cls._CONFIGS.keys())
            raise ValueError(f"Unsupported license config: {name}. Available: {available}")
