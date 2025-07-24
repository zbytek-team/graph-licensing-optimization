from typing import List
from src.core.types import LicenseType


class LicenseConfigFactory:
    @classmethod
    def get_config(cls, name: str) -> List[LicenseType]:
        configs = {
            "duolingo_super": cls._duolingo_super_config,
            "spotify": cls._spotify_config,
            "roman_domination": cls._roman_domination_config,
        }

        if name not in configs:
            available = ", ".join(configs.keys())
            raise ValueError(f"Unsupported license config: {name}. Available: {available}")

        return configs[name]()

    @staticmethod
    def _duolingo_super_config() -> List[LicenseType]:
        return [
            LicenseType(name="Individual", cost=13.99, min_capacity=1, max_capacity=1, color="#388B00"),
            LicenseType(name="Family", cost=29.17, min_capacity=2, max_capacity=6, color="#006394"),
        ]

    @staticmethod
    def _spotify_config() -> List[LicenseType]:
        return [
            LicenseType(name="Individual", cost=23.99, min_capacity=1, max_capacity=1, color="#91b924"),
            LicenseType(name="Duo", cost=30.99, min_capacity=2, max_capacity=2, color="#00805c"),
            LicenseType(name="Family", cost=37.99, min_capacity=2, max_capacity=6, color="#7b004c"),
        ]

    @staticmethod
    def _roman_domination_config() -> List[LicenseType]:
        return [
            LicenseType(name="Solo", cost=1.0, min_capacity=1, max_capacity=1, color="#d38600"),
            LicenseType(name="Group", cost=2.0, min_capacity=2, max_capacity=999, color="#940020"),
        ]
