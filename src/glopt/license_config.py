from collections.abc import Callable
from typing import ClassVar

from .core import LicenseType


class LicenseConfigFactory:
    PURPLE = "#542f82"
    GOLD = "#cb8a35"
    GREEN = "#5d9f49"

    _CONFIGS: ClassVar[dict[str, Callable[[], list[LicenseType]]]] = {
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
            LicenseType("Group", 2.0, 2, 99999, LicenseConfigFactory.GOLD),
        ],
    }

    @classmethod
    def get_config(cls, name: str) -> list[LicenseType]:
        # Dynamic roman domination sweep: roman_p_1_5 or roman_p:2.5
        if name.startswith("roman_p_") or name.startswith("roman_p:"):
            p_str = name.split("_", 2)[2] if name.startswith("roman_p_") else name.split(":", 1)[1]
            p_str = p_str.replace("_", ".")
            try:
                p_val = float(p_str)
            except Exception:
                available = ", ".join(cls._CONFIGS.keys())
                raise ValueError(
                    f"Invalid roman price '{name}'. Available: {available} or roman_p_<x_y>/roman_p:<x.y>"
                )
            return [
                LicenseType("Solo", 1.0, 1, 1, cls.PURPLE),
                LicenseType("Group", p_val, 2, 99999, cls.GOLD),
            ]
        try:
            return cls._CONFIGS[name]()
        except KeyError:
            available = ", ".join(cls._CONFIGS.keys())
            msg = f"Unsupported license config: {name}. Available: {available} or roman_p_<x_y>/roman_p:<x.y>"
            raise ValueError(msg) from None
