from collections.abc import Callable
from typing import ClassVar

from .models import LicenseType


class LicenseConfigFactory:
    RED = "#B91C1C"
    BLUE = "#1E3A8A"
    GREEN = "#065F46"
    BLACK = "#000000"
    _CONFIGS: ClassVar[dict[str, Callable[[], list[LicenseType]]]] = {
        "duolingo_super": lambda: [
            LicenseType("Individual", 13.99, 1, 1, LicenseConfigFactory.BLACK),
            LicenseType("Family", 29.17, 2, 6, LicenseConfigFactory.BLACK),
        ],
        "spotify": lambda: [
            LicenseType("Individual", 23.99, 1, 1, LicenseConfigFactory.RED),
            LicenseType("Duo", 30.99, 2, 2, LicenseConfigFactory.GREEN),
            LicenseType("Family", 37.99, 2, 6, LicenseConfigFactory.BLUE),
        ],
        "netflix": lambda: [
            LicenseType("Basic", 33, 1, 1, LicenseConfigFactory.RED),
            LicenseType("Standard", 49, 1, 2, LicenseConfigFactory.GREEN),
            LicenseType("Premium", 67, 1, 4, color=LicenseConfigFactory.BLUE),
        ],
        "roman_domination": lambda: [
            LicenseType("Solo", 1.0, 1, 1, LicenseConfigFactory.BLUE),
            LicenseType("Group", 2.0, 2, 999999, LicenseConfigFactory.RED),
        ],
    }

    @classmethod
    def get_config(cls, name: str) -> list[LicenseType]:
        if name.startswith("roman_p_"):
            p_str = name.split("_", 2)[2]
            p_str = p_str.replace("_", ".")
            try:
                p_val = float(p_str)
            except Exception:
                available = ", ".join(cls._CONFIGS.keys())
                raise ValueError(
                    f"Invalid roman price '{name}'. " \
                    f"Available: {available} or roman_p_<x_y>"
                )
            return [
                LicenseType("Solo", 1.0, 1, 1, cls.BLUE),
                LicenseType("Group", p_val, 2, 999999, cls.RED),
            ]
        if name.startswith("duolingo_p_"):
            p_str = name.split("_", 2)[2]
            p_str = p_str.replace("_", ".")
            try:
                p_val = float(p_str)
            except Exception:
                available = ", ".join(cls._CONFIGS.keys())
                raise ValueError(
                    f"Invalid duolingo price '{name}'. " \
                    f"Available: {available} or duolingo_p_<x_y>"
                )
            return [
                LicenseType("Individual", 1.0, 1, 1, cls.RED),
                LicenseType("Family", p_val, 2, 6, cls.BLUE),
            ]
        try:
            return cls._CONFIGS[name]()
        except KeyError:
            available = ", ".join(cls._CONFIGS.keys())
            msg = f"Unsupported license config: {name}. " \
                f"Available: {available} or roman_p_<x_y> " \
                f"or duolingo_p_<x_y>"
            raise ValueError(msg) from None
