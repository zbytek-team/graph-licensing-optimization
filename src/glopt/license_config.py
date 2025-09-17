from collections.abc import Callable
from typing import ClassVar

from .core import LicenseType


class LicenseConfigFactory:
    # Dark, high-contrast palette mapped by license type
    # Individual = Red, Family = Blue, Duo = Green
    RED = "#B91C1C"  # Individual
    BLUE = "#1E3A8A"  # Family
    GREEN = "#065F46"  # Duo

    _CONFIGS: ClassVar[dict[str, Callable[[], list[LicenseType]]]] = {
        "duolingo_super": lambda: [
            LicenseType("Individual", 13.99, 1, 1, LicenseConfigFactory.RED),
            LicenseType("Family", 29.17, 2, 6, LicenseConfigFactory.BLUE),
        ],
        "spotify": lambda: [
            LicenseType("Individual", 23.99, 1, 1, LicenseConfigFactory.RED),
            LicenseType("Duo", 30.99, 2, 2, LicenseConfigFactory.GREEN),
            LicenseType("Family", 37.99, 2, 6, LicenseConfigFactory.BLUE),
        ],
        "netflix": lambda: [
            LicenseType("StdWithAds", 7.99, 1, 1, LicenseConfigFactory.RED),
            LicenseType("Standard", 17.99, 2, 2, LicenseConfigFactory.GREEN),
            LicenseType("Premium", 24.99, 2, 4, LicenseConfigFactory.BLUE),
        ],
        "roman_domination": lambda: [
            LicenseType("Solo", 1.0, 1, 1, LicenseConfigFactory.BLUE),
            LicenseType("Group", 2.0, 2, 999999, LicenseConfigFactory.RED),
        ],
    }

    @classmethod
    def get_config(cls, name: str) -> list[LicenseType]:
        # Dynamic roman domination sweep: roman_p_1_5 (unbounded group capacity)
        if name.startswith("roman_p_"):
            p_str = name.split("_", 2)[2]
            p_str = p_str.replace("_", ".")
            try:
                p_val = float(p_str)
            except Exception:
                available = ", ".join(cls._CONFIGS.keys())
                raise ValueError(f"Invalid roman price '{name}'. Available: {available} or roman_p_<x_y>")
            return [
                LicenseType("Solo", 1.0, 1, 1, cls.BLUE),
                LicenseType("Group", p_val, 2, 999999, cls.RED),
            ]
        # Duolingo-style sweep with capacity limited to 6: duolingo_p_2_0 → group=2.0× solo, cap=6
        if name.startswith("duolingo_p_"):
            p_str = name.split("_", 2)[2]
            p_str = p_str.replace("_", ".")
            try:
                p_val = float(p_str)
            except Exception:
                available = ", ".join(cls._CONFIGS.keys())
                raise ValueError(f"Invalid duolingo price '{name}'. Available: {available} or duolingo_p_<x_y>")
            return [
                LicenseType("Individual", 1.0, 1, 1, cls.RED),
                LicenseType("Family", p_val, 2, 6, cls.BLUE),
            ]
        try:
            return cls._CONFIGS[name]()
        except KeyError:
            available = ", ".join(cls._CONFIGS.keys())
            msg = f"Unsupported license config: {name}. Available: {available} or roman_p_<x_y> or duolingo_p_<x_y>"
            raise ValueError(msg) from None
