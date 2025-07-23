from typing import List
from src.core.types import LicenseType


class LicenseConfigFactory:
    """Factory for creating different license configurations."""

    @staticmethod
    def get_config(config_name: str) -> List[LicenseType]:
        """
        Get license configuration by name.

        Args:
            config_name: Name of the license configuration

        Returns:
            List of LicenseType objects

        Raises:
            ValueError: If config_name is not supported
        """
        configs = {
            "duolingo_super": LicenseConfigFactory._duolingo_super_config,
            "spotify": LicenseConfigFactory._spotify_config,
            "roman_domination": LicenseConfigFactory._roman_domination_config,
        }

        if config_name not in configs:
            available = ", ".join(configs.keys())
            raise ValueError(f"Unsupported license config: {config_name}. Available: {available}")

        return configs[config_name]()

    @staticmethod
    def list_available_configs() -> List[str]:
        """List all available license configurations."""
        return ["duolingo_super", "spotify", "roman_domination"]

    @staticmethod
    def _duolingo_super_config() -> List[LicenseType]:
        """
        Duolingo Super license configuration.
        - Individual: 13.99 PLN/mo for 1 person
        - Family: 29.17 PLN/mo for 2-6 members
        """
        return [
            LicenseType(
                name="Individual",
                cost=13.99,
                min_capacity=1,
                max_capacity=1,
                color="#58cc02",  # Duolingo green
            ),
            LicenseType(
                name="Family",
                cost=29.17,
                min_capacity=2,
                max_capacity=6,
                color="#ff9600",  # Duolingo orange
            ),
        ]

    @staticmethod
    def _spotify_config() -> List[LicenseType]:
        """
        Spotify Premium license configuration.
        - Individual: 23.99 PLN/mo for 1 person
        - Duo: 30.99 PLN/mo for 2 people
        - Family: 37.99 PLN/mo for 2-6 people
        """
        return [
            LicenseType(
                name="Individual",
                cost=23.99,
                min_capacity=1,
                max_capacity=1,
                color="#1db954",  # Spotify green
            ),
            LicenseType(
                name="Duo",
                cost=30.99,
                min_capacity=2,
                max_capacity=2,
                color="#1ed760",  # Spotify light green
            ),
            LicenseType(
                name="Family",
                cost=37.99,
                min_capacity=2,
                max_capacity=6,
                color="#191414",  # Spotify black
            ),
        ]

    @staticmethod
    def _roman_domination_config() -> List[LicenseType]:
        """
        Roman Domination problem configuration.
        - Solo: cost 1, exactly 1 person
        - Group: cost 2, minimum 2 people, unlimited capacity
        """
        return [
            LicenseType(
                name="Solo",
                cost=1.0,
                min_capacity=1,
                max_capacity=1,
                color="#8b0000",  # Dark red
            ),
            LicenseType(
                name="Group",
                cost=2.0,
                min_capacity=2,
                max_capacity=999,  # Practically unlimited
                color="#ffd700",  # Gold
            ),
        ]


class LicenseConfigInfo:
    """Information about license configurations."""

    @staticmethod
    def get_config_description(config_name: str) -> str:
        """Get human-readable description of a license configuration."""
        descriptions = {
            "duolingo_super": ("Duolingo Super subscription plans:\n- Individual: 13.99 PLN/month for 1 person\n- Family: 29.17 PLN/month for 2-6 members"),
            "spotify": (
                "Spotify Premium subscription plans:\n"
                "- Individual: 23.99 PLN/month for 1 person\n"
                "- Duo: 30.99 PLN/month for exactly 2 people\n"
                "- Family: 37.99 PLN/month for 2-6 people"
            ),
            "roman_domination": (
                "Roman Domination problem configuration:\n- Solo: cost 1 unit, exactly 1 person\n- Group: cost 2 units, minimum 2 people, unlimited capacity"
            ),
        }

        return descriptions.get(config_name, f"No description available for {config_name}")

    @staticmethod
    def print_config_info(config_name: str) -> None:
        """Print detailed information about a license configuration."""
        try:
            licenses = LicenseConfigFactory.get_config(config_name)
            description = LicenseConfigInfo.get_config_description(config_name)

            print(f"\n{config_name.upper()} LICENSE CONFIGURATION")
            print("=" * (len(config_name) + 22))
            print(description)
            print("\nLicense types:")

            for license_type in licenses:
                cost_per_max = license_type.cost / license_type.max_capacity
                print(
                    f"- {license_type.name}: "
                    f"cost={license_type.cost}, "
                    f"capacity={license_type.min_capacity}-{license_type.max_capacity}, "
                    f"cost/max={cost_per_max:.2f}, "
                    f"color={license_type.color}"
                )

        except ValueError as e:
            print(f"Error: {e}")

    @staticmethod
    def print_all_configs() -> None:
        """Print information about all available license configurations."""
        configs = LicenseConfigFactory.list_available_configs()
        print("AVAILABLE LICENSE CONFIGURATIONS")
        print("=" * 35)

        for config in configs:
            LicenseConfigInfo.print_config_info(config)
            print()  # Add spacing between configs
