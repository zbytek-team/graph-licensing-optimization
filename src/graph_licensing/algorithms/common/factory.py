"""Factory for creating algorithm instances."""

from typing import Dict, Type, Union, Any, Optional
from ..base import BaseAlgorithm
from .config import AlgorithmConfig


class AlgorithmFactory:
    """Factory for creating algorithm instances with proper configuration."""
    
    _algorithm_registry: Dict[str, Type[BaseAlgorithm]] = {}
    _config_registry: Dict[str, Type[AlgorithmConfig]] = {}
    
    @classmethod
    def register_algorithm(
        cls, 
        name: str, 
        algorithm_class: Type[BaseAlgorithm],
        config_class: Type[AlgorithmConfig]
    ) -> None:
        """Register an algorithm with its configuration class."""
        cls._algorithm_registry[name] = algorithm_class
        cls._config_registry[name] = config_class
    
    @classmethod
    def create_algorithm(
        cls, 
        name: str, 
        config: Optional[Union[AlgorithmConfig, dict]] = None,
        **kwargs
    ) -> BaseAlgorithm:
        """Create algorithm instance with configuration."""
        if name not in cls._algorithm_registry:
            raise ValueError(f"Unknown algorithm: {name}")
        
        algorithm_class = cls._algorithm_registry[name]
        config_class = cls._config_registry[name]
        
        # Handle configuration
        if config is None:
            config = config_class(**kwargs)
        elif isinstance(config, dict):
            config = config_class(**config)
        elif not isinstance(config, config_class):
            raise TypeError(f"Config must be {config_class.__name__} or dict")
        
        # Create algorithm instance
        return algorithm_class.from_config(config)
    
    @classmethod
    def get_available_algorithms(cls) -> list[str]:
        """Get list of available algorithm names."""
        return list(cls._algorithm_registry.keys())
    
    @classmethod
    def get_algorithm_config_class(cls, name: str) -> Type[AlgorithmConfig]:
        """Get configuration class for an algorithm."""
        if name not in cls._config_registry:
            raise ValueError(f"Unknown algorithm: {name}")
        return cls._config_registry[name]


# Register algorithms when module is imported
def _register_algorithms():
    """Register all available algorithms."""
    # Import algorithms here to avoid circular imports
    from ..greedy.greedy import GreedyAlgorithm
    
    # Register greedy algorithm (others will be added when they support from_config)
    AlgorithmFactory.register_algorithm(
        "greedy", 
        GreedyAlgorithm, 
        AlgorithmConfig
    )


# Auto-register algorithms
_register_algorithms()
