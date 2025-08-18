"""Implementacja algorytmu genetycznego do dystrybucji licencji.

Pracuje na grafach `networkx` z użyciem konfiguracji licencji
(`LicenseType`, `LicenseGroup`)."""

from .main import GeneticAlgorithm

__all__ = ["GeneticAlgorithm"]
