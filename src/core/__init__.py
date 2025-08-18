"""Podstawowe klasy i funkcje modelu licencyjnego.

Moduły korzystają z obiektów `networkx.Graph` i definicji licencji
(`LicenseType`, `LicenseGroup`)."""

from .models import LicenseType, LicenseGroup, Solution, Algorithm
from .licenses import LicenseConfigFactory
from .validation import SolutionValidator


__all__ = [
    "LicenseType",
    "LicenseGroup",
    "Solution",
    "Algorithm",
    "LicenseConfigFactory",
    "SolutionValidator",
]
