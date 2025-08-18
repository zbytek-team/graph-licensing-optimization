import networkx as nx
import pytest

from src.core import LicenseType, LicenseGroup, SolutionValidator
from src.utils import SolutionBuilder


def basic_license_types():
    return [
        LicenseType("single", cost=5.0, min_capacity=1, max_capacity=1),
        LicenseType("pair", cost=6.0, min_capacity=2, max_capacity=2),
        LicenseType("triple", cost=8.0, min_capacity=3, max_capacity=3),
    ]


def test_find_cheapest_single_license():
    lts = basic_license_types()
    cheapest = SolutionBuilder.find_cheapest_single_license(lts)
    assert cheapest.name == "single"

    # When no license covers a single user, the cheapest overall should be returned
    lts_no_single = [
        LicenseType("double", cost=4.0, min_capacity=2, max_capacity=2),
        LicenseType("triple", cost=3.0, min_capacity=3, max_capacity=3),
    ]
    cheapest_overall = SolutionBuilder.find_cheapest_single_license(lts_no_single)
    assert cheapest_overall.name == "triple"


def test_find_cheapest_license_for_size():
    lts = basic_license_types()
    chosen = SolutionBuilder.find_cheapest_license_for_size(3, lts)
    assert chosen.name == "triple"
    assert SolutionBuilder.find_cheapest_license_for_size(4, lts) is None


def test_solution_validator_detects_overlap():
    graph = nx.Graph()
    graph.add_edge(0, 1)
    single, pair, _ = basic_license_types()

    valid_group = LicenseGroup(pair, owner=0, additional_members={1})
    valid_solution = SolutionBuilder.create_solution_from_groups([valid_group])
    validator = SolutionValidator()
    assert validator.is_valid_solution(valid_solution, graph)

    overlapping_groups = [
        valid_group,
        LicenseGroup(single, owner=1, additional_members=set()),
    ]
    invalid_solution = SolutionBuilder.create_solution_from_groups(overlapping_groups)
    assert not validator.is_valid_solution(invalid_solution, graph)
    with pytest.raises(ValueError):
        validator.validate_solution_strict(invalid_solution, graph)
