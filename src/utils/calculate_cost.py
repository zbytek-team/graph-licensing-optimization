from __future__ import annotations
from typing import Iterable
import networkx as nx
from src.algorithms.base import Solution


def calculate_cost(solution: Solution, c_single: float, c_group: float) -> float:
    return len(solution["singles"]) * c_single + len(solution["groups"]) * c_group


def _duplicates(items: Iterable[int]) -> set[int]:
    seen: set[int] = set()
    dup: set[int] = set()
    for x in items:
        if x in seen:
            dup.add(x)
        else:
            seen.add(x)
    return dup


def validate_solution(graph: nx.Graph, solution: Solution, group_size: int) -> list[str]:
    errors: list[str] = []

    singles: list[int] = solution["singles"]
    groups = solution["groups"]

    dup = _duplicates(singles)
    if dup:
        errors.append(f"duplicate_singles:{sorted(dup)}")

    holders: list[int] = [g["license_holder"] for g in groups]
    dup = _duplicates(holders)
    if dup:
        errors.append(f"duplicate_holders:{sorted(dup)}")

    covered: set[int] = set(singles)

    for g in groups:
        holder: int = g["license_holder"]
        members: list[int] = g["members"]

        if holder not in members:
            errors.append(f"holder_missing:{holder}")

        if not 2 <= len(members) <= group_size:
            errors.append(f"wrong_group_size:{holder}")

        for m in members:
            if m in singles:
                errors.append(f"member_as_single:{m}")
            if m in covered:
                errors.append(f"member_multi_group:{m}")
            if m != holder and not graph.has_edge(holder, m):
                errors.append(f"not_adjacent:{holder}-{m}")
            covered.add(m)

    if covered != set(graph.nodes):
        errors.append("not_all_nodes_dominated")

    return errors


def is_valid_solution(graph: nx.Graph, solution: Solution, group_size: int) -> bool:
    return not validate_solution(graph, solution, group_size)
