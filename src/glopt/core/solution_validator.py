from collections.abc import Hashable
from dataclasses import dataclass
from typing import TypeVar

import networkx as nx

from .models import LicenseGroup, Solution

N = TypeVar("N", bound=Hashable)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    msg: str


class SolutionValidator:
    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

    def validate(
        self,
        solution: Solution[N],
        graph: nx.Graph,
        all_nodes: set[N] | None = None,
    ) -> tuple[bool, list[ValidationIssue]]:
        issues: list[ValidationIssue] = []
        nodes: set[N] = set(graph.nodes()) if all_nodes is None else set(all_nodes)
        groups: tuple[LicenseGroup[N], ...] = tuple(solution.groups)

        issues += self._check_group_members(groups, nodes)
        issues += self._check_group_capacity(groups)
        issues += self._check_neighbors(groups, graph, nodes)
        issues += self._check_no_overlap(groups)
        issues += self._check_coverage(groups, nodes)

        if self.debug and issues:
            for _i in issues:
                pass

        return (not issues, issues)

    def is_valid_solution(
        self,
        solution: Solution[N],
        graph: nx.Graph,
        all_nodes: set[N] | None = None,
    ) -> bool:
        ok, _ = self.validate(solution, graph, all_nodes)
        return ok

    def _check_group_members(self, groups: tuple[LicenseGroup[N], ...], nodes: set[N]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for idx, g in enumerate(groups):
            outside = g.all_members - nodes
            if outside:
                issues.append(
                    ValidationIssue(
                        "MEMBER_NOT_IN_GRAPH",
                        f"group#{idx} owner {g.owner!r} has members not in graph: {sorted(outside)!r}",
                    )
                )
            if g.owner not in g.all_members:
                issues.append(
                    ValidationIssue("OWNER_NOT_IN_GROUP", f"group#{idx} owner {g.owner!r} not included in its members")
                )
        return issues

    def _check_group_capacity(self, groups: tuple[LicenseGroup[N], ...]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for idx, g in enumerate(groups):
            mn, mx, sz = g.license_type.min_capacity, g.license_type.max_capacity, g.size
            if not (mn <= sz <= mx):
                issues.append(
                    ValidationIssue(
                        "CAPACITY_VIOLATION",
                        f"group#{idx} owner {g.owner!r} size={sz} not in [{mn}, {mx}] for license '{g.license_type.name}'",
                    ),
                )
        return issues

    def _check_neighbors(
        self, groups: tuple[LicenseGroup[N], ...], graph: nx.Graph, nodes: set[N]
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for idx, g in enumerate(groups):
            if g.owner not in nodes:
                issues.append(ValidationIssue("OWNER_NOT_IN_GRAPH", f"group#{idx} owner {g.owner!r} not in graph"))
                continue
            allowed = set(graph.neighbors(g.owner)) | {g.owner}
            not_neighbors = g.all_members - allowed
            if not_neighbors:
                issues.append(
                    ValidationIssue(
                        "DISCONNECTED_MEMBER",
                        f"group#{idx} owner {g.owner!r} has non-neighbor members: {sorted(not_neighbors)!r}",
                    )
                )
        return issues

    def _check_no_overlap(self, groups: tuple[LicenseGroup[N], ...]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        seen: set[N] = set()
        for idx, g in enumerate(groups):
            overlap = seen & g.all_members
            if overlap:
                issues.append(
                    ValidationIssue("OVERLAP", f"group#{idx} owner {g.owner!r} overlaps members {sorted(overlap)!r}")
                )
            seen.update(g.all_members)
        return issues

    def _check_coverage(self, groups: tuple[LicenseGroup[N], ...], nodes: set[N]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        covered = set().union(*(g.all_members for g in groups)) if groups else set()
        missing = nodes - covered
        extra = covered - nodes
        if missing:
            issues.append(ValidationIssue("MISSING_COVERAGE", f"missing nodes: {sorted(missing)!r}"))
        if extra:
            issues.append(ValidationIssue("EXTRA_COVERAGE", f"extra nodes not in graph: {sorted(extra)!r}"))
        return issues
