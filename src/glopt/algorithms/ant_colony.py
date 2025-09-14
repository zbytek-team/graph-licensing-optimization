from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, cast
from collections.abc import Sequence

from glopt.algorithms.greedy import GreedyAlgorithm
from glopt.core import Algorithm, LicenseGroup, LicenseType, Solution
from glopt.core.solution_validator import SolutionValidator

if TYPE_CHECKING:
    import networkx as nx

PKey = tuple[Any, str]


class AntColonyOptimization(Algorithm):
    @property
    def name(self) -> str:
        return "ant_colony_optimization"

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation: float = 0.5,
        q0: float = 0.9,
        num_ants: int = 20,
        max_iterations: int = 100,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.evap = evaporation
        self.q0 = q0
        self.num_ants = num_ants
        self.max_iter = max_iterations
        self.validator = SolutionValidator(debug=False)

    def solve(self, graph: nx.Graph, license_types: Sequence[LicenseType], **kwargs: Any) -> Solution:
        seed = kwargs.get("seed")
        if isinstance(seed, int):
            random.seed(seed)
        deadline = kwargs.get("deadline")
        max_iter = int(kwargs.get("max_iterations", self.max_iter))
        num_ants = int(kwargs.get("num_ants", self.num_ants))
        initial: Solution | None = kwargs.get("initial_solution")

        pher = self._init_pher(graph, license_types)
        heur = self._init_heur(graph, license_types)

        # Warm start or greedy baseline
        if initial is not None and self.validator.is_valid_solution(initial, graph):
            best = initial
        else:
            best = GreedyAlgorithm().solve(graph, license_types)
        ok, _ = self.validator.validate(best, graph)
        if not ok:
            best = self._fallback_singletons(graph, license_types)
        best_cost = best.total_cost
        self._deposit(pher, best)

        from time import perf_counter as _pc

        for _ in range(max_iter):
            if deadline is not None and _pc() >= float(deadline):
                break
            improved = False
            for _ in range(num_ants):
                cand = self._construct(graph, license_types, pher, heur)
                ok, _ = self.validator.validate(cand, graph)
                if not ok:
                    continue
                if cand.total_cost < best_cost:
                    best, best_cost, improved = cand, cand.total_cost, True
            self._evaporate(pher)
            self._deposit(pher, best)
            if not improved:
                continue
        # Final safety: ensure we never return an invalid solution.
        ok, _ = self.validator.validate(best, graph)
        if not ok:
            return self._fallback_singletons(graph, license_types)
        return best

    def _construct(
        self,
        graph: nx.Graph,
        lts: Sequence[LicenseType],
        pher: dict[PKey, float],
        heur: dict[PKey, float],
    ) -> Solution:
        uncovered: set[Any] = set(graph.nodes())
        groups: list[LicenseGroup] = []
        while uncovered:
            owner = self._select_owner(uncovered, lts, pher, heur)
            owner = owner if owner is not None else next(iter(uncovered))
            lt = self._select_license(owner, lts, pher, heur) or min(lts, key=lambda x: x.cost)

            pool = (set(graph.neighbors(owner)) | {owner}) & uncovered
            if len(pool) < lt.min_capacity:
                # Not enough neighbors for this license. Prefer singles; otherwise, use cheapest single-like fallback.
                singles = [x for x in lts if x.min_capacity <= 1 <= x.max_capacity]
                if singles:
                    lt1 = min(singles, key=lambda x: x.cost)
                    groups.append(LicenseGroup(lt1, owner, frozenset()))
                    uncovered.remove(owner)
                else:
                    # Fallback: use the globally cheapest license and place owner alone; validator may later reject,
                    # but we guarantee progress (and final safety check will repair via fallback).
                    lt1 = min(lts, key=lambda x: x.cost)
                    groups.append(LicenseGroup(lt1, owner, frozenset()))
                    uncovered.remove(owner)
                continue

            k = max(0, lt.max_capacity - 1)
            degv = cast(Any, graph.degree)

            def _deg_val(node: Any) -> int:
                return int(degv[node])

            add = sorted((pool - {owner}), key=_deg_val, reverse=True)[:k]
            groups.append(LicenseGroup(lt, owner, frozenset(add)))
            uncovered -= {owner} | set(add)
        return Solution(groups=tuple(groups))

    def _select_owner(
        self,
        uncovered: set[Any],
        lts: Sequence[LicenseType],
        pher: dict[PKey, float],
        heur: dict[PKey, float],
    ) -> Any | None:
        if not uncovered:
            return None
        scores: dict[Any, float] = {}
        for n in uncovered:
            acc = 0.0
            for lt in lts:
                tau = pher.get((n, lt.name), 1.0)
                eta = heur.get((n, lt.name), 1.0)
                acc += (tau**self.alpha) * (eta**self.beta)
            scores[n] = acc / max(1, len(lts))
        return self._roulette_or_best(list(uncovered), scores)

    def _select_license(self, owner: Any, lts: Sequence[LicenseType], pher: dict[PKey, float], heur: dict[PKey, float]) -> LicenseType | None:
        if not lts:
            return None
        scores = {lt: (pher.get((owner, lt.name), 1.0) ** self.alpha) * (heur.get((owner, lt.name), 1.0) ** self.beta) for lt in lts}
        return self._roulette_or_best(lts, scores)

    def _roulette_or_best(self, choices: list[Any], scores: dict[Any, float]) -> Any:
        if not choices:
            return None
        if random.random() < self.q0:
            return max(choices, key=lambda c: scores.get(c, 0.0))
        total = sum(max(0.0, scores.get(c, 0.0)) for c in choices)
        if total <= 0:
            return random.choice(choices)
        r = random.uniform(0, total)
        acc = 0.0
        for c in choices:
            acc += max(0.0, scores.get(c, 0.0))
            if acc >= r:
                return c
        return random.choice(choices)

    def _init_pher(self, graph: nx.Graph, lts: Sequence[LicenseType]) -> dict[PKey, float]:
        return {(n, lt.name): 1.0 for n in graph.nodes() for lt in lts}

    def _init_heur(self, graph: nx.Graph, lts: Sequence[LicenseType]) -> dict[PKey, float]:
        h: dict[PKey, float] = {}
        degv = cast(Any, graph.degree)
        for n in graph.nodes():
            deg = int(degv[n])
            for lt in lts:
                cap_eff = (lt.max_capacity / lt.cost) if lt.cost > 0 else 1e9
                h[n, lt.name] = cap_eff * (1.0 + float(deg))
        return h

    def _evaporate(self, pher: dict[PKey, float]) -> None:
        f = max(0.0, min(1.0, self.evap))
        for k in pher:
            pher[k] *= 1.0 - f

    def _deposit(self, pher: dict[PKey, float], sol: Solution) -> None:
        if sol.total_cost <= 0:
            return
        q = 1.0 / sol.total_cost
        for g in sol.groups:
            for n in g.all_members:
                k = (n, g.license_type.name)
                if k in pher:
                    pher[k] += q

    def _fallback_singletons(self, graph: nx.Graph, lts: Sequence[LicenseType]) -> Solution:
        lt1 = min([x for x in lts if x.min_capacity <= 1] or lts, key=lambda x: x.cost)
        groups = [LicenseGroup(lt1, n, frozenset()) for n in graph.nodes()]
        return Solution(groups=tuple(groups))
