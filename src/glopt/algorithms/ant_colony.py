from __future__ import annotations
import random
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
from ..core import Algorithm, LicenseGroup, LicenseType, Solution
from ..core.solution_validator import SolutionValidator
from .greedy import GreedyAlgorithm

PKey = Tuple[Any, str]  # (owner, license_name)


class AntColonyOptimization(Algorithm):
    @property
    def name(self) -> str:
        return "ant_colony_optimization"

    def __init__(self, alpha=1.0, beta=2.0, evaporation=0.5, q0=0.9, num_ants=20, max_iterations=100) -> None:
        self.alpha = alpha
        self.beta = beta
        self.evap = evaporation
        self.q0 = q0
        self.num_ants = num_ants
        self.max_iter = max_iterations
        self.validator = SolutionValidator(debug=False)

    def solve(self, graph: nx.Graph, license_types: List[LicenseType], **_: Any) -> Solution:
        pher = self._init_pher(graph, license_types)
        heur = self._init_heur(graph, license_types)

        best = GreedyAlgorithm().solve(graph, license_types)
        ok, _ = self.validator.validate(best, graph)
        if not ok:
            # fallback do bardzo taniej konstrukcji
            best = self._fallback_singletons(graph, license_types)
        best_cost = best.total_cost
        self._deposit(pher, best)

        for _ in range(self.max_iter):
            improved = False
            for _ in range(self.num_ants):
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
        return best

    # --- core construction ---
    def _construct(self, G: nx.Graph, lts: List[LicenseType], pher: Dict[PKey, float], heur: Dict[PKey, float]) -> Solution:
        uncovered: Set[Any] = set(G.nodes())
        groups: List[LicenseGroup] = []
        while uncovered:
            owner = self._select_owner(uncovered, lts, pher, heur, G)
            owner = owner if owner is not None else next(iter(uncovered))
            lt = self._select_license(owner, lts, pher, heur) or min(lts, key=lambda x: x.cost)

            pool = (set(G.neighbors(owner)) | {owner}) & uncovered
            if len(pool) < lt.min_capacity:
                # single jeśli wolno
                if lt.min_capacity == 1:
                    groups.append(LicenseGroup(lt, owner, frozenset()))
                    uncovered.remove(owner)
                else:
                    # wybierz najtańszą licencję, która się zmieści w pool
                    feas = [x for x in lts if x.min_capacity <= len(pool) <= x.max_capacity]
                    if feas:
                        lt2 = min(feas, key=lambda x: x.cost)
                        add_need = max(0, lt2.min_capacity - 1)
                        add = sorted((pool - {owner}), key=lambda n: G.degree(n), reverse=True)[:add_need]
                        groups.append(LicenseGroup(lt2, owner, frozenset(add)))
                        uncovered -= {owner} | set(add)
                    else:
                        uncovered.remove(owner)
                continue

            # formuj grupę do max_capacity
            k = max(0, lt.max_capacity - 1)
            add = sorted((pool - {owner}), key=lambda n: G.degree(n), reverse=True)[:k]
            groups.append(LicenseGroup(lt, owner, frozenset(add)))
            uncovered -= {owner} | set(add)
        return Solution(groups=tuple(groups))

    # --- selection ---
    def _select_owner(self, uncovered: Set[Any], lts: List[LicenseType], pher: Dict[PKey, float], heur: Dict[PKey, float], G: nx.Graph) -> Optional[Any]:
        if not uncovered:
            return None
        scores: Dict[Any, float] = {}
        for n in uncovered:
            acc = 0.0
            for lt in lts:
                tau = pher.get((n, lt.name), 1.0)
                eta = heur.get((n, lt.name), 1.0)
                acc += (tau**self.alpha) * (eta**self.beta)
            scores[n] = acc / max(1, len(lts))
        return self._roulette_or_best(list(uncovered), scores)

    def _select_license(self, owner: Any, lts: List[LicenseType], pher: Dict[PKey, float], heur: Dict[PKey, float]) -> Optional[LicenseType]:
        if not lts:
            return None
        scores = {lt: (pher.get((owner, lt.name), 1.0) ** self.alpha) * (heur.get((owner, lt.name), 1.0) ** self.beta) for lt in lts}
        return self._roulette_or_best(lts, scores)

    def _roulette_or_best(self, choices: List[Any], scores: Dict[Any, float]) -> Any:
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

    # --- pheromones / heuristics ---
    def _init_pher(self, G: nx.Graph, lts: List[LicenseType]) -> Dict[PKey, float]:
        return {(n, lt.name): 1.0 for n in G.nodes() for lt in lts}

    def _init_heur(self, G: nx.Graph, lts: List[LicenseType]) -> Dict[PKey, float]:
        h: Dict[PKey, float] = {}
        for n in G.nodes():
            deg = G.degree(n)
            for lt in lts:
                cap_eff = (lt.max_capacity / lt.cost) if lt.cost > 0 else 1e9
                h[(n, lt.name)] = cap_eff * (1.0 + deg)
        return h

    def _evaporate(self, pher: Dict[PKey, float]) -> None:
        f = max(0.0, min(1.0, self.evap))
        for k in pher:
            pher[k] *= 1.0 - f

    def _deposit(self, pher: Dict[PKey, float], sol: Solution) -> None:
        if sol.total_cost <= 0:
            return
        q = 1.0 / sol.total_cost
        for g in sol.groups:
            for n in g.all_members:
                k = (n, g.license_type.name)
                if k in pher:
                    pher[k] += q

    # --- fallback ---
    def _fallback_singletons(self, G: nx.Graph, lts: List[LicenseType]) -> Solution:
        lt1 = min([x for x in lts if x.min_capacity <= 1] or lts, key=lambda x: x.cost)
        groups = [LicenseGroup(lt1, n, frozenset()) for n in G.nodes()]
        return Solution(groups=tuple(groups))
