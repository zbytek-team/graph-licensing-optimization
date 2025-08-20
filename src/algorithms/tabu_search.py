from collections import deque
from typing import Any, Deque, List
import networkx as nx

from src.core import Algorithm, LicenseType, Solution
from src.validation.solution_validator import SolutionValidator
from src.utils.mutations import MutationOperators
from .greedy import GreedyAlgorithm


class TabuSearch(Algorithm):
    @property
    def name(self) -> str:
        return "tabu_search"

    def solve(
        self,
        graph: nx.Graph,
        license_types: List[LicenseType],
        **kwargs: Any,
    ) -> Solution:
        max_iterations: int = kwargs.get("max_iterations", 1000)
        tabu_tenure: int = kwargs.get("tabu_tenure", 20)
        neighbors_per_iter: int = kwargs.get("neighbors_per_iter", 10)

        validator = SolutionValidator(debug=False)

        # seed: greedy solution
        greedy = GreedyAlgorithm()
        current = greedy.solve(graph, license_types)
        best = current

        # FIFO tabu list of solution hashes
        tabu: Deque[str] = deque(maxlen=max(1, tabu_tenure))
        tabu.append(self._hash(current))

        for _ in range(max_iterations):
            # neighborhood via mutation operators
            neighborhood: List[Solution] = MutationOperators.generate_neighbors(base=current, graph=graph, license_types=license_types, k=neighbors_per_iter)
            if not neighborhood:
                break

            chosen: Solution | None = None
            chosen_cost = float("inf")

            for cand in neighborhood:
                ok, _ = validator.validate(cand, graph)
                if not ok:
                    continue
                h = self._hash(cand)

                # tabu unless it improves global best (aspiration)
                if h in tabu and cand.total_cost >= best.total_cost:
                    continue

                if cand.total_cost < chosen_cost:
                    chosen = cand
                    chosen_cost = cand.total_cost

            if chosen is None:
                break

            current = chosen
            if current.total_cost < best.total_cost:
                best = current

            tabu.append(self._hash(current))

        return best

    # stable string signature of a solution
    def _hash(self, solution: Solution) -> str:
        parts: List[str] = []
        for g in sorted(solution.groups, key=lambda gg: (str(gg.owner), gg.license_type.name)):
            members = ",".join(map(str, sorted(g.all_members)))
            parts.append(f"{g.license_type.name}:{g.owner}:{members}")
        return "|".join(parts)
