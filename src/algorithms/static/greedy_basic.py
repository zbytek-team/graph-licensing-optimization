from ..base import BaseSolver, Solution
import networkx as nx


class GreedyBasicSolver(BaseSolver):
    def solve(self, graph: nx.Graph, c_single: float, c_group: float, group_size: int) -> Solution:
        vertices = sorted(graph.nodes())
        n = len(vertices)

        covered = set()
        licenses: Solution = {"singles": [], "groups": []}

        while len(covered) < n:
            best_vertex = None
            best_score = -1

            not_covered = [v for v in vertices if v not in covered]
            if not not_covered:
                break

            for v in not_covered:
                nei_not_covered = sum(1 for nei in graph[v] if nei not in covered)
                score = 1 + nei_not_covered
                if score > best_score:
                    best_score = score
                    best_vertex = v

            if best_vertex is None:
                break

            nei_candidates = [nei for nei in graph[best_vertex] if nei not in covered]
            possible_cover_count = 1 + min(len(nei_candidates), group_size - 1)

            if possible_cover_count >= 2 and c_group < possible_cover_count * c_single:
                chosen_nei = nei_candidates[: group_size - 1]
                members = [best_vertex] + chosen_nei

                licenses["groups"].append({"license_holder": best_vertex, "members": members})

                for m in members:
                    covered.add(m)
            else:
                licenses["singles"].append(best_vertex)
                covered.add(best_vertex)

        return licenses
