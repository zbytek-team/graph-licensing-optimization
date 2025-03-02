from src.solvers.base import Solver, SolverResult
import networkx as nx


class GreedySolver(Solver):
    def _solve(self, graph: nx.Graph) -> SolverResult:
        covered = set()
        result: SolverResult = {"individual": set(), "group": {}}

        sorted_nodes = sorted(graph.nodes, key=lambda x: len(list(graph.neighbors(x))), reverse=True)

        for node in sorted_nodes:
            if node in covered:
                continue

            potential_group_members = set(graph.neighbors(node)) - covered
            selected_members = {node}

            if len(potential_group_members) > self.group_size - 1:
                sorted_neighbors = sorted(
                    potential_group_members,
                    key=lambda x: len(list(graph.neighbors(x))),
                    reverse=True,
                )
                selected_members.update(sorted_neighbors[: self.group_size - 1])
            else:
                selected_members.update(potential_group_members)

            if len(selected_members) > 1:
                result["group"][node] = selected_members
                covered.update(selected_members)
            else:
                result["individual"].add(node)
                covered.add(node)

        return result
