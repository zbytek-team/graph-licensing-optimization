from app.models.solve import License, LicenseAssignment, Assignments
from app.models.graph import Graph, Node


def greedy_solver(graph: Graph, licenses: list[License]) -> Assignments:
    assignments: Assignments = {}
    covered: set[Node] = set()

    neighbors_dict = {v: set(graph.get_neighbors(v)) for v in graph.nodes}

    while True:
        uncovered: set[Node] = set(graph.nodes) - covered
        if not uncovered:
            break

        best_score = -1.0
        best_assignment: tuple[License, Node, list[Node]] | None = None
        coverage_list: list[Node] = []

        for v in uncovered:
            # TODO: Do comparison between the following implementations:
            # 1. Without optimizations
            # 2. Using neighbors_dict
            # 3. Using candidate_set
            # 4. Optional: Old implementation for just two license types (individual and limited by 6 users), available in commit history

            # candidate_set = {v} | {u for u in graph.get_neighbors(v) if u not in covered}
            # candidate_list = list(candidate_set)

            candidate_set = {v} | neighbors_dict[v] - covered
            candidate_list = sorted(
                candidate_set,
                key=lambda x: len(neighbors_dict[x] - covered),
                reverse=True,
            )

            for license in licenses:
                coverage_list = candidate_list[: license.limit]
                score = len(coverage_list) / license.cost
                if score > best_score:
                    best_score = score
                    best_assignment = (license, v, coverage_list)

        if best_assignment is None:
            break

        selected_license, license_holder, covered_nodes = best_assignment

        for node in covered_nodes:
            covered.add(node)

        assignment = LicenseAssignment(
            license_holder=license_holder, covered_nodes=covered_nodes
        )

        if selected_license.name not in assignments:
            assignments[selected_license.name] = []
        assignments[selected_license.name].append(assignment)

    return assignments
