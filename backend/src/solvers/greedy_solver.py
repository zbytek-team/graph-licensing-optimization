
from ..models import LicenseAssignmentItem, LicenseAssignment, LicenseType, Graph


def greedy_solver(graph: Graph, license_types: list[LicenseType]) -> list[LicenseAssignment]:
    assignments: dict[str, list[LicenseAssignmentItem]] = {}
    covered: set[int] = set()

    neighbors_dict = {v: set(graph.neighbors(v)) for v in graph.nodes}

    while True:
        uncovered: set[int] = set(graph.nodes) - covered
        if not uncovered:
            break

        best_score = -1.0
        best_assignment = None
        coverage_list = []

        for v in uncovered:
            # TODO: Do comparison between the following implementations:
            # 1. Without optimizations
            # 2. Using neighbors_dict
            # 3. Using candidate_set
            # 4. Optional: Old implementation for just two license types (individual and limited by 6 users), available in commit history

            # candidate_set = {v} | {u for u in graph.neighbors(v) if u not in covered}
            # candidate_list = list(candidate_set)

            candidate_set = {v} | neighbors_dict[v] - covered
            candidate_list = sorted(candidate_set, key=lambda x: len(neighbors_dict[x] - covered), reverse=True)

            for lt in license_types:
                coverage_list = candidate_list[:lt.limit]
                score = len(coverage_list) / lt.cost
                if score > best_score:
                    best_score = score
                    best_assignment = (lt, v, coverage_list)
        
        if best_assignment is None:
            break

        selected_license, owner, users = best_assignment

        for u in users:
            covered.add(u)

        assignment_item = LicenseAssignmentItem(owner=owner, users=users)
        
        if selected_license.name not in assignments:
            assignments[selected_license.name] = []
        assignments[selected_license.name].append(assignment_item)

    return [LicenseAssignment(license_type=lt, item=items) for lt, items in assignments.items()]