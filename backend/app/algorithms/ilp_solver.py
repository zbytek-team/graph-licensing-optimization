import pulp
import networkx as nx


def ilp_license_distribution(
    nx_graph: nx.Graph,
    price_individual: float,
    price_family: float,
    max_group_size: int,
):
    problem = pulp.LpProblem("License_Distribution_ILP", pulp.LpMinimize)

    vertices = list(nx_graph.nodes())

    x = pulp.LpVariable.dicts("x", vertices, cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", vertices, cat=pulp.LpBinary)

    z = {}
    for v in vertices:
        for u in vertices:
            if nx_graph.has_edge(u, v) or u == v:
                z[(u, v)] = pulp.LpVariable(f"z_{u}_{v}", cat=pulp.LpBinary)
            else:
                pass

    problem += (
        pulp.lpSum([price_individual * x[v] + price_family * y[v] for v in vertices]),
        "Minimize total cost",
    )

    for v in vertices:
        problem += (
            x[v] + y[v] + pulp.lpSum([z[(v, w)] for w in vertices if (v, w) in z]) == 1,
            f"cover_{v}",
        )

    for v in vertices:
        for u in vertices:
            if (u, v) in z:
                problem += z[(u, v)] <= y[v], f"family_assign_only_if_owner_{u}_{v}"

    for v in vertices:
        problem += (
            pulp.lpSum([z[(u, v)] for u in vertices if (u, v) in z])
            <= (max_group_size - 1) * y[v],
            f"group_size_{v}",
        )

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    from config.config import LicenseType

    licenses = {
        LicenseType.INDIVIDUAL: [],
        LicenseType.GROUP_OWNER: [],
        LicenseType.GROUP_MEMBER: [],
    }

    if pulp.LpStatus[problem.status] != "Optimal":
        return licenses

    for v in vertices:
        xv = pulp.value(x[v])
        yv = pulp.value(y[v])
        if xv > 0.5:
            licenses[LicenseType.INDIVIDUAL].append(v)
        elif yv > 0.5:
            licenses[LicenseType.GROUP_OWNER].append(v)
            for u in vertices:
                if (u, v) in z and pulp.value(z[(u, v)]) > 0.5:
                    if u != v:
                        licenses[LicenseType.GROUP_MEMBER].append(u)
        else:
            for w in vertices:
                if (v, w) in z and pulp.value(z[(v, w)]) > 0.5:
                    licenses[LicenseType.GROUP_MEMBER].append(v)

    return licenses
