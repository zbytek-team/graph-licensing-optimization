from src.solvers.greedy import GreedySolver
from src.solvers.mip import MIPSolver
from src.solvers.tabu import TabuSolver
from src.solvers.ant_colony import AntColonySolver
from src.graphs.generators import generate_clustered_graph
from src.visualize import visualize_graph


def main():
    graph = generate_clustered_graph(
        N=10_000,
        p_ws=0.1,
        extra_links=10,
        num_subgroups=20,
        min_subgroup_size=2,
        max_subgroup_size=14,
        inter_cluster_prob=0.1,
    )

    greedy_solver = GreedySolver(individual_cost=1, group_cost=1.2, group_size=6)
    mip_solver = MIPSolver(individual_cost=1, group_cost=1.2, group_size=6)
    tabu_solver = TabuSolver(
        individual_cost=1, group_cost=1.2, group_size=6, tabu_size=256, iterations=1024
    )
    ant_colony_solver = AntColonySolver(
        individual_cost=1,
        group_cost=1.2,
        group_size=6,
        ant_count=64,
        alpha=8,
        beta=1,
        evaporation_rate=0.15,
        iterations=1024,
    )

    costs = {}

    result, total_cost = greedy_solver.run(graph)

    costs["greedy"] = total_cost
    print(result)

    visualize_graph(graph, result, "images/greedy.png")

    result, total_cost = mip_solver.run(graph)
    costs["mip"] = total_cost
    print(result)

    visualize_graph(graph, result, "images/mip.png")

    result, total_cost = ant_colony_solver.run(graph)
    costs["ant_colony"] = total_cost
    print(result)

    visualize_graph(graph, result, "images/ant_colony.png")

    result, total_cost = tabu_solver.run(graph)
    costs["tabu"] = total_cost
    print(result)

    visualize_graph(graph, result, "images/tabu.png")

    print(costs)


if __name__ == "__main__":
    main()
