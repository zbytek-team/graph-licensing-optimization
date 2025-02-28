from src.solvers.greedy import GreedySolver
from src.graphs.generators import generate_clustered_graph
from src.visualize import visualize_graph


def main():
    graph = generate_clustered_graph(N=300, p_ws=0.02, extra_links=3, num_subgroups=15)
    greedy_solver = GreedySolver(individual_cost=1, group_cost=1, group_size=6)

    result, total_cost = greedy_solver.run(graph)

    print(result)
    print(f"Total cost: {total_cost}")

    visualize_graph(graph, result)


if __name__ == "__main__":
    main()
