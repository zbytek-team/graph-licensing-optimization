from src.solvers.greedy import GreedySolver
import networkx as nx


def main():
    graph = nx.erdos_renyi_graph(100, 0.05)
    individual_cost = 1.0
    group_cost = 1.2
    group_size = 6
    coverage = GreedySolver(graph, individual_cost, group_cost, group_size)
    print(coverage)


if __name__ == "__main__":
    main()
