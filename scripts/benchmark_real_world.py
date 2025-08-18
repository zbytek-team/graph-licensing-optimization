from src.algorithms import ILPSolver, GreedyAlgorithm, RandomizedAlgorithm, DominatingSetAlgorithm, NaiveAlgorithm
from src.graphs import RealWorldDataLoader, GraphGeneratorFactory
from src.core import LicenseConfigFactory
from datetime import datetime
import csv
import os
import time
import argparse
import logging

logger = logging.getLogger(__name__)


def benchmark_real_world_networks():
    """Benchmark algorytmów na rzeczywistych sieciach Facebook."""

    loader = RealWorldDataLoader()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Przygotuj algorytmy
    algorithms = [
        ("Naive", lambda: NaiveAlgorithm()),
        ("Dominating Set", lambda: DominatingSetAlgorithm()),
        ("Randomized (90% greedy)", lambda: RandomizedAlgorithm(greedy_probability=0.9, seed=42)),
        ("Randomized (70% greedy)", lambda: RandomizedAlgorithm(greedy_probability=0.7, seed=42)),
        ("Randomized (50% greedy)", lambda: RandomizedAlgorithm(greedy_probability=0.5, seed=42)),
        ("Greedy", lambda: GreedyAlgorithm()),
        ("ILP Optimal", lambda: ILPSolver()),
    ]

    # Konfiguracje licencji do testowania
    license_configs = ["spotify", "duolingo_super", "roman_domination"]

    # Znajdź odpowiednie sieci do testowania
    suitable_networks = loader.get_suitable_networks_for_testing(min_nodes=20, max_nodes=200)
    logger.info("Testowanie na %d sieciach Facebook: %s", len(suitable_networks), suitable_networks)

    # Przygotuj plik wyników
    results_dir = "results/real_world"
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"{results_dir}/facebook_benchmark_{timestamp}.csv"

    with open(results_file, "w", newline="") as csvfile:
        fieldnames = [
            "network_id",
            "nodes",
            "edges",
            "density",
            "avg_clustering",
            "license_config",
            "algorithm",
            "cost",
            "groups",
            "execution_time",
            "quality_ratio",
            "is_optimal",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for network_id in suitable_networks:
            logger.info("\nTesting network %s...", network_id)

            # Załaduj sieć
            try:
                graph = loader.load_facebook_ego_network(network_id)

                # Statystyki sieci
                import networkx as nx

                nodes = len(graph.nodes())
                edges = len(graph.edges())
                density = nx.density(graph)
                avg_clustering = nx.average_clustering(graph)

                logger.info("  %d węzłów, %d krawędzi, density=%.3f", nodes, edges, density)

                for license_config in license_configs:
                    license_types = LicenseConfigFactory.get_config(license_config)
                    optimal_cost = None

                    for algo_name, algo_factory in algorithms:
                        # Skip naive dla dużych grafów
                        if algo_name == "Naive" and nodes > 10:
                            continue

                        try:
                            algorithm = algo_factory()
                            start_time = time.time()
                            solution = algorithm.solve(graph, license_types)
                            execution_time = time.time() - start_time

                            # Sprawdź czy to jest rozwiązanie optymalne
                            is_optimal = algo_name == "ILP Optimal" or (algo_name == "Naive" and nodes <= 10)

                            if is_optimal:
                                optimal_cost = solution.total_cost

                            # Oblicz quality ratio
                            quality_ratio = None
                            if optimal_cost and optimal_cost > 0:
                                quality_ratio = solution.total_cost / optimal_cost

                            # Zapisz wyniki
                            writer.writerow(
                                {
                                    "network_id": network_id,
                                    "nodes": nodes,
                                    "edges": edges,
                                    "density": density,
                                    "avg_clustering": avg_clustering,
                                    "license_config": license_config,
                                    "algorithm": algo_name,
                                    "cost": solution.total_cost,
                                    "groups": len(solution.groups),
                                    "execution_time": execution_time,
                                    "quality_ratio": quality_ratio,
                                    "is_optimal": is_optimal,
                                }
                            )

                            logger.info(
                                "    %s | %s | Cost: %.2f | Time: %.3fs",
                                f"{algo_name:25}",
                                f"{license_config:15}",
                                solution.total_cost,
                                execution_time,
                            )

                        except Exception as e:
                            logger.error(
                                "    %s | %s | ERROR: %s",
                                f"{algo_name:25}",
                                f"{license_config:15}",
                                e,
                            )

            except Exception as e:
                logger.error("  ERROR loading network %s: %s", network_id, e)

    logger.info("\nWyniki zapisane w: %s", results_file)


def compare_real_vs_generated():
    """Porównanie algorytmów na danych rzeczywistych vs. wygenerowanych."""

    loader = RealWorldDataLoader()

    # Wybierz reprezentatywne sieci rzeczywiste
    suitable_networks = loader.get_suitable_networks_for_testing(min_nodes=50, max_nodes=150)

    if len(suitable_networks) < 3:
        logger.warning("Za mało odpowiednich sieci do porównania")
        return

    # Weź 3 różne rozmiary sieci
    test_networks = suitable_networks[:3]

    algorithms = [
        ("Dominating Set", lambda: DominatingSetAlgorithm()),
        ("Randomized", lambda: RandomizedAlgorithm(greedy_probability=0.7, seed=42)),
        ("Greedy", lambda: GreedyAlgorithm()),
    ]

    license_types = LicenseConfigFactory.get_config("spotify")

    logger.info("Porównanie: Sieci Rzeczywiste vs. Wygenerowane")
    logger.info("=" * 60)

    for network_id in test_networks:
        real_graph = loader.load_facebook_ego_network(network_id)
        real_nodes = len(real_graph.nodes())

        # Utwórz graf wygenerowany o podobnym rozmiarze
        generated_graph = GraphGeneratorFactory.small_world(real_nodes, k=min(4, real_nodes - 1), p=0.3, seed=42)

        logger.info("\nSieć %s (%d węzłów):", network_id, real_nodes)
        logger.info("-" * 40)

        for algo_name, algo_factory in algorithms:
            algorithm = algo_factory()

            # Test na sieci rzeczywistej
            real_solution = algorithm.solve(real_graph, license_types)

            # Test na sieci wygenerowanej
            generated_solution = algorithm.solve(generated_graph, license_types)

            cost_ratio = real_solution.total_cost / generated_solution.total_cost

            logger.info(
                "%s | Rzeczywista: %.2f | Wygenerowana: %.2f | Ratio: %.2f",
                f"{algo_name:15}",
                real_solution.total_cost,
                generated_solution.total_cost,
                cost_ratio,
            )


def analyze_facebook_network_properties():
    """Analiza właściwości sieci Facebook."""

    loader = RealWorldDataLoader()
    stats = loader.get_facebook_network_stats()

    logger.info("Analiza właściwości sieci Facebook")
    logger.info("=" * 50)

    # Sortuj według rozmiaru
    sorted_networks = sorted(stats.items(), key=lambda x: x[1]["nodes"])

    logger.info(f"{'Network ID':<10} {'Nodes':<6} {'Edges':<6} {'Density':<8} {'Clustering':<10} {'Components':<10}")
    logger.info("-" * 70)

    for network_id, stat in sorted_networks:
        logger.info(
            f"{network_id:<10} {stat['nodes']:<6} {stat['edges']:<6} {stat['density']:<8.3f} {stat['avg_clustering']:<10.3f} {stat['components']:<10}"
        )

    # Statystyki ogólne
    all_nodes = [stat["nodes"] for stat in stats.values()]
    all_densities = [stat["density"] for stat in stats.values()]
    all_clusterings = [stat["avg_clustering"] for stat in stats.values()]

    logger.info("\nStatystyki ogólne:")
    logger.info("Liczba sieci: %d", len(stats))
    logger.info(
        "Rozmiar węzłów - min: %d, max: %d, avg: %.1f",
        min(all_nodes),
        max(all_nodes),
        sum(all_nodes) / len(all_nodes),
    )
    logger.info(
        "Gęstość - min: %.3f, max: %.3f, avg: %.3f",
        min(all_densities),
        max(all_densities),
        sum(all_densities) / len(all_densities),
    )
    logger.info(
        "Clustering - min: %.3f, max: %.3f, avg: %.3f",
        min(all_clusterings),
        max(all_clusterings),
        sum(all_clusterings) / len(all_clusterings),
    )


def main():
    parser = argparse.ArgumentParser(description="Real world benchmark")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Set the logging level (e.g., DEBUG, INFO, WARNING)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    logger.info("=== Benchmark algorytmów na rzeczywistych danych Facebook ===\n")

    # 1. Analiza właściwości sieci
    analyze_facebook_network_properties()

    logger.info("\n" + "=" * 60 + "\n")

    # 2. Porównanie real vs generated
    compare_real_vs_generated()

    logger.info("\n" + "=" * 60 + "\n")

    # 3. Pełny benchmark
    benchmark_real_world_networks()

    logger.info("\n=== Benchmark zakończony ===")


if __name__ == "__main__":
    main()
