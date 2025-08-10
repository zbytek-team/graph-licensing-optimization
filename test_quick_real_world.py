import sys
sys.path.append('.')

from src.graphs import RealWorldDataLoader
from src.algorithms import RandomizedAlgorithm, GreedyAlgorithm
from src.core import LicenseConfigFactory


def quick_real_world_test():
    loader = RealWorldDataLoader()
    
    # Załaduj małą sieć
    graph = loader.load_facebook_ego_network("3980")  # 53 węzły
    license_types = LicenseConfigFactory.get_config("spotify")
    
    print(f"Testowanie na sieci Facebook 3980: {len(graph.nodes())} węzłów")
    
    algorithms = [
        ("Greedy", GreedyAlgorithm()),
        ("Randomized 90%", RandomizedAlgorithm(greedy_probability=0.9, seed=42)),
        ("Randomized 50%", RandomizedAlgorithm(greedy_probability=0.5, seed=42)),
        ("Randomized 10%", RandomizedAlgorithm(greedy_probability=0.1, seed=42)),
    ]
    
    for name, algorithm in algorithms:
        solution = algorithm.solve(graph, license_types)
        print(f"{name:15} | Cost: {solution.total_cost:7.2f} | Groups: {len(solution.groups):2d}")
    
    print("\nTest rzeczywistych danych zakończony pomyślnie!")


if __name__ == "__main__":
    quick_real_world_test()
