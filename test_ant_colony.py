#!/usr/bin/env python3
"""
Test script for Ant Colony Optimization algorithm
"""
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from graph_licensing.algorithms.ant_colony import AntColonyAlgorithm
from graph_licensing.algorithms.greedy import GreedyAlgorithm
from graph_licensing.models.license import LicenseConfig
from graph_licensing.generators.graph_generator import GraphGenerator


def test_ant_colony():
    """Test the Ant Colony algorithm with different parameters."""
    
    print("=== Ant Colony Optimization Algorithm Test ===\n")
    
    # Create test scenarios
    graph_configs = [
        ("Small Random", "random", 10),
        ("Medium Scale-Free", "scale_free", 20),
        ("Small-World", "small_world", 15),
    ]
    
    license_config = LicenseConfig(solo_price=1.0, group_price=2.08, group_size=6)
    
    for graph_name, graph_type, size in graph_configs:
        print(f"Testing on {graph_name} graph (size: {size})")
        print("-" * 50)
        
        # Generate graph
        graph = GraphGenerator.generate_graph(graph_type, size, seed=42)
        
        # Test different ACO parameters
        aco_configs = [
            ("Default", {}),
            ("Exploration", {"q0": 0.3, "alpha": 0.5, "beta": 3.0}),
            ("Exploitation", {"q0": 0.95, "alpha": 2.0, "beta": 1.0}),
            ("High Evaporation", {"rho": 0.8, "num_ants": 30}),
        ]
        
        results = []
        
        for config_name, params in aco_configs:
            # Create algorithm with specific parameters
            aco = AntColonyAlgorithm(
                num_ants=params.get("num_ants", 25),
                max_iterations=50,
                alpha=params.get("alpha", 1.0),
                beta=params.get("beta", 2.0),
                rho=params.get("rho", 0.5),
                q0=params.get("q0", 0.9),
                seed=42
            )
            
            # Solve
            solution = aco.solve(graph, license_config)
            cost = solution.calculate_cost(license_config)
            
            results.append((config_name, cost, len(solution.solo_nodes), len(solution.group_owners)))
            
            print(f"  {config_name:12} - Cost: {cost:6.2f}, Solo: {len(solution.solo_nodes):2d}, Groups: {len(solution.group_owners):2d}")
        
        # Compare with Greedy algorithm
        greedy = GreedyAlgorithm()
        greedy_solution = greedy.solve(graph, license_config)
        greedy_cost = greedy_solution.calculate_cost(license_config)
        
        print(f"  {'Greedy':12} - Cost: {greedy_cost:6.2f}, Solo: {len(greedy_solution.solo_nodes):2d}, Groups: {len(greedy_solution.group_owners):2d}")
        
        # Find best ACO result
        best_aco = min(results, key=lambda x: x[1])
        improvement = ((greedy_cost - best_aco[1]) / greedy_cost) * 100
        
        print(f"  Best ACO improvement over Greedy: {improvement:+.1f}%")
        print()
    
    print("=== Algorithm Information ===")
    aco = AntColonyAlgorithm()
    info = aco.get_algorithm_info()
    for key, value in info.items():
        if key != "pheromone_levels":
            print(f"{key}: {value}")
    print()


def demonstrate_pheromone_evolution():
    """Demonstrate how pheromones evolve during optimization."""
    
    print("=== Pheromone Evolution Demonstration ===\n")
    
    # Create a simple graph
    graph = GraphGenerator.generate_graph("random", 12, seed=42)
    config = LicenseConfig(solo_price=1.0, group_price=2.08, group_size=6)
    
    # Create algorithm with fewer iterations to show evolution
    aco = AntColonyAlgorithm(
        num_ants=20,
        max_iterations=10,
        seed=42
    )
    
    # Solve and track pheromone levels
    print("Tracking pheromone evolution:")
    print("Iteration | Solo Pheromone Avg | Group Pheromone Avg | Best Cost")
    print("-" * 65)
    
    # We'll need to modify the algorithm to track this, but for demo purposes:
    solution = aco.solve(graph, config)
    
    final_info = aco.get_algorithm_info()
    print(f"Final     | {final_info['pheromone_levels']['solo_avg']:16.3f} | {final_info['pheromone_levels']['group_avg']:17.3f} | {final_info['best_cost']:8.2f}")
    
    print(f"\nFinal solution cost: {solution.calculate_cost(config):.2f}")
    print(f"Solo licenses: {len(solution.solo_nodes)}")
    print(f"Group licenses: {len(solution.group_owners)}")


if __name__ == "__main__":
    test_ant_colony()
    print("\n" + "="*60 + "\n")
    demonstrate_pheromone_evolution()
