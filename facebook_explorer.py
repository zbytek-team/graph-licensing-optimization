#!/usr/bin/env python3
"""Facebook dataset explorer.

This script helps explore the available Facebook ego networks and their properties.
"""

import click
from tabulate import tabulate

from src.graph_licensing.generators.facebook_loader import FacebookDataLoader


@click.group()
def cli():
    """Facebook Dataset Explorer.
    
    Explore the available Facebook ego networks and their properties.
    """
    pass


@cli.command()
def list_networks():
    """List all available ego networks."""
    try:
        loader = FacebookDataLoader()
        networks = loader.get_available_ego_networks()
        
        click.echo(f"Found {len(networks)} ego networks:")
        for i, network_id in enumerate(networks, 1):
            click.echo(f"{i:2d}. {network_id}")
        
        click.echo(f"\nTo load a specific network, use: --facebook-ego <ID>")
        click.echo(f"Example: uv run main.py single --algorithm greedy --graph-type facebook --facebook-ego {networks[0]}")
        
    except Exception as e:
        click.echo(f"Error: {e}")


@cli.command()
@click.option("--ego-id", help="Specific ego network ID to analyze")
def analyze(ego_id):
    """Analyze ego network properties."""
    try:
        loader = FacebookDataLoader()
        
        if ego_id:
            # Analyze specific network
            info = loader.get_network_info(ego_id)
            click.echo(f"Network {ego_id} Analysis:")
            click.echo(f"  Nodes: {info['num_nodes']}")
            click.echo(f"  Edges: {info['num_edges']}")
            click.echo(f"  Density: {info['density']:.4f}")
            click.echo(f"  Average Clustering: {info['average_clustering']:.4f}")
            click.echo(f"  Average Degree: {info['average_degree']:.2f}")
        else:
            # Analyze all networks
            all_info = loader.get_all_networks_info()
            
            # Sort by number of nodes
            all_info.sort(key=lambda x: x['num_nodes'], reverse=True)
            
            # Prepare table data
            headers = ["Ego ID", "Nodes", "Edges", "Density", "Avg Clustering", "Avg Degree"]
            rows = []
            
            for info in all_info:
                rows.append([
                    info['ego_id'],
                    info['num_nodes'],
                    info['num_edges'],
                    f"{info['density']:.4f}",
                    f"{info['average_clustering']:.4f}",
                    f"{info['average_degree']:.2f}"
                ])
            
            click.echo("Facebook Ego Networks Analysis:")
            click.echo(tabulate(rows, headers=headers, tablefmt="grid"))
            
            # Summary statistics
            nodes = [info['num_nodes'] for info in all_info]
            edges = [info['num_edges'] for info in all_info]
            densities = [info['density'] for info in all_info]
            
            click.echo(f"\nSummary Statistics:")
            click.echo(f"  Total networks: {len(all_info)}")
            click.echo(f"  Node count range: {min(nodes)} - {max(nodes)}")
            click.echo(f"  Edge count range: {min(edges)} - {max(edges)}")
            click.echo(f"  Density range: {min(densities):.4f} - {max(densities):.4f}")
            
    except Exception as e:
        click.echo(f"Error: {e}")


@cli.command()
@click.option("--min-nodes", default=10, help="Minimum number of nodes")
@click.option("--max-nodes", default=1000, help="Maximum number of nodes")
def recommend(min_nodes, max_nodes):
    """Recommend suitable ego networks for testing."""
    try:
        loader = FacebookDataLoader()
        all_info = loader.get_all_networks_info()
        
        # Filter by size
        suitable = [
            info for info in all_info 
            if min_nodes <= info['num_nodes'] <= max_nodes
        ]
        
        if not suitable:
            click.echo(f"No networks found with {min_nodes}-{max_nodes} nodes")
            return
        
        # Sort by interesting properties
        suitable.sort(key=lambda x: (x['num_nodes'], -x['density']))
        
        click.echo(f"Recommended networks ({min_nodes}-{max_nodes} nodes):")
        click.echo()
        
        # Show a few recommendations with different characteristics
        recommendations = [
            ("Smallest", suitable[0]),
            ("Medium", suitable[len(suitable)//2] if len(suitable) > 1 else suitable[0]),
            ("Largest", suitable[-1] if len(suitable) > 1 else suitable[0]),
        ]
        
        # Add most/least dense if different
        most_dense = max(suitable, key=lambda x: x['density'])
        least_dense = min(suitable, key=lambda x: x['density'])
        
        if most_dense not in [r[1] for r in recommendations]:
            recommendations.append(("Most Dense", most_dense))
        if least_dense not in [r[1] for r in recommendations]:
            recommendations.append(("Least Dense", least_dense))
        
        for label, info in recommendations:
            click.echo(f"{label}: Ego {info['ego_id']}")
            click.echo(f"  Command: uv run main.py single --algorithm greedy --graph-type facebook --facebook-ego {info['ego_id']}")
            click.echo(f"  Nodes: {info['num_nodes']}, Edges: {info['num_edges']}, Density: {info['density']:.4f}")
            click.echo()
        
    except Exception as e:
        click.echo(f"Error: {e}")


if __name__ == "__main__":
    cli()
