# 🎓 Modeling Optimal Software License Purchasing in Social Networks using Graph Domination

_Master's Thesis Project_

## 📖 Project Overview

This project implements a master's thesis on **modeling optimal software license purchasing in social networks**. The main use case is **Duolingo Super**, which offers individual licenses and group versions for six people (typically twice as expensive as individual licenses).

### 🎯 Thesis Objectives

The project aims to:

1. **Prove the equivalence** between the minimum cost license purchasing problem and the **Roman domination problem in graphs**
2. **Analyze different pricing models** (e.g., six-person version 3x more expensive than individual)
3. **Consider the problem of non-simultaneous license purchasing**
4. **Study the dynamic version of the problem**

### 🌐 Mathematical Model

In the social network model:

- **Vertices** represent users
- **Edges** represent friendship relationships
- **Goal**: Minimize license cost while covering all users

**Available licensing options:**

- **Individual license**: One user, cost S
- **Group license**: Up to 6 connected users, cost G (where G = 2.4S in the basic Duolingo model)

### 🔬 Equivalence with Roman Domination

The license purchasing problem is **mathematically equivalent** to the Roman domination problem in graphs, where:

- Vertices with individual licenses correspond to dominating function value 1
- Vertices that are group license owners correspond to function value 2.4
- Neighbors of group owners are "protected" (value 0)

## ✨ Algorithmic Implementation

### 🔬 Exact Algorithms

#### **Integer Linear Programming (ILP)**

- Optimal solutions using PuLP
- Implementation of Roman domination constraints
- Optimal for small graphs (up to ~25 vertices)

#### **Naive Algorithm (Brute Force)**

- Exhaustive search of all possibilities
- Verification of correctness for other algorithms
- Useful for very small instances

### 🎯 Approximation Algorithms

#### **Greedy Algorithm**

- Heuristic based on vertex degrees
- Fast solutions for large graphs
- Prioritization of high-degree nodes

#### **Dominating Set Algorithm**

- Graph theory-based approximation
- Utilization of classical domination algorithms
- Good solution quality for structural graphs

#### **Randomized Algorithm**

- Baseline for comparisons
- Random license assignment
- Worst-case scenario analysis

### 🧬 Metaheuristics

#### **Genetic Algorithm**

- Evolutionary optimization with crossover and mutation
- Exploration of solution space
- Adapted to domination problem structure

#### **Simulated Annealing**

- Temperature-based probabilistic search
- Escape from local optima
- Controlled acceptance of worse solutions

#### **Tabu Search**

- Local search with tabu list
- Memory of previous moves
- Efficient for medium-sized graphs

### 📊 Test Graph Generation

Support for various types of social networks:

- **Random graphs** (Erdős–Rényi) - modeling random connections
- **Scale-free networks** (Barabási–Albert) - modeling social networks with central nodes
- **Small-world networks** (Watts-Strogatz) - modeling real social networks
- **Regular structures** (Grid, Star, Path, Cycle, Complete) - theoretical cases
- **Power-law cluster graphs** - advanced community modeling

### 📈 Comprehensive Benchmarking

- **Performance analysis** of algorithms on different graph types
- **Scalability testing** with different network sizes
- **Price sensitivity analysis** (S:G price ratios)
- **Dynamic graph evolution** support
- **Statistical reporting** with visualizations

### 🎨 Result Visualization

- Solution visualization with color-coded nodes
- Algorithm performance comparison charts
- Cost and runtime analysis
- Correlation heatmaps
- Solution quality charts

## 🚀 Quick Start

### Requirements

- Python 3.13+
- UV package manager (recommended) or pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd graph-licensing-optimization

# Install dependencies using UV
uv sync

# Or using pip
pip install -e .
```

### Basic Usage

#### Run Single Algorithm

```bash
# Greedy algorithm on random graph (Duolingo Super modeling)
uv run main.py single --algorithm greedy --graph-type random --graph-size 20

# ILP with custom pricing (3:1 ratio instead of 2.4:1)
uv run main.py single --algorithm ilp --solo-cost 1 --group-cost 3 --graph-size 15
```

#### Algorithm Comparison

```bash
# Benchmark comparison
uv run main.py benchmark \
    --algorithms greedy --algorithms ilp --algorithms genetic \
    --graph-types random --graph-types scale_free \
    --graph-sizes 10 --graph-sizes 20
```

## 📈 Analysis and Visualization

The project includes comprehensive analysis tools:

```bash
# Generate full analysis report
uv run analysis_visualizer.py

# Creates:
# - Executive summary
# - Performance comparison plots
# - Statistical analysis
# - Correlation heatmaps
# - Scalability charts
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [NetworkX](https://networkx.org/) for graph operations
- Optimization powered by [PuLP](https://pypi.org/project/PuLP/)
- Visualizations created with [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)
- CLI interface using [Click](https://click.palletsprojects.com/)
