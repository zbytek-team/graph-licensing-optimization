# 🎓 Graph Licensing Optimization System

_Advanced Software License Optimization for Social Networks_

## 🚀 Quick Start

```bash
# Run a quick demo
python demo.py

# Test single algorithm
python main.py single --algorithm greedy --graph-type random --graph-size 20

# Compare algorithms
python main.py compare --algorithms greedy --algorithms genetic --algorithms ilp

# Run comprehensive benchmark
python main.py benchmark --algorithms greedy --algorithms genetic --graph-types random --graph-sizes 10 --graph-sizes 15
```

## 📖 Project Overview

This system provides **flexible, multi-tier license optimization** for social networks and graph structures. Originally developed for a master's thesis on **modeling optimal software license purchasing in social networks**, it now supports arbitrary license types with custom pricing and group size constraints.

### 🎯 Key Capabilities

1. **Flexible License System**: Support for any license types (individual, duo, family, enterprise, etc.) with custom pricing
2. **Advanced Algorithms**: 9 different optimization algorithms from exact to heuristic approaches
3. **Comprehensive Analysis**: Benchmarking, comparison, dynamic analysis, and parameter tuning
4. **Rich Visualizations**: Multi-color graphs showing license assignments and group relationships
5. **Production Ready**: Clean CLI interface, extensive testing, and professional documentation

### 🌐 Mathematical Foundation

**Original Problem**: Minimize software license costs in social networks (e.g., Duolingo Super):
- **Individual license**: One user, cost S  
- **Group license**: Up to 6 connected users, cost G (typically G = 2S)

**Mathematical Equivalence**: The license purchasing problem is equivalent to the **Roman domination problem** in graph theory, enabling the use of advanced graph algorithms for optimization.

## ✨ Algorithm Suite

### 🎯 Exact Algorithms
- **Integer Linear Programming (ILP)**: Optimal solutions using mathematical optimization
- **Naive Algorithm**: Brute force enumeration for small graphs

### 🧬 Metaheuristic Algorithms  
- **Genetic Algorithm**: Evolutionary optimization with crossover and mutation
- **Ant Colony Optimization**: Bio-inspired swarm intelligence
- **Simulated Annealing**: Temperature-based probabilistic optimization
- **Tabu Search**: Memory-based local search with forbidden moves

### ⚡ Heuristic Algorithms
- **Greedy Algorithm**: Fast approximate solutions using degree-based selection
- **Dominating Set Algorithm**: Graph theory-based approach
- **Randomized Algorithm**: Baseline for performance comparison

## 🛠️ Usage Guide

### Single Algorithm Testing
```bash
# Test specific algorithm on different graph types
python main.py single --algorithm greedy --graph-type random --graph-size 20
python main.py single --algorithm genetic --graph-type small_world --graph-size 15
python main.py single --algorithm ilp --graph-type complete --graph-size 10

# Custom license costs
python main.py single --algorithm greedy --solo-cost 1.5 --group-cost 4.0 --group-size 8
```

### Algorithm Comparison
```bash
# Compare multiple algorithms on same graph
python main.py compare --algorithms greedy --algorithms genetic --algorithms ilp
python main.py compare --algorithms ant_colony --algorithms simulated_annealing --algorithms tabu_search

# Compare on specific graph type
python main.py compare --algorithms greedy --algorithms ilp --graph-type scale_free --graph-size 15
```

### Comprehensive Benchmarking
```bash
# Full benchmark across algorithms and graph types
python main.py benchmark

# Custom benchmark parameters
python main.py benchmark --algorithms greedy --algorithms genetic \
  --graph-types random --graph-types scale_free \
  --graph-sizes 10 --graph-sizes 20 --graph-sizes 30 \
  --iterations 3
```

### Dynamic Analysis
```bash
# Analyze algorithm performance on evolving graphs
python main.py dynamic --algorithm greedy --graph-type random --initial-size 15 --iterations 10

# Intense graph modifications
python main.py dynamic --algorithm genetic --modification-prob 2.0 --create-gif
```

### Parameter Tuning
```bash
# Optimize algorithm parameters using Optuna
python main.py tune --algorithm genetic --n-trials 50
python main.py tune --algorithm ant_colony --n-trials 100 --graph-type scale_free
```

## 🎨 Visualization Features

- **Multi-Color License Types**: Different colors for each license type
- **Role Differentiation**: Visual distinction between owners and members  
- **Interactive Legends**: Complete cost and type information
- **Export Capabilities**: High-quality PNG outputs
- **Animated GIFs**: Dynamic analysis visualization (with `--create-gif`)

## 📊 Output and Results

All results are saved in timestamped directories under `results/`:

- **Single tests**: `results/single_YYYYMMDD_HHMMSS/`
- **Comparisons**: `results/compare_YYYYMMDD_HHMMSS/`  
- **Benchmarks**: `results/benchmark_YYYYMMDD_HHMMSS/`
- **Dynamic analysis**: `results/dynamic_YYYYMMDD_HHMMSS/`
- **Parameter tuning**: `results/tune_YYYYMMDD_HHMMSS/`

Each directory contains:
- 📊 **CSV/JSON data files** with detailed results
- 🖼️ **PNG visualizations** of solutions
- 📈 **Analysis plots** and comparisons
- 📝 **Metadata** with run parameters

## 🏗️ Project Structure

```
├── main.py                    # Main CLI interface
├── demo.py                    # Quick demonstration script
├── src/graph_licensing/       # Core package
│   ├── algorithms/            # All optimization algorithms
│   │   ├── ant_colony/        # Ant Colony Optimization
│   │   ├── genetic/           # Genetic Algorithm
│   │   ├── greedy/            # Greedy Algorithm
│   │   ├── ilp/               # Integer Linear Programming
│   │   ├── simulated_annealing/ # Simulated Annealing
│   │   ├── tabu_search/       # Tabu Search
│   │   ├── dominating_set/    # Dominating Set Algorithm
│   │   ├── randomized/        # Randomized Algorithm
│   │   └── naive/             # Naive/Brute Force
│   ├── models/                # Data models and structures
│   │   └── license.py         # License configuration and solution
│   ├── generators/            # Graph generation utilities
│   │   └── graph_generator.py # Various graph type generators
│   ├── visualizers/           # Visualization components
│   │   └── graph_visualizer.py # Solution visualization
│   ├── utils/                 # Utility functions
│   │   ├── benchmark.py       # Benchmarking framework
│   │   └── file_io.py         # File I/O operations
│   ├── optimization/          # Parameter tuning
│   │   └── optuna_tuner.py    # Optuna-based optimization
│   └── analysis/              # Analysis tools
│       └── analysis_runner.py # Result analysis
├── results/                   # Generated outputs
├── FEATURES.md               # Detailed feature documentation
└── README.md                 # This file
```

## 🔧 Installation & Setup

### Requirements
- Python 3.8+
- NetworkX for graph operations
- NumPy/SciPy for numerical computing
- Matplotlib for visualizations
- PuLP for linear programming
- Optuna for parameter tuning
- Click for CLI interface

### Quick Installation
```bash
# Clone repository
git clone <repository-url>
cd graph-licensing-optimization

# Install dependencies (using uv, pip, or poetry)
uv sync  # or pip install -e .

# Run quick demo
python demo.py
```

## 🎯 Advanced Features

### Flexible License System
The system supports arbitrary license types beyond the traditional solo/group model:

```python
# Example: Custom license configuration
config = LicenseConfig({
    'individual': {'price': 1.0, 'min_size': 1, 'max_size': 1},
    'duo': {'price': 1.8, 'min_size': 2, 'max_size': 2}, 
    'family': {'price': 3.0, 'min_size': 3, 'max_size': 5},
    'enterprise': {'price': 10.0, 'min_size': 6, 'max_size': 20}
})
```

### Graph Types Supported
- **Random graphs** (Erdős–Rényi) - Random connections
- **Scale-free networks** (Barabási–Albert) - Social networks with hubs
- **Small-world networks** (Watts-Strogatz) - Real social networks
- **Regular structures** (Grid, Star, Path, Cycle, Complete) - Theoretical cases
- **Power-law cluster graphs** - Community-based networks

### Performance Analysis
- **Runtime analysis** across different algorithms and graph sizes
- **Solution quality comparison** between exact and approximate methods
- **Scalability testing** for large networks
- **Parameter sensitivity analysis** for metaheuristic algorithms

## � Documentation

- **FEATURES.md**: Comprehensive feature documentation with examples
- **Code Documentation**: Extensive docstrings and type hints throughout
- **Algorithm Details**: Each algorithm includes theoretical background and implementation notes
- **Results Analysis**: Built-in tools for analyzing and comparing results

## 🤝 Contributing

This project is production-ready with clean architecture, comprehensive testing, and professional documentation. Contributions are welcome for:

- Additional optimization algorithms
- New graph generation methods
- Enhanced visualization features
- Performance improvements
- Extended analysis capabilities

## 📄 License

This project implements research from a master's thesis on optimal software license purchasing in social networks using graph domination theory. The mathematical foundation demonstrates equivalence between license optimization and Roman domination problems.
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
