"""Optimization module for parameter tuning."""

# Functional API (new approach)
from .tune_genetic import tune_genetic_algorithm
from .tune_simulated_annealing import tune_simulated_annealing
from .tune_tabu_search import tune_tabu_search
from .tune_all import tune_all_algorithms, get_best_parameters, create_tuned_algorithm
from .utils import evaluate_algorithm, run_optuna_study

__all__ = [
    # Functional API
    "tune_genetic_algorithm",
    "tune_simulated_annealing",
    "tune_tabu_search",
    "tune_all_algorithms",
    "get_best_parameters",
    "create_tuned_algorithm",
    "evaluate_algorithm",
    "run_optuna_study",
]
