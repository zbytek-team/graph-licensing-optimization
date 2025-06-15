"""Tabu Search parameter tuning."""

from typing import TYPE_CHECKING, List, Dict, Any
import optuna

if TYPE_CHECKING:
    import networkx as nx
    from ..models.license import LicenseConfig

from .utils import evaluate_algorithm, run_optuna_study


def tune_tabu_search(
    graphs: List["nx.Graph"],
    configs: List["LicenseConfig"],
    metric: str = "cost",
    n_trials: int = 100,
    timeout: float = None,
    n_jobs: int = 1,
    seed: int = None,
) -> Dict[str, Any]:
    """Tune Tabu Search parameters using Optuna."""
    
    def objective(trial: optuna.Trial) -> float:
        # Parameter suggestions
        max_iterations = trial.suggest_int("max_iterations", 50, 500, step=25)
        max_no_improvement = trial.suggest_int("max_no_improvement", 10, 100, step=5)
        
        # Import here to avoid circular imports
        from ..algorithms.tabu_search.tabu_search import TabuSearchAlgorithm
        
        algorithm = TabuSearchAlgorithm(
            max_iterations=max_iterations,
            max_no_improvement=max_no_improvement,
            seed=seed,
        )
        
        return evaluate_algorithm(algorithm, graphs, configs, metric)
    
    return run_optuna_study(
        objective_func=objective,
        study_name="tabu_search",
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        seed=seed,
    )
