"""Simulated Annealing parameter tuning."""

from typing import TYPE_CHECKING, List, Dict, Any
import optuna

if TYPE_CHECKING:
    import networkx as nx
    from ..models.license import LicenseConfig

from .utils import evaluate_algorithm, run_optuna_study


def tune_simulated_annealing(
    graphs: List["nx.Graph"],
    configs: List["LicenseConfig"],
    metric: str = "cost",
    n_trials: int = 100,
    timeout: float = None,
    n_jobs: int = 1,
    seed: int = None,
) -> Dict[str, Any]:
    """Tune Simulated Annealing parameters using Optuna."""

    def objective(trial: optuna.Trial) -> float:
        # Parameter suggestions
        initial_temp = trial.suggest_float("initial_temp", 50.0, 500.0)
        final_temp = trial.suggest_float("final_temp", 0.01, 1.0)
        cooling_rate = trial.suggest_float("cooling_rate", 0.8, 0.99)
        max_iterations = trial.suggest_int("max_iterations", 500, 2000, step=100)

        # Import here to avoid circular imports
        from ..algorithms.simulated_annealing.simulated_annealing import SimulatedAnnealingAlgorithm

        algorithm = SimulatedAnnealingAlgorithm(
            initial_temp=initial_temp,
            final_temp=final_temp,
            cooling_rate=cooling_rate,
            max_iterations=max_iterations,
            seed=seed,
        )

        return evaluate_algorithm(algorithm, graphs, configs, metric)

    return run_optuna_study(
        objective_func=objective,
        study_name="simulated_annealing",
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        seed=seed,
    )
