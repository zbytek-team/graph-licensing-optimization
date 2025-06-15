"""Genetic Algorithm parameter tuning."""

from typing import TYPE_CHECKING, List, Dict, Any
import optuna

if TYPE_CHECKING:
    import networkx as nx
    from ..models.license import LicenseConfig

from .utils import evaluate_algorithm, run_optuna_study


def tune_genetic_algorithm(
    graphs: List["nx.Graph"],
    configs: List["LicenseConfig"],
    metric: str = "cost",
    n_trials: int = 100,
    timeout: float = None,
    n_jobs: int = 1,
    seed: int = None,
) -> Dict[str, Any]:
    """Tune Genetic Algorithm parameters using Optuna."""

    def objective(trial: optuna.Trial) -> float:
        # Parameter suggestions
        mutation_rate = trial.suggest_float("mutation_rate", 0.01, 0.5)
        population_size = 200
        generations = 500
        crossover_rate = trial.suggest_float("crossover_rate", 0.1, 0.99)

        # Import here to avoid circular imports
        from ..algorithms.genetic.genetic import GeneticAlgorithm

        algorithm = GeneticAlgorithm(
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            seed=seed,
        )

        return evaluate_algorithm(algorithm, graphs, configs, metric)

    return run_optuna_study(
        objective_func=objective,
        study_name="genetic_algorithm",
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        seed=seed,
    )
