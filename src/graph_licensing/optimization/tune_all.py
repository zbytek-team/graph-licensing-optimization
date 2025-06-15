"""Multi-algorithm tuning utilities."""

import logging
from typing import TYPE_CHECKING, List, Dict, Any, Optional

if TYPE_CHECKING:
    import networkx as nx
    from ..models.license import LicenseConfig

from .tune_genetic import tune_genetic_algorithm
from .tune_simulated_annealing import tune_simulated_annealing
from .tune_tabu_search import tune_tabu_search


def tune_all_algorithms(
    graphs: List["nx.Graph"],
    configs: List["LicenseConfig"],
    algorithms: Optional[List[str]] = None,
    metric: str = "cost",
    n_trials: int = 100,
    timeout: float = None,
    n_jobs: int = 1,
    seed: int = None,
) -> Dict[str, Dict[str, Any]]:
    """Tune parameters for multiple algorithms."""
    logger = logging.getLogger(__name__)

    if algorithms is None:
        algorithms = ["genetic", "simulated_annealing", "tabu_search"]

    results = {}
    tuning_functions = {
        "genetic": tune_genetic_algorithm,
        "simulated_annealing": tune_simulated_annealing,
        "tabu_search": tune_tabu_search,
    }

    for algorithm_name in algorithms:
        if algorithm_name not in tuning_functions:
            logger.warning(f"Unknown algorithm: {algorithm_name}")
            continue

        logger.info(f"Tuning {algorithm_name}...")

        tuning_func = tuning_functions[algorithm_name]
        results[algorithm_name] = tuning_func(
            graphs=graphs,
            configs=configs,
            metric=metric,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            seed=seed,
        )

    return results


def get_best_parameters(algorithm_name: str, tuning_results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract best parameters from tuning results."""
    if algorithm_name not in tuning_results:
        raise ValueError(f"No tuning results found for algorithm: {algorithm_name}")

    return tuning_results[algorithm_name].get("best_params", {})


def create_tuned_algorithm(algorithm_name: str, best_params: Dict[str, Any], seed: int = None):
    """Create algorithm instance with tuned parameters."""
    if algorithm_name == "genetic":
        from ..algorithms.genetic.genetic import GeneticAlgorithm

        return GeneticAlgorithm(**best_params, seed=seed)
    elif algorithm_name == "simulated_annealing":
        from ..algorithms.simulated_annealing.simulated_annealing import SimulatedAnnealingAlgorithm

        return SimulatedAnnealingAlgorithm(**best_params, seed=seed)
    elif algorithm_name == "tabu_search":
        from ..algorithms.tabu_search.tabu_search import TabuSearchAlgorithm

        return TabuSearchAlgorithm(**best_params, seed=seed)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
