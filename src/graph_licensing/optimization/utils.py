"""Common utilities for Optuna optimization."""

import logging
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import networkx as nx
    from ..algorithms.base import BaseAlgorithm
    from ..models.license import LicenseConfig


def evaluate_algorithm(
    algorithm: "BaseAlgorithm",
    graphs: List["nx.Graph"],
    configs: List["LicenseConfig"],
    metric: str,
) -> float:
    """Evaluate algorithm performance on test instances."""
    logger = logging.getLogger(__name__)
    scores = []
    
    for graph, config in zip(graphs, configs):
        try:
            if metric == "cost":
                solution = algorithm.solve(graph, config)
                scores.append(solution.calculate_cost(config))
            elif metric == "runtime":
                import time
                start_time = time.perf_counter()
                algorithm.solve(graph, config)
                end_time = time.perf_counter()
                scores.append(end_time - start_time)
            elif metric == "quality":
                solution = algorithm.solve(graph, config)
                total_nodes = graph.number_of_nodes()
                group_members = sum(
                    sum(len(members) for members in groups.values()) 
                    for groups in solution.licenses.values()
                )
                quality = group_members / total_nodes if total_nodes > 0 else 0
                scores.append(-quality)  # Negative because we minimize
        except Exception as e:
            logger.warning(f"Algorithm failed on test instance: {e}")
            scores.append(float("inf"))
            
    return sum(scores) / len(scores) if scores else float("inf")


def run_optuna_study(
    objective_func,
    study_name: str,
    n_trials: int = 100,
    timeout: float = None,
    n_jobs: int = 1,
    seed: int = None,
) -> dict:
    """Run Optuna optimization study."""
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        study_name=study_name,
    )
    
    study.optimize(
        objective_func,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )
    
    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "study": study,
    }
