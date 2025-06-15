import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

if TYPE_CHECKING:
    import networkx as nx

    from ..algorithms.base import BaseAlgorithm
    from ..models.license import LicenseConfig


class OptunaTuner:
    def __init__(
        self,
        n_trials: int = 100,
        timeout: Optional[float] = None,
        n_jobs: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.seed = seed
        self.logger = logging.getLogger(__name__)

    def tune_genetic_algorithm(
        self,
        graphs: List["nx.Graph"],
        configs: List["LicenseConfig"],
        metric: str = "cost",
        minimize: bool = True,
    ) -> Dict[str, Any]:
        def objective(trial: optuna.Trial) -> float:
            mutation_rate = trial.suggest_float("mutation_rate", 0.01, 0.5)
            population_size = 200
            generations = 500
            crossover_rate = trial.suggest_float("crossover_rate", 0.1, 0.99)
            from ..algorithms.genetic.genetic import GeneticAlgorithm

            algorithm = GeneticAlgorithm(
                population_size=population_size,
                generations=generations,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                seed=self.seed,
            )
            return self._evaluate_algorithm(algorithm, graphs, configs, metric)

        return self._run_optimization(objective, "genetic_algorithm")

    def tune_simulated_annealing(
        self,
        graphs: List["nx.Graph"],
        configs: List["LicenseConfig"],
        metric: str = "cost",
        minimize: bool = True,
    ) -> Dict[str, Any]:
        def objective(trial: optuna.Trial) -> float:
            initial_temp = trial.suggest_float("initial_temp", 50.0, 500.0)
            final_temp = trial.suggest_float("final_temp", 0.01, 1.0)
            cooling_rate = trial.suggest_float("cooling_rate", 0.8, 0.99)
            max_iterations = trial.suggest_int("max_iterations", 500, 2000, step=100)
            from ..algorithms.simulated_annealing.simulated_annealing import SimulatedAnnealingAlgorithm

            algorithm = SimulatedAnnealingAlgorithm(
                initial_temp=initial_temp,
                final_temp=final_temp,
                cooling_rate=cooling_rate,
                max_iterations=max_iterations,
                seed=self.seed,
            )
            return self._evaluate_algorithm(algorithm, graphs, configs, metric)

        return self._run_optimization(objective, "simulated_annealing")

    def tune_tabu_search(
        self,
        graphs: List["nx.Graph"],
        configs: List["LicenseConfig"],
        metric: str = "cost",
        minimize: bool = True,
    ) -> Dict[str, Any]:
        def objective(trial: optuna.Trial) -> float:
            max_iterations = trial.suggest_int("max_iterations", 50, 500, step=25)
            max_no_improvement = trial.suggest_int("max_no_improvement", 10, 100, step=5)
            from ..algorithms.tabu_search.tabu_search import TabuSearchAlgorithm

            algorithm = TabuSearchAlgorithm(
                max_iterations=max_iterations,
                max_no_improvement=max_no_improvement,
                seed=self.seed,
            )
            return self._evaluate_algorithm(algorithm, graphs, configs, metric)

        return self._run_optimization(objective, "tabu_search")

    def _evaluate_algorithm(
        self,
        algorithm: "BaseAlgorithm",
        graphs: List["nx.Graph"],
        configs: List["LicenseConfig"],
        metric: str,
    ) -> float:
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
                        sum(len(members) for members in groups.values()) for groups in solution.licenses.values()
                    )
                    quality = group_members / total_nodes if total_nodes > 0 else 0
                    scores.append(-quality)
            except Exception as e:
                self.logger.warning(f"Algorithm failed on test instance: {e}")
                scores.append(float("inf"))
        return sum(scores) / len(scores) if scores else float("inf")

    def _run_optimization(
        self,
        objective: Callable[[optuna.Trial], float],
        study_name: str,
    ) -> Dict[str, Any]:
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=self.seed),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            study_name=study_name,
        )
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            show_progress_bar=True,
        )
        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "study": study,
        }

    def tune_all_algorithms(
        self,
        graphs: List["nx.Graph"],
        configs: List["LicenseConfig"],
        algorithms: Optional[List[str]] = None,
        metric: str = "cost",
    ) -> Dict[str, Dict[str, Any]]:
        if algorithms is None:
            algorithms = ["genetic", "simulated_annealing", "tabu_search"]
        results = {}
        for algorithm_name in algorithms:
            self.logger.info(f"Tuning {algorithm_name}...")
            if algorithm_name == "genetic":
                results[algorithm_name] = self.tune_genetic_algorithm(graphs, configs, metric)
            elif algorithm_name == "simulated_annealing":
                results[algorithm_name] = self.tune_simulated_annealing(graphs, configs, metric)
            elif algorithm_name == "tabu_search":
                results[algorithm_name] = self.tune_tabu_search(graphs, configs, metric)
            else:
                self.logger.warning(f"Unknown algorithm: {algorithm_name}")
        return results
