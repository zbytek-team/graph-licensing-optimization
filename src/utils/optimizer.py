import optuna
from src.solvers.static import AntColonySolver
from src.generators import ScaleFreeGenerator

INDIVIDUAL_COST = 5.0
GROUP_COST = 8.0
GROUP_SIZE = 6

generator = ScaleFreeGenerator()
graph = generator.generate(200)


def objective(trial):
    ant_count = trial.suggest_int("ant_count", 32, 128)
    alpha = trial.suggest_float("alpha", 1, 16)
    beta = trial.suggest_float("beta", 1, 16)
    evaporation_rate = trial.suggest_float("evaporation_rate", 0.01, 1)
    iterations = trial.suggest_int("iterations", 256, 1024)

    try:
        solver = AntColonySolver(
            INDIVIDUAL_COST,
            GROUP_COST,
            GROUP_SIZE,
            ant_count=ant_count,
            alpha=alpha,
            beta=beta,
            evaporation_rate=evaporation_rate,
            iterations=iterations,
        )
        result = solver.run(graph)

    except Exception:
        return float("inf")

    return result["total_cost"]


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=500)

print("\n=== Najlepsze parametry dla Tabu Search ===")
print(study.best_params)
print(f"Total cost: {study.best_value}")
