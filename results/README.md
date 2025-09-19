# Wyniki -- Podsumowanie

Ten plik zbiera najważniejsze wnioski z analiz w katalogach `results/benchmark_all` oraz `results/benchmark_real_all`.

## benchmark_all

- Liczba grafów: 3
- Najniższy średni koszt: ILPSolver (≈ 343.249)
- Najlepszy średni czas: RandomizedAlgorithm (≈ 3.088 ms)
- Występuje trade-off koszt vs czas -- różni zwycięzcy.

### Interpretacja

- Ranking kosztu (Friedman): 1) ILPSolver (avg rank≈1.14), dalej: AntColonyOptimization(3.17), GeneticAlgorithm(3.56)
- Ranking czasu (Friedman): 1) GreedyAlgorithm (avg rank≈1.19), dalej: RandomizedAlgorithm(1.86), DominatingSetAlgorithm(2.95)
- Empiryczna złożoność czasu: najniższe b (time≈a·n^b): GreedyAlgorithm (b≈0.68), RandomizedAlgorithm (b≈0.69), TabuSearch (b≈0.87)
- Istotne różnice kosztu (Wilcoxon p<0.05): AntColonyOptimization (7), GreedyAlgorithm (7)
- Istotne różnice czasu (Wilcoxon p<0.05): DominatingSetAlgorithm (7), GeneticAlgorithm (7)
- Leaderboard kosztu (wygrane per instancja zgrupowane po binach n): ILPSolver (203), AntColonyOptimization (89)
- Leaderboard czasu (wygrane per instancja zgrupowane po binach n): GreedyAlgorithm (285), RandomizedAlgorithm (114)

## benchmark_real_all

- Liczba grafów: 1
- Najniższy średni koszt: ILPSolver (≈ 11.721)
- Najlepszy średni czas: GreedyAlgorithm (≈ 2.107 ms)
- Występuje trade-off koszt vs czas -- różni zwycięzcy.

### Interpretacja

- Ranking kosztu (Friedman): 1) ILPSolver (avg rank≈3.39), dalej: TabuSearch(3.43), SimulatedAnnealing(3.48)
- Ranking czasu (Friedman): 1) GreedyAlgorithm (avg rank≈1.26), dalej: RandomizedAlgorithm(2.17), TabuSearch(2.72)
- Empiryczna złożoność czasu: najniższe b (time≈a·n^b): GreedyAlgorithm (b≈0.27), RandomizedAlgorithm (b≈0.42), TabuSearch (b≈0.86)
- Istotne różnice kosztu (Wilcoxon p<0.05): RandomizedAlgorithm (6), DominatingSetAlgorithm (1)
- Istotne różnice czasu (Wilcoxon p<0.05): DominatingSetAlgorithm (6), GeneticAlgorithm (6)
- Leaderboard kosztu (wygrane per instancja zgrupowane po binach n): GreedyAlgorithm (42), ILPSolver (28)
- Leaderboard czasu (wygrane per instancja zgrupowane po binach n): GreedyAlgorithm (46), RandomizedAlgorithm (14)

## Figury i tabele

- Pełne wykresy i tabele: patrz odpowiednio `results/benchmark_all/**` i `results/benchmark_real_all/**`.
