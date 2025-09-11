# Genetic Algorithm

- Idea: populacja rozwiązań z elityzmem, selekcją turniejową, krzyżowaniem i mutacją opartą o sąsiedztwa. Startuję populację silnym seedem: greedy oraz ewentualnie rozwiązaniem warm start, resztę dopełniam losowo.

## Pseudocode
```
input: population_size P, generations G, elite_fraction f, crossover_rate cr
Population ← []
if warm_start valid: Population.append(warm_start)
Population.append(Greedy())
while |Population| < P: Population.append(Randomized())
best ← min(Population, by cost)
for gen in 1..G:
  sort Population by cost asc
  Elite ← top ⌈f·P⌉
  New ← Elite
  while |New| < P:
    if rand() < cr and |Population|≥2:
      p1 ← tournament_select(Population, k=3)
      p2 ← tournament_select(Population, k=3)
      child ← crossover_merge(p1,p2)  # łączymy grupy po efektywności i domykamy greedy na reszcie
      if not valid(child): child ← mutate(best_of(p1,p2))
    else:
      parent ← tournament_select(Population, k=3)
      child ← mutate(parent)          # MutationOperators.generate_neighbors + walidacja
    New.append(child)
  Population ← New
  best ← min(best, min(Population))
return best
```

## Złożoność
- Czas: O(G × (P log P + P × (koszt_krzyżowania lub koszt_mutacji))). Koszt mutacji to generacja sąsiadów i walidacja ~O(V+E) na k kandydatów.
- Pamięć: O(P × |solution|).

## Uwagi
- Krzyżowanie łączy grupy rodziców według efektywności i domyka luki rozwiązaniem greedy na podgrafie niepokrytych węzłów.
- Mutacja to wybór najlepszego sąsiada z puli ruchów lokalnych, tylko rozwiązania poprawne przechodzą dalej.

## Mapowanie pseudokodu na kod
- Plik: `src/glopt/algorithms/genetic.py`
- `solve(...)` - główna pętla GA, elityzm, selekcja, krzyżowanie i mutacja
- `_init_population(...)` - warm start + greedy + losowe rozwiązania
- `_tournament_selection(...)` - selekcja turniejowa
- `_mutate(...)` - sąsiedztwa z `MutationOperators`, walidacja i ewentualny fallback do greedy
- `_crossover(...)` - łączenie grup po efektywności i domknięcie greedy na niepokrytych
