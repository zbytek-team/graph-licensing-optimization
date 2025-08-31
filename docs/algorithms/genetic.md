# Genetic Algorithm

- Idea: populacja rozwiązań, selekcja turniejowa, “mutacja” poprzez ruchy sąsiedztwa, elityzm.

## Pseudocode
```
input: population_size P, generations G, elite_fraction f
Population ← [Randomized() for _ in 1..P]
best ← argmin(Population, cost)
for gen in 1..G:
  sort Population by cost asc
  Elite ← top ⌈f·P⌉ solutions
  New ← Elite
  while |New| < P:
    parent ← tournament_select(Population, k=3)
    child ← best_valid_neighbor(parent, k=5)  # via MutationOperators, pick valid with min cost
    New.append(child)
  Population ← New
  best ← min(best, min(Population))
return best
```

## Złożoność
- Czas: O(G × (P log P + P × koszt_mutacji)), koszt_mutacji ~ generacja sąsiadów + walidacja.
- Pamięć: O(P × |solution|).

## Wyniki z ostatniego custom.py
- Graph: small_world(n=100), License: spotify
- cost: 900.75, time_ms: 435.45, valid: True

## Uwagi
- “Mutacja” bazuje na ruchach lokalnych (MutationOperators); brak jawnego krzyżowania — prostsza i stabilna wersja GA.
