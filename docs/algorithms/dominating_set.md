# Dominating Set Heuristic

- Idea: najpierw buduje tani zbiór dominujący, potem każdemu dominatorowi przypisuje najtańszą dopuszczalną grupę; resztę domyka podobnie.

## Pseudocode
```
input: G=(V,E), licenses L
# 1) budowa taniego zbioru dominującego
uncovered ← V; D ← ∅
while uncovered ≠ ∅:
  for v in V:
    cov ← (N(v)∪{v}) ∩ uncovered
    if cov=∅: continue
    min_cpn[v] ← min_{ℓ∈L, ℓ.min≤|cov|≤ℓ.max} ℓ.cost/|cov|  (∞ jeśli brak)
    score[v] ← |cov| / min_cpn[v]  (jeśli min_cpn[v]=∞, pomiń)
  u ← argmax_v score[v] (lub dowolny z uncovered jeśli brak score)
  D ← D ∪ {u}; uncovered ← uncovered \ (N(u)∪{u})

# 2) przypisanie grup dominatorom
groups ← ∅; remaining ← V
for u in D (malejąco po deg(u)):
  S ← (N(u)∪{u}) ∩ remaining
  best ← argmin_{ℓ∈L, s∈[ℓ.min, min(|S|,ℓ.max)]} ℓ.cost/s, gdzie członkowie to {u}∪top_{s-1} po deg z S\{u}
  if best exists: add group(best); remaining ← remaining \ best.members

# 3) domknięcie pozostałych
for v in remaining (malejąco po deg(v)):
  S ← (N(v)∪{v}) ∩ remaining
  best ← jw. (najtańsza dopuszczalna grupa) lub w ostateczności pojedyncza najtańsza licencja
  add group(best); remaining ← remaining \ best.members

return groups
```

## Złożoność
- Czas: ~O(V·(Δ log Δ + |L|·Δ)) -- oceny pokrycia i wybór grup według stopnia.
- Pamięć: O(V).

## Wyniki z ostatniego custom.py
- Graph: small_world(n=100), License: spotify
- cost: 872.75, time_ms: 1.28, valid: True

## Uwagi
- Heurystyka “coverage per cost” kieruje wyborem dominatorów; dobór członków grup preferuje węzły o największym stopniu.

## Mapowanie pseudokodu na kod
- Plik: `src/glopt/algorithms/dominating_set.py`
- `solve(...)` - główna pętla: wyznaczenie zbioru dominującego, przypisania grup i domknięcia
- `_find_cost_effective_dominating_set(...)` - budowa D według pokrycia i kosztu na węzeł
- `_find_best_cost_assignment(...)` - najtańsza dopuszczalna grupa dla lidera
- `_select_best_group_members(...)` - wybór członków według stopnia
- `_find_cheapest_single_license(...)` - fallback do licencji 1
