# Greedy Algorithm

- Idea: iteracyjnie wybiera najtańsze dopuszczalne grupy maksymalizujące pokrycie sąsiadów.

## Pseudocode
```
input: G=(V,E), licenses L
uncovered ← V; groups ← ∅
for owner in V sorted by deg(owner) desc:
  if owner ∉ uncovered: continue
  S ← (N(owner)∪{owner}) ∩ uncovered
  if S=∅: continue
  # wybór najlepszej grupy: minimalny koszt na wierzchołek
  best ← None; best_eff ← +∞
  for ℓ in L sorted by (-ℓ.max_capacity, ℓ.cost):
    k ← min(ℓ.max_capacity-1, |S\{owner}|)
    members ← {owner} ∪ top_k by degree from S\{owner}
    if |members| < ℓ.min_capacity: continue
    eff ← ℓ.cost / |members|
    if eff < best_eff: best ← (ℓ, members); best_eff ← eff
  if best: add group(best); uncovered ← uncovered \ best.members

# domknięcie pozostałych
while uncovered ≠ ∅:
  u ← any element of uncovered
  S ← (N(u)∪{u}) ∩ uncovered
  # spróbuj najtańszą dopuszczalną grupę o min_capacity
  for ℓ in L sorted by (ℓ.cost, -ℓ.max_capacity):
    if |S| ≥ ℓ.min_capacity:
      members ← {u} ∪ top_{ℓ.min_capacity-1} from S\{u}
      add (ℓ, members); uncovered ← uncovered \ members
      break
  else:
    # jeśli nie ma grupy >1, weź najtańszą licencję jednoosobową
    ℓ1 ← argmin_{ℓ∈L} ℓ.cost s.t. ℓ.min≤1≤ℓ.max
    add (ℓ1, {u}); uncovered ← uncovered \ {u}

return groups
```

## Złożoność
- Czas: ~O(E log V) – selekcja właścicieli i sortowania po stopniu/efektywności.
- Pamięć: O(V).

## Wyniki z ostatniego custom.py
- Graph: small_world(n=100), License: spotify
- cost: 1003.71, time_ms: 0.20, valid: True

## Uwagi
- Bardzo szybki, zazwyczaj rozsądna jakość; dobry baseline.

## Mapowanie pseudokodu na kod
- Plik: `src/glopt/algorithms/greedy.py`
- `solve(...)` – pętla po właścicielach w kolejności po stopniu oraz faza domknięcia resztek
- `_best_group_for_owner(...)` – wybór licencji i członków grupy minimalizujący koszt na węzeł
- `_cheapest_feasible_group(...)` – domknięcie najmniejszą dopuszczalną grupą albo licencją 1
