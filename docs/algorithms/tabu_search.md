# Tabu Search

- Idea: przeszukiwanie lokalne z pamięcią tabu; akceptuje najlepszych sąsiadów nie będących tabu (z aspiracją, gdy poprawiają best).

## Pseudocode
```
current ← greedy_solution(); best ← current
tabu ← FIFO(maxlen=tabu_tenure); tabu.push(hash(current))
for iter in 1..max_iterations:
  N ← generate_neighbors(current, k=neighbors_per_iter)
  if N=∅: break
  chosen ← None; chosen_cost ← +∞
  for cand in N:
    if not valid(cand): continue
    h ← hash(cand)
    if h ∈ tabu and cand.cost ≥ best.cost: continue   # reguła tabu z aspiracją
    if cand.cost < chosen_cost: chosen ← cand; chosen_cost ← cand.cost
  if chosen is None: break
  current ← chosen
  if current.cost < best.cost: best ← current
  tabu.push(hash(current))
return best
```

## Złożoność
- Czas: O(max_iterations × k × koszt_walidacji), k = neighbors_per_iter.
- Pamięć: O(tabu_tenure).

## Wyniki z ostatniego custom.py
- Graph: small_world(n=100), License: spotify
- cost: 869.76, time_ms: 1151.82, valid: True

## Uwagi
- Jakość zależy od generowania sąsiedztwa i długości tabu; aspiracja pozwala “przebić” tabu lepszym rozwiązaniem.

## Mapowanie pseudokodu na kod
- Plik: `src/glopt/algorithms/tabu_search.py`
- `solve(...)` – pętla tabu, lista tabu jako `deque`, aspiracja gdy kandydat poprawia `best`
- Sąsiedztwa: `MutationOperators.generate_neighbors(...)` w `glopt/core/mutations.py`
- Walidacja: `SolutionValidator`
- `_hash(...)` – haszowanie konfiguracji grup do listy tabu
