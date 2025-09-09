# Simulated Annealing

- Idea: start z rozwiązania zachłannego, a następnie losowe ruchy z akceptacją pogorszeń zależną od temperatury.

## Pseudocode
```
current ← greedy_solution()
best ← current
T ← initial_temperature
stall ← 0
repeat up to max_iterations:
  if T < min_temperature: break
  cand ← random_neighbor(current)  # jeden z {change_license, move_member, swap_members, merge_groups, split_group}
  if cand is None: stall++
  else:
    Δ ← cand.cost - current.cost
    if Δ < 0 or rand() < exp(-Δ / max(T, ε)):
      current ← cand
      if current.cost < best.cost: best ← current; stall ← 0
      else: stall++
    else: stall++
  if stall ≥ max_stall: stall ← 0; T ← max(min_temperature, 0.5·T)
  T ← cooling_rate · T
return best
```

Ruchy są generowane tak, by zachować ograniczenia licencji i sąsiedztwa (walidacja rozwiązania po ruchu).

## Złożoność
- Czas: O(iterations × koszt_sąsiedztwa). Generowanie i walidacja sąsiadów ~O(V+E).
- Pamięć: O(V).

## Wyniki z ostatniego custom.py
- Graph: small_world(n=100), License: spotify
- cost: 1003.71, time_ms: 203.36, valid: True

## Uwagi
- Dobór ruchów i harmonogram chłodzenia silnie wpływają na jakość; walidacja filtruje ruchy niepoprawne.

## Mapowanie pseudokodu na kod
- Plik: `src/glopt/algorithms/simulated_annealing.py`
- `solve(...)` – główna pętla SA, chłodzenie i licznik zastoju
- `_neighbor(...)` – losowanie typu ruchu i walidacja sąsiada
- Ruchy loklane: `_mv_change_license`, `_mv_move_member`, `_mv_swap_members`, `_mv_merge_groups`, `_mv_split_group`
- `_fallback_singletons(...)` – bezpieczny start gdy greedy nie przejdzie walidacji
