# Tree Dynamic Programming

- Idea: programowanie dynamiczne po drzewie (rozwiązanie dokładne dla grafów-drzew).

## Pseudocode
```
def solve_subtree(u, parent):
  children ← N(u) \ {parent}
  if children=∅:
    return (cost=cost_of_cheapest_license_for_size(1), groups=[(ℓ,{u})])
  child_best = {v: solve_subtree(v,u) for v in children}
  best_cost ← +∞; best_groups ← []
  for ℓ in L:
    for x in 0..min(|children|, ℓ.max-1):  # liczba dołączonych dzieci do grupy u
      if 1+x < ℓ.min: continue
      for C ⊆ children, |C|=x:
        cost ← cost(ℓ)
        groups ← [(ℓ, {u}∪C)]
        # dzieci nie dołączone: całe ich poddrzewa
        for v in children\C: cost += child_best[v].cost; groups += child_best[v].groups
        # dzieci dołączone: bierzemy wnuki (pomijamy dołączone dziecko)
        for v in C: cost += cost_of_grandchildren(v, u)
        if cost < best_cost: best_cost, best_groups = cost, groups
  return (best_cost, best_groups)

pick arbitrary root r; return solve_subtree(r, None)
```

## Złożoność
- Czas: O(V · f(deg)) - liniowe względem V dla stałego stopnia.
- Pamięć: O(V).

## Wyniki z ostatniego custom.py
- Brak (algorytm dotyczy tylko drzew; nie użyty w obecnym biegu).

## Uwagi
- Gwarantuje optimum na drzewach; do grafów ogólnych wymaga innej dekompozycji (np. drzew rozpinających + heurystyki).

## Mapowanie pseudokodu na kod
- Plik: `src/glopt/algorithms/tree_dp.py`
- `solve(...)` - wybór korzenia i odpalenie `_solve_subtree`
- `_solve_subtree(...)` - rozważanie licencji i łączenie części dzieci do grupy właściciela, reszta przez poddrzewa
- `_solve_child_subtree(...)` - koszt wnuków dla dzieci włączonych do grupy właściciela
