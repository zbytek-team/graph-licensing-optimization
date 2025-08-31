# ILP (Integer Linear Programming)

- Idea: model ILP minimalizujący sumaryczny koszt grup pod warunkami pokrycia wierzchołków i pojemności licencji.

## Pseudocode (model)
```
Variables:
  x[i,j,t] ∈ {0,1}  # przypisanie węzła j do grupy (owner=i, license=t)
  a[i,t]   ∈ {0,1}  # aktywacja grupy (owner=i, license=t)
Objective:
  minimize Σ_i Σ_t a[i,t] * cost(t)
Constraints:
  ∀j: Σ_i in N(j)∪{j} Σ_t x[i,j,t] = 1
  ∀i,t: Σ_j in N(i)∪{i} x[i,j,t] ≤ a[i,t] * max_cap(t)
  ∀i,t: Σ_j in N(i)∪{i} x[i,j,t] ≥ a[i,t] * min_cap(t)
  ∀i,t: x[i,i,t] ≥ a[i,t]
```

## Złożoność
- Czas: w najgorszym wypadku wykładnicza (NP‑trudne). W praktyce rozsądne dla małych/średnich grafów.
- Pamięć: zależna od solvera i liczby zmiennych O(V·deg + V·|licenses|).

## Wyniki z ostatniego custom.py
- Graph: small_world(n=100), License: spotify
- cost: 790.79, time_ms: 5662.15, valid: True

## Uwagi
- Daje zwykle najlepszą jakość (dolna granica porównawcza), ale jest najwolniejszy.
