# Randomized Algorithm

- Idea: Jednoprzebiegowy algorytm, który dla każdego wierzchołka decyduje losowo między wariantem zachłannym a czysto losowym.

## Pseudocode
```
input: graph G=(V,E), license types L, greedy_probability p∈[0,1]
uncovered ← V
order ← shuffled list of V
groups ← ∅
for node in order:
  if node ∉ uncovered: continue
  if rand() < p:
    # greedy-assignment
    best ← None; best_eff ← +∞
    S ← (N(node)∪{node}) ∩ uncovered
    for each license ℓ in L:
      for size s from ℓ.min to min(|S|, ℓ.max):
        members ← {node} ∪ top_{s-1} nodes by degree from S\{node}
        eff ← ℓ.cost / s
        if eff < best_eff: best ← (ℓ, members); best_eff ← eff
  else:
    # random-assignment
    S ← (N(node)∪{node}) ∩ uncovered
    C ← {ℓ ∈ L : ℓ.min ≤ |S|}
    if C≠∅:
      ℓ ← random element of C
      s ← random integer in [ℓ.min, min(|S|, ℓ.max)]
      members ← {node} ∪ random (s-1) nodes from S\{node}
      best ← (ℓ, members)
  if best exists:
    add group(best) to groups; uncovered ← uncovered \ best.members

# domknięcie resztek pojedynczymi
while uncovered ≠ ∅:
  u ← pop(uncovered)
  ℓ1 ← cheapest license with min_capacity ≤ 1
  add (ℓ1, {u}) to groups

return groups
```

## Złożoność
- Czas: O(V·Δ log Δ) przy wariancie greedy (sortowanie sąsiadów), losowy wariant ~O(V·Δ).
- Pamięć: O(V).

## Wyniki z ostatniego custom.py
- Graph: small_world(n=100), License: spotify
- cost: 1171.64, time_ms: 0.25, valid: True

## Uwagi
- Parametr p steruje kompromisem jakość/czas. Fallback uzupełnia niepokryte węzły najtańszymi singlami.
