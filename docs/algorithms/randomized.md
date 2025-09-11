# Randomized Algorithm

- Idea: jednoprzebiegowy algorytm, który w losowej kolejności próbuje dobrać losową licencję i losowy rozmiar grupy dla danego właściciela w granicach pojemności. Gdy nie da się dobrać licencji dla dostępnych sąsiadów, algorytm spada do prostego wariantu zachłannego dla tego wierzchołka. Na końcu domyka pozostałe wierzchołki najtańszą licencją jednoosobową.

## Pseudocode
```
input: graph G=(V,E), license types L
uncovered ← V
order ← shuffled list of V
groups ← ∅
for node in order:
  if node ∉ uncovered: continue
  S ← (N(node)∪{node}) ∩ uncovered
  C ← {ℓ ∈ L : ℓ.min ≤ |S|}
  best ← None
  if C≠∅:
    ℓ ← random element of C
    s ← random integer in [ℓ.min, min(|S|, ℓ.max)]
    members ← {node} ∪ random (s-1) nodes from S\{node}
    best ← (ℓ, members)
  else:
    # fallback: mały krok zachłanny
    best_eff ← +∞
    for ℓ in L:
      for s in [ℓ.min .. min(|S|, ℓ.max)]:
        cand ← {node} ∪ top_{s-1} by degree from S\{node}
        if |cand|=s and ℓ.cost/s < best_eff: best ← (ℓ,cand); best_eff ← ℓ.cost/s
  if best exists:
    add group(best); uncovered ← uncovered \ best.members

while uncovered ≠ ∅:  # domknięcie singlami
  u ← pop(uncovered)
  ℓ1 ← cheapest license with min_capacity ≤ 1
  add (ℓ1, {u})

return groups
```

## Złożoność
- Czas: ~O(V·Δ) dla części losowej, z okazjonalnym lokalnym krokiem zachłannym ~O(Δ log Δ).
- Pamięć: O(V).

## Uwagi
- Losowość zwiększa różnorodność rozwiązań, a prosty fallback pilnuje, żeby nie utknąć gdy nie ma dopasowania pojemności.

## Mapowanie pseudokodu na kod
- Plik: `src/glopt/algorithms/randomized.py`
- `solve(...)` - kolejność losowa, tworzenie grup losowych, domknięcie singlami
- `_random_assignment(...)` - dobór losowej licencji i rozmiaru plus fallback zachłanny
- `_select_random_group_members(...)` - losowy wybór członków
- `_find_cheapest_single_license(...)` - najtańsza licencja jednoosobowa
