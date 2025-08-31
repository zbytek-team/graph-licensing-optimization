# Ant Colony Optimization

- Idea: buduje rozwiązania przy użyciu feromonów i heurystyk; najlepsze ścieżki wzmacniane depozycją feromonów.

## Pseudocode
```
initialize τ[n,ℓ] = 1.0 for each node n and license ℓ
η[n,ℓ] = (ℓ.max_capacity/ℓ.cost) * (1 + deg(n))  # heurystyka efektywności
best = greedy_solution(); deposit(τ, best)
for it in 1..max_iterations:
  improved = false
  repeat num_ants times:
    uncovered ← V; groups ← ∅
    while uncovered ≠ ∅:
      # wybór właściciela – ruletka po sumie (τ^α · η^β) dla wszystkich ℓ
      owner ~ roulette_over({n∈uncovered}, score(n) = Σ_ℓ (τ[n,ℓ]^α · η[n,ℓ]^β))
      # wybór licencji dla ownera – ruletka po (τ^α · η^β)
      ℓ ~ roulette_over(L, score(ℓ) = τ[owner,ℓ]^α · η[owner,ℓ]^β)
      pool ← (N(owner)∪{owner}) ∩ uncovered
      if |pool| < ℓ.min:  # naprawa niedopasowania
        ℓ' ← cheapest feasible license for |pool| or singleton if min=1
        assign accordingly
      else:
        add_count ← ℓ.max-1; A ← top_{add_count} by degree from pool\{owner}
        add group (ℓ, {owner}∪A); uncovered ← uncovered \ ({owner}∪A)
    sol ← groups
    if valid(sol) and cost(sol) < cost(best): best ← sol; improved = true
  evaporate(τ, rate=ρ)
  deposit(τ, best, amount=1/cost(best))
return best
```

## Złożoność
- Czas: O(iterations × ants × (V + E)) w przybliżeniu (konstrukcja i ocena rozwiązań).
- Pamięć: O(V × |licenses|) na tablice feromonów/heurystyk.

## Wyniki z ostatniego custom.py
- Graph: small_world(n=100, k=4, p=0.1), License: spotify
- cost: 945.74, time_ms: 1517.14, valid: True

## Uwagi
- Parametry (α, β, ρ, q0, liczba mrówek/iteracji) silnie wpływają na wynik; q0 steruje eksploatacją (best) vs eksploracją (ruletka).
- Konstrukcja korzysta z top‑stopnia dla doboru członków w granicach pojemności.
