# Project Dump: graph-licensing-optimization

Generated from: `/home/sobek/projects/graph-licensing-optimization`

## Project Structure

```
graph-licensing-optimization/
    ├── docs/
    │   ├── algorithms/
    │   │   ├── ant_colony.md
    │   │   ├── dominating_set.md
    │   │   ├── genetic.md
    │   │   ├── greedy.md
    │   │   ├── ilp.md
    │   │   ├── naive.md
    │   │   ├── randomized.md
    │   │   ├── simulated_annealing.md
    │   │   ├── tabu_search.md
    │   │   └── tree_dp.md
    │   └── thesis_readiness_report.md
    ├── runs/
    ├── scripts/
    │   ├── analysis/
    │   │   ├── __init__.py
    │   │   ├── commons.py
    │   │   ├── dynamic_warmcold.py
    │   │   ├── main.py
    │   │   ├── plots_cost_time.py
    │   │   ├── plots_density.py
    │   │   ├── plots_heatmap.py
    │   │   ├── plots_license_mix.py
    │   │   ├── plots_pareto.py
    │   │   ├── plots_profiles.py
    │   │   └── tables_aggregates.py
    │   └── analyze.py
    ├── src/
    │   └── glopt/
    │       ├── algorithms/
    │       │   ├── __init__.py
    │       │   ├── ant_colony.py
    │       │   ├── dominating_set.py
    │       │   ├── genetic.py
    │       │   ├── greedy.py
    │       │   ├── ilp.py
    │       │   ├── naive.py
    │       │   ├── randomized.py
    │       │   ├── simulated_annealing.py
    │       │   ├── tabu_search.py
    │       │   └── tree_dp.py
    │       ├── cli/
    │       │   ├── __init__.py
    │       │   ├── all.py
    │       │   ├── benchmark.py
    │       │   ├── benchmark_real.py
    │       │   ├── custom.py
    │       │   ├── dynamic.py
    │       │   └── dynamic_real.py
    │       ├── core/
    │       │   ├── __init__.py
    │       │   ├── models.py
    │       │   ├── mutations.py
    │       │   ├── run.py
    │       │   ├── solution_builder.py
    │       │   └── solution_validator.py
    │       ├── io/
    │       │   ├── __init__.py
    │       │   ├── csv_writer.py
    │       │   ├── data_loader.py
    │       │   ├── fs.py
    │       │   ├── graph_generator.py
    │       │   └── graph_visualizer.py
    │       ├── __init__.py
    │       ├── dynamic_simulator.py
    │       └── license_config.py
    ├── tests/
    │   ├── __init__.py
    │   └── test_algorithms.py
    ├── AGENTS.md
    ├── Makefile
    ├── pyproject.toml
    └── README.md
```

## File Contents

### AGENTS.md

```markdown
# Kontekst problemu

Badanie dotyczy **optymalizacji zakupu licencji oprogramowania w sieciach społecznościowych**.
Platformy SaaS (np. Spotify, Netflix, Duolingo) oferują plany indywidualne i grupowe.
Użytkownicy mogą współdzielić subskrypcje w ramach relacji społecznych, co prowadzi do pytania: **kto powinien kupić jaką licencję, aby łączny koszt dostępu był minimalny?**

Problem modelowany jest jako **graf G = (V, E)**:

* **V** – użytkownicy,
* **E** – relacje znajomości umożliwiające współdzielenie,
* licencje mają parametry: koszt (c₁ dla indywidualnej, c\_g dla grupowej) i pojemność (L użytkowników).

---

# Definicja problemu

* Każdy użytkownik może:

  1. kupić licencję indywidualną (koszt = 1),
  2. kupić licencję grupową (koszt = p, współdzieli ≤ L osób),
  3. być odbiorcą w grupie znajomego.
* Każdy węzeł musi mieć dostęp:

  * jeśli **0** (odbiorca), to musi mieć sąsiada **2** (posiadacza grupowej),
  * **1** (indywidualna) pokrywa wyłącznie siebie,
  * **2** (grupowa) pokrywa siebie + ≤ L–1 sąsiadów.

**Cel optymalizacji**: znaleźć podział V = I ∪ G ∪ R minimalizujący całkowity koszt:
C = |I| + p·|G|.

---

# Powiązania z teorią grafów

* Problem jest redukowalny do **dominowania w grafach**, a w wariancie p=2 → do **dominacji rzymskiej**.
* Znany jest jako **NP-trudny** (brak algorytmu wielomianowego, konieczność heurystyk i przybliżeń).
* Ograniczenia techniczne: limit L (np. Spotify: L=6, Netflix: L=4).
* Ograniczenia społeczne: współdzielenie wymaga bezpośredniej relacji (tylko sąsiedzi w grafie).

---

# Dane testowe

* **Grafy syntetyczne**:

  * Erdős–Rényi (losowe),
  * Barabási–Albert (bezskalowe, huby),
  * Watts–Strogatz (mały świat, wysoki clustering).
* **Grafy rzeczywiste**: ego-sieci z Facebooka (SNAP), 20–200 węzłów, wysokie clustering, wiele komponentów.

---

# Metody rozwiązania

### Algorytmy dokładne (dla małych grafów)

- ILP (programowanie całkowitoliczbowe): model minimalizujący sumę kosztów grup z ograniczeniami pojemności i pokrycia; solver PuLP/CBC; zwykle najlepsza jakość, rośnie czasowo wykładniczo.
- Naiwne przeszukiwanie: pełna enumeracja podziałów i właścicieli (praktycznie tylko dla n ≤ 10).
- Tree DP: dynamiczne programowanie po drzewie; gwarantuje optimum na grafach-drzewach (nie stosuje się bezpośrednio do grafów ogólnych).

### Algorytmy przybliżone i heurystyki

- Greedy (zachłanny): iteracyjnie wybiera najtańsze efektywne grupy (koszt/wielkość) z sąsiedztwa właściciela, domyka resztę; bardzo szybki baseline.
- Heurystyka zbioru dominującego: najpierw buduje „tani” zbiór dominujący (coverage per cost), potem każdemu dominatorowi przypisuje grupę; resztę domyka podobnie.
- Randomized: jednoprzebiegowy miks decyzji zachłannych i losowych dla każdego wierzchołka (sterowany parametrem prawdopodobieństwa).

### Metaheurystyki (dla większych grafów)

- Genetic Algorithm: populacja rozwiązań inicjalizowana losowo, selekcja turniejowa, „mutacja” przez ruchy lokalne (MutationOperators), elityzm; stabilna wersja GA bez jawnego krzyżowania.
- Tabu Search: lokalne przeszukiwanie z listą tabu oraz regułą aspiracji; w każdej iteracji wybiera najlepszego poprawnego sąsiada spoza tabu.
- Ant Colony Optimization (ACO): konstrukcja rozwiązań sterowana feromonami i heurystyką efektywności pojemności (τ, η, α, β, ρ, q0), z parowaniem i depozycją 1/cost.
- Simulated Annealing (SA): ruchy lokalne (zmiana licencji, przesunięcie/ zamiana członków, łączenie/ dzielenie grup) z akceptacją pogorszeń zależną od temperatury i harmonogramem chłodzenia.

---

# Eksperymenty

* Testy na grafach syntetycznych i rzeczywistych.

---

# Kluczowe wnioski

* Problem ma charakter **NP-trudny** – algorytmy dokładne działają tylko dla małych instancji.
* Heurystyki (greedy, dominujące) dają szybkie i dobre wyniki.
* Metaheurystyki (ACO, SA, Tabu, GA) zapewniają lepsze koszty, kosztem dłuższego czasu.
* Struktura grafu (huby, clustering, komponenty) silnie wpływa na wynik optymalizacji.

---

# Zasady implementacyjne (styl kodu)

Poniższe zasady są inspirowane stylem Google i dotyczą całej implementacji w repozytorium. Stosujemy je w miarę możliwości, zachowując lokalną spójność istniejącego kodu.

1. Ogólne zasady

- Uruchamiaj pylint i poprawiaj ostrzeżenia lub świadomie je wyłączaj z uzasadnieniem w komentarzu.
- Używaj autoformatowania (Black / Pyink) aby uniknąć sporów o styl.
- Pisz kod czytelny, a nie najkrótszy.

2. Język i konstrukcje

- Importy: tylko moduły/pakiety (`import x`, `from x import y`). Bez importów względnych.
- Wyjątki: używaj klas wbudowanych (`ValueError`, `TypeError`). Nie łap wszystkiego (`except:`). Używaj `finally` do sprzątania.
- Globalne mutable: unikać. Stałe pisz `ALL_CAPS_WITH_UNDER`.
- Funkcje zagnieżdżone: tylko jeśli zamykają nad zmienną.
- Comprehensions: proste OK, wielu zagnieżdżonych `for` unikaj.
- Iteratory: używaj `for x in dict`, `for line in file`. Nie `dict.keys()` ani `file.readlines()`.
- Generatory: w docstringach sekcja `Yields:`.
- Lambda: tylko krótkie (< 80 znaków).
- Wyrażenia warunkowe: krótkie w jednej linii; dłuższe → zwykłe `if`.
- Argumenty domyślne: nie mutowalne (np. `None` zamiast `[]`).
- Właściwości (`@property`): tylko gdy logika prosta i „oczywista”.
- Prawda/fałsz: używaj „implicit false” (`if not users:`). Dla `None` zawsze `is None`.
- Dekoratory: oszczędnie. `staticmethod` – unikać, `classmethod` tylko dla konstruktorów nazwanych.
- Wątki/procesy: nie polegaj na atomowości typów wbudowanych; używaj `queue.Queue`/`multiprocessing` i jawnych timeoutów.
- „Power features” (metaklasy, hacki importowe itp.): unikać.
- Type hints: dodawaj w API publicznym, w razie potrzeby sprawdzaj pytype/mypy.

3. Styl kodu

- Długość linii: max 80 znaków (wyjątki: importy, URL-e, `# noqa`, disable-komentarze).
- Wcięcia: 4 spacje. Bez tabów.
- Nawiasy: tylko gdy konieczne.
- Puste linie: 2 między funkcjami/klasami, 1 między metodami.
- Białe znaki: brak spacji wewnątrz nawiasów; spacje wokół operatorów binarnych.
- Komentarze i docstringi: `"""..."""`, pełne zdania. Sekcje: `Args:`, `Returns:`, `Raises:` (dla generatorów: `Yields:`). Krótkie komentarze inline → po dwóch spacjach `# ...`.
- Cudzysłowy: jednolity wybór `'` lub `"`; docstringi zawsze `"""`.
- Łączenie stringów: f-string, `%`, `.format()`. Nie `+` w pętli.
- Zamykanie zasobów: zawsze `with open(...) as f:`.
- TODO: format `# TODO: <link> - <opis>`. Nie używać nazw osób.
- Importy porządkuj blokami: future, stdlib, third-party, project. Sortowane leksykograficznie.

Nazewnictwo:

- moduły/pakiety → `lower_with_under`
- klasy/wyjątki → `CapWords`
- funkcje, metody, zmienne → `lower_with_under`
- stałe → `ALL_CAPS`

4. Organizacja

- Pliki wykonywalne: `if __name__ == "__main__": main()`.
- Funkcje krótkie, ≤ 40 linii gdy możliwe.
- Konsystencja lokalna ważniejsza niż globalna perfekcja.

---

# Kontekst pracy dyplomowej i spójność implementacji

- Numer dyplomu: —
- Imię i nazwisko: Marcin Połajdowicz
- Stopień/Tytuł: mgr inż.
- Kierunek: Informatyka (WETI), II stopnia, stacjonarne, 2023/2024 (sem. 3)
- Temat: „Modelowanie optymalnych sposobów zakupu licencji oprogramowania w sieciach społecznościowych za pomocą dominowania w grafach”
- English: “Modeling optimal ways to purchase software licenses in social networks via graph domination”
- Język pracy: polski
- Promotor: dr inż. Joanna Raczek

Cele badawcze (mapowane na repo):

- Pokazanie równoważności z dominacją rzymską → konfiguracja `roman_domination` oraz sweep cen `roman_p_*` w benchmarkach.
- Analiza metod: ILP (dokładny), heurystyki (Greedy, DominatingSet, Randomized), metaheurystyki (GA/SA/Tabu/ACO) → CLI `benchmark`/`benchmark_real` i analiza w `scripts/analysis`.
- Wersje cen/licencji i ograniczeń → `glopt/license_config.py` + sweep `p` + metryki (mix licencji, cost_per_node).
- Wersja dynamiczna → `glopt/cli/dynamic(.py|_real.py)` (warm vs cold), metryki delta kosztu i czasu.

Uruchamianie eksperymentów (skrót):

- `make install` – instalacja zależności (uv)
- `make benchmark` – syntetyczne grafy + cache + timeout 60 s
- `make benchmark_real` – ego-sieci Facebooka (folder `data/facebook`)
- `make dynamic` / `make dynamic_real` – mini-benchmark dynamiczny warm vs cold
- `make analyze` – generacja wykresów i tabel w `runs/<run_id>/analysis`

```

### Makefile

```
.PHONY: install test lint format clean \
	benchmark benchmark_real dynamic dynamic_real analyze \
	all

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------
PYTHON := python3
PYPATH := PYTHONPATH=src

# ------------------------------------------------------------
# Setup & QA
# ------------------------------------------------------------
install:
	uv sync

test:
	$(PYPATH) uv run --with pytest pytest -q -vv

lint:
	uv run --with ruff ruff check --fix src scripts

format:
	uv run --with black black --line-length 160 src scripts

clean:
	rm -rf .pytest_cache .ruff_cache .venv __pycache__

# ------------------------------------------------------------
# Benchmarks
# ------------------------------------------------------------
benchmark:
	$(PYPATH) uv run -m glopt.cli.benchmark

benchmark_real:
	$(PYPATH) uv run -m glopt.cli.benchmark_real

dynamic:
	$(PYPATH) uv run -m glopt.cli.dynamic

dynamic_real:
	$(PYPATH) uv run -m glopt.cli.dynamic_real

# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------
analyze:
	$(PYTHON) scripts/analysis/main.py

# Convenience
all: install lint test benchmark analyze

```

### README.md

```markdown
# Graph Licensing Optimization (GLOPT)

Optimize the cost of software licenses in social networks. Model users as a graph and decide who buys an individual plan vs. a group plan to minimize total cost. Includes exact methods (ILP), heuristics, metaheuristics (GA/SA/Tabu/ACO), a static benchmark (synthetic and real graphs), and a dynamic benchmark (warm vs. cold start).

## Requirements
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for environment management (used by Makefile)

## Install
```
make install
```

## Run benchmarks
- Static benchmark on synthetic graphs (uses on-disk cache, 60s timeout per run):
```
make benchmark
```
- Static benchmark on real graphs (Facebook ego networks in `data/facebook`):
```
make benchmark_real
```
- Dynamic benchmark (synthetic graphs): warm vs. cold start across mutation steps:
```
make dynamic
```
- Dynamic benchmark on real graphs:
```
make dynamic_real
```

Outputs are stored under `runs/<run_id>/*` (CSV under `runs/<run_id>/csv`).

## Analyze results
Modular analysis that generates plots and tables under `runs/<run_id>/analysis/`:
```
make analyze
```
Set `ANALYZE_PDF=1` to also output PDFs.

## Common tasks
```
make test      # run unit tests
make lint      # ruff lint + fix
make format    # black formatting
```

## Project layout
- `src/glopt/` — core models, validators, algorithms, I/O, and CLI entrypoints
- `scripts/analysis/` — modular analysis (main entry: `scripts/analysis/main.py`)
- `data/` — graph caches and real datasets (e.g., `data/facebook` ego-networks)
- `runs/` — outputs (CSV, plots, aggregates)

## Notes
- Benchmarks enforce a hard 60s cap per algorithm run by killing the subprocess. Larger sizes for a (graph, algorithm) pair are skipped after the first timeout.
- License sweeps: use special configs like `roman_p_2_5` (group cost = 2.5) to evaluate price sensitivity.

```

### docs/algorithms/ant_colony.md

```markdown
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

```

### docs/algorithms/dominating_set.md

```markdown
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
- Czas: ~O(V·(Δ log Δ + |L|·Δ)) — oceny pokrycia i wybór grup według stopnia.
- Pamięć: O(V).

## Wyniki z ostatniego custom.py
- Graph: small_world(n=100), License: spotify
- cost: 872.75, time_ms: 1.28, valid: True

## Uwagi
- Heurystyka “coverage per cost” kieruje wyborem dominatorów; dobór członków grup preferuje węzły o największym stopniu.

```

### docs/algorithms/genetic.md

```markdown
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

```

### docs/algorithms/greedy.md

```markdown
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

```

### docs/algorithms/ilp.md

```markdown
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

```

### docs/algorithms/naive.md

```markdown
# Naive Algorithm

- Idea: pełne przeszukiwanie (podziały na grupy i właścicieli) – dokładny, ale wykładniczy.

## Pseudocode
```
for each partition of V into blocks:
  for each choice of owner+license per block:
    if all groups feasible and cover V:
      keep best by cost
return best
```

## Złożoność
- Czas: wykładnicza w liczbie wierzchołków (brak skalowalności).
- Pamięć: w zależności od sposobu generowania – co najmniej O(V).

## Wyniki z ostatniego custom.py
- Brak w obecnym biegu custom (wyłączony domyślnie ze względu na koszty obliczeń).

## Uwagi
- Przydatny tylko dla bardzo małych grafów (≤ 10 węzłów).

```

### docs/algorithms/randomized.md

```markdown
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

```

### docs/algorithms/simulated_annealing.md

```markdown
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

```

### docs/algorithms/tabu_search.md

```markdown
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

```

### docs/algorithms/tree_dp.md

```markdown
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
- Czas: O(V · f(deg)) – liniowe względem V dla stałego stopnia.
- Pamięć: O(V).

## Wyniki z ostatniego custom.py
- Brak (algorytm dotyczy tylko drzew; nie użyty w obecnym biegu).

## Uwagi
- Gwarantuje optimum na drzewach; do grafów ogólnych wymaga innej dekompozycji (np. drzew rozpinających + heurystyki).

```

### docs/thesis_readiness_report.md

```markdown
# Ocena gotowości repozytorium do pracy magisterskiej

Temat: „Modelowanie optymalnych sposobów zakupu licencji oprogramowania w sieciach społecznościowych za pomocą dominowania w grafach”  (ang. „Modeling optimal ways to purchase software licenses in social networks via graph domination”)

Autor: Marcin Połajdowicz  •  Promotor: dr inż. Joanna Raczek


## Streszczenie
Repozytorium implementuje kompletny aparat obliczeniowy do badania problemu: model formalny, zestaw algorytmów (dokładne i metaheurystyki), rozbudowany benchmark (statyczny) oraz mini‑benchmark dynamiczny z porównaniem warm‑start vs cold‑start. Obecne skrypty generują wyniki CSV oraz wspierają analizę (skrypty wizualizacji). Część teoretyczna (dowód równoważności z dominacją rzymską oraz opis wariantów) wymaga opracowania w pracy, ale kod umożliwia przeprowadzenie potrzebnych eksperymentów.

Najważniejsze braki do rozważenia: dołączenie grafów rzeczywistych (Facebook ego‑networks) do pętli benchmarków CLI, przegląd parametryczny cen (np. sweep stosunku p=c₂/c₁), oraz jasne osadzenie konfiguracji „roman_domination” jako instancji dominacji rzymskiej w tekście pracy.


## Mapowanie na wymagania pracy (zadania)

- Opisanie grafowego modelu sieci społecznościowej
  - Stan: Zaimplementowane. Rdzeń w `src/glopt/core` (modele, walidator), generatory grafów syntetycznych w `glopt.io.graph_generator` (Erdős–Rényi, Barabási–Albert, Watts–Strogatz, oraz proste grafy kontrolne).
  - Dodatkowo: walidator rozwiązań sprawdza pojemności, sąsiedztwo właścicieli i pełne pokrycie.

- Opisanie możliwości zakupu podstawowych licencji typu Duolingo Super
  - Stan: Zaimplementowane jako konfiguracje licencji w `glopt.license_config.LicenseConfigFactory` (np. `spotify`, `duolingo_super`) – każda licencja ma koszt, min/max pojemność i kolor do wizualizacji.
  - Uwagi: łatwo dopisać dodatkowe warianty (np. zmiana stosunku cen 2×/3×/… lub innych limitów L) i objąć je benchmarkiem.

- Model rozwiązania problemu za pomocą dominowania rzymskiego w grafach
  - Stan: Wspierane przez konfigurację `roman_domination` (koszt 1 dla „solo”, koszt 2 dla „group”, pojemność „2..∞”), co praktycznie odwzorowuje etykietowanie {0,1,2}, gdzie węzeł „2” dominuje sąsiadów. Walidator wymusza sąsiedztwo członków z właścicielem.
  - Do wykonania w pracy: formalny opis i dowód równoważności (tekst), wskazanie różnic notacyjnych (grupy kontra etykietowanie) i uzasadnienie, że model grupowy z L=∞ zachowuje własności dominacji rzymskiej.

- Analiza metod i narzędzi algorytmicznych/matematycznych
  - Stan: Zaimplementowane algorytmy: dokładne (ILP, Naive, Tree DP), heurystyki (Greedy, DominatingSet, Randomized), metaheurystyki (GA, SA, Tabu, ACO). ILP oparte o PuLP/CBC. Każdy algorytm zwraca pełne rozwiązanie, mierzone są koszty i czasy.
  - Dodatki: GA posiada teraz crossover i warm‑start; SA/Tabu/ACO wspierają warm‑start. Istnieje walidacja rozwiązań oraz metryki rozmiarów grup i miks licencji.

- Definicja problemu i analiza innych wersji (ceny i modele dominowania)
  - Stan: obsługa wielu konfiguracji licencji, łatwo dopisać nowe scenariusze cenowe (np. „sześć osób = 3× koszt single”). Benchmark pętluje po nazwach konfiguracji.
  - Rekomendacja: dodać przegląd parametryczny p=c₂/c₁ (np. siatka wartości) i uwzględnić w benchmarku, aby pokazać wpływ cen na strukturę rozwiązań.

- Analiza dynamicznej wersji problemu
  - Stan: Zaimplementowana w `glopt.dynamic_simulator` i `glopt.cli.dynamic`. Mini‑benchmark generuje sekwencję mutacji (dodawanie/usuwanie węzłów i krawędzi), porównuje algorytmy warm‑start vs cold‑start (oraz baseline’y), zapisuje czasy, koszty, delta_cost i średni czas kroku.
  - Uwagi: parametry mutacji są skalowane do rozmiaru grafu i łatwo edytowalne na górze pliku `dynamic.py`.


## Eksperymenty i narzędzia

- Benchmark statyczny (`glopt.cli.benchmark`)
  - Pętla po: typach grafu × rozmiarach (gęsty zakres dla małych n, rzadszy dla dużych) × konfiguracjach licencji × algorytmach.
  - Twardy timeout 60 s per bieg (zabijanie procesu), wczesne zatrzymanie skali n po pierwszym timeout dla danej pary (graph, algorithm).
  - Cache grafów na dysku (`data/graphs_cache`) – rozgrzewany automatycznie, bieg pracuje na gotowych instancjach (szybko i deterministycznie).
  - CSV z obszernym zestawem metryk (czas, koszt, walidacja, zagęszczenie, średni stopień, klasteryzacja, liczba komponentów, rozkład typów licencji itp.).

- Benchmark dynamiczny (`glopt.cli.dynamic`)
  - Wspólna sekwencja mutacji dla porównywanych algorytmów, warm vs cold dla metaheurystyk i baseline’y (Greedy/ILP) – logowane krok po kroku, CSV z delta_cost i średnim czasem.

- Analiza wyników (`scripts/analyze.py`)
  - Rysunki kosztu vs n, czasy (log‑scale) z pasmami ufności, Pareto (czas vs koszt), zależność od gęstości, profile wydajności (Dolan–Moré), miks licencji, tabele agregatów (średnie/odchylenia/CI95).

- Dane rzeczywiste
  - Loader Facebook ego‑networks w `glopt.io.data_loader.RealWorldDataLoader` (folder `data/facebook`). Zwraca grafy i metadane (cechy, kręgi).
  - Rekomendacja: dołączyć te grafy do pętli benchmarków (np. tryb „real”) – prosty krok integracyjny.


## Co jeszcze warto dodać/przygotować (lista kontrolna)

1. Integracja grafów rzeczywistych do CLI
   - Dodać tryb benchmarku dla `RealWorldDataLoader` – pętla po ego‑sieciach z `data/facebook` i tych samych algorytmach/licencjach; wynik do osobnego CSV.

2. Sweep cen grupowych (p)
   - Wygenerować serię konfiguracji licencyjnych (np. `p ∈ {1.5, 2.0, 2.5, 3.0}`) i przebiec benchmark; w analizie pokazać wpływ p na koszt i strukturę grup.

3. Część teoretyczna (tekst pracy)
   - Formalny dowód równoważności z dominacją rzymską (opis odwzorowania rozwiązań grupowych na etykiety 0/1/2 i odwrotnie; analiza funkcji kosztu; rola L=∞). 
   - Szkice redukcji do klasycznych problemów dominacji; przegląd złożoności obliczeniowej (NP‑trudność) i konsekwencji dla metod.

4. Metryki dodatkowe
   - Np. koszt per węzeł, udział pokrycia przez różne typy licencji, korelacja kosztu z gęstością/klasteryzacją/średnim stopniem; w dynamicznych – stabilność kosztu (wariancja delta_cost).

5. Replikacja / instrukcje uruchomienia
   - Zwięzła sekcja w pracy: `make install`, `make benchmark`, `make dynamic`, `python3 scripts/analyze.py`. Wskazanie ścieżek z wynikami (`runs/<run_id>/csv`, `runs/<run_id>/analysis`).


## Wnioski
- Część implementacyjna i eksperymentalna: gotowa do przeprowadzenia szerokich testów (statycznych i dynamicznych). Repo zawiera pełny zestaw algorytmów (dokładne + metaheurystyki), walidator rozwiązań, mierniki i benchmarki, a także skrypty do analizy.
- Część teoretyczna (równoważność z dominacją rzymską, przegląd wariantów cenowych) wymaga opracowania w dokumencie pracy, ale kod (konfiguracja `roman_domination`) wspiera bezpośrednie eksperymenty w tym modelu.
- Zalecane uzupełnienia: integracja grafów rzeczywistych do benchmarków, przegląd parametryczny cen oraz sekcja „how‑to reproduce” w pracy.


## Załącznik: Szybki przewodnik uruchomieniowy

- Instalacja: `make install`
- Testy jednostkowe: `make test`
- Benchmark statyczny (syntetyczne): `make benchmark`
- Benchmark dynamiczny (warm vs cold): `make dynamic`
- Analiza wyników i wykresy: `python3 scripts/analyze.py`

Wyniki znajdują się w `runs/<run_id>/csv`, a wykresy i agregaty w `runs/<run_id>/analysis`.


```

### pyproject.toml

[project]
name = "glopt"
version = "1.0.2"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "matplotlib>=3.10.6",
    "networkx>=3.5",
    "pulp>=3.2.2",
]


### scripts/analysis/__init__.py

```python
"""Modular analysis package for glopt runs.

This package provides reusable utilities and plotting/report modules.
Use scripts/analysis/main.py as the entry point.
"""


```

### scripts/analysis/commons.py

```python
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

GENERATE_PDF = os.getenv("ANALYZE_PDF", "0") not in {"0", "false", "False", ""}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_rows(csv_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def group_cost_by_n(rows: list[dict[str, Any]], algokey: str = "algorithm") -> dict[str, list[tuple[int, float]]]:
    data: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        try:
            alg = str(r[algokey])
            n = int(float(r.get("n_nodes", 0)))
            cost = float(r.get("total_cost", 0.0))
        except Exception:
            continue
        data[alg].append((n, cost))
    for alg, pts in data.items():
        pts.sort(key=lambda x: x[0])
    return data

```

### scripts/analysis/dynamic_warmcold.py

```python
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .commons import ensure_dir, GENERATE_PDF


def plot_dynamic_warm_cold(rows: list[dict[str, Any]], title_prefix: str, out_dir: Path) -> None:
    ensure_dir(out_dir)
    # Group by algorithm and warm_start flag
    by_alg: dict[str, dict[bool, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        try:
            alg = str(r['algorithm'])
            warm = str(r.get('warm_start','False')) in {'True','true','1'}
            by_alg[alg][warm].append(r)
        except Exception:  # robust parsing
            continue

    for alg, modes in by_alg.items():
        plt.figure(figsize=(8, 5))
        for warm in (True, False):
            seq = sorted(modes.get(warm, []), key=lambda x: int(float(x.get('step',0))))
            xs = [int(float(r.get('step', 0))) for r in seq]
            costs = [float(r.get('total_cost', 0.0)) for r in seq]
            if xs:
                plt.plot(xs, costs, marker='o', label=f"{'warm' if warm else 'cold'}")
        plt.xlabel('step'); plt.ylabel('total_cost'); plt.title(f"{title_prefix} — {alg} cost per step")
        plt.legend()
        out = out_dir / f"{alg}_cost_per_step"
        plt.tight_layout(); plt.savefig(out.with_suffix('.png'), dpi=220)
        if GENERATE_PDF:
            plt.savefig(out.with_suffix('.pdf'))
        plt.close()

        plt.figure(figsize=(8, 5))
        for warm in (True, False):
            seq = sorted(modes.get(warm, []), key=lambda x: int(float(x.get('step',0))))
            xs = [int(float(r.get('step', 0))) for r in seq]
            times = [float(r.get('time_ms', 0.0)) for r in seq]
            if xs:
                plt.plot(xs, times, marker='o', label=f"{'warm' if warm else 'cold'}")
        plt.xlabel('step'); plt.ylabel('time_ms'); plt.title(f"{title_prefix} — {alg} time per step")
        plt.legend()
        out = out_dir / f"{alg}_time_per_step"
        plt.tight_layout(); plt.savefig(out.with_suffix('.png'), dpi=220)
        if GENERATE_PDF:
            plt.savefig(out.with_suffix('.pdf'))
        plt.close()

```

### scripts/analysis/main.py

```python
from __future__ import annotations

from pathlib import Path

from .commons import load_rows, ensure_dir
from .plots_cost_time import plot_cost_vs_n, plot_time_vs_n
from .plots_pareto import plot_pareto
from .plots_profiles import plot_performance_profiles
from .plots_density import plot_density_vs_time
from .plots_heatmap import plot_cost_heatmap
from .plots_license_mix import plot_license_mix
from .tables_aggregates import write_aggregates
from .dynamic_warmcold import plot_dynamic_warm_cold


def analyze_benchmark(csv_path: Path, rows: list[dict[str, object]]) -> None:
    run_dir = csv_path.parent.parent
    title = csv_path.stem
    # per (license, graph)
    combos = set()
    for r in rows:
        lic = str(r.get('license_config', ''))
        gname = str(r.get('graph', ''))
        if lic and gname:
            combos.add((lic, gname))
    for lic, gname in sorted(combos):
        sub = [r for r in rows if str(r.get('license_config','')) == lic and str(r.get('graph','')) == gname]
        out_dir = run_dir / 'analysis' / lic / gname
        ensure_dir(out_dir)
        plot_cost_vs_n(sub, title=f"{title} — {lic} — {gname}", out_path=out_dir / f"{gname}_cost_vs_n")
        plot_time_vs_n(sub, title=f"{title} — {lic} — {gname} time vs n", out_path=out_dir / f"{gname}_time_vs_n")
        plot_pareto(sub, title=f"{title} — {lic} — {gname} Pareto", out_path=out_dir / f"{gname}_pareto_cost_time")
        plot_density_vs_time(sub, title=f"{title} — {lic} — {gname} density vs time", out_path=out_dir / f"{gname}_density_vs_time")
        plot_performance_profiles(sub, title_prefix=f"{title} — {lic} — {gname}", out_dir=out_dir)
    # overall heatmap and license mix
    out_dir_all = run_dir / 'analysis' / 'all'
    ensure_dir(out_dir_all)
    plot_cost_heatmap(rows, title=f"{title} — cost heatmap", out_path=out_dir_all / "heatmap_cost")
    plot_license_mix(rows, title=f"{title} — license mix by algorithm", out_path=out_dir_all / "license_mix")
    write_aggregates(rows, out_path=out_dir_all / "aggregates.csv")


def analyze_dynamic(csv_path: Path, rows: list[dict[str, object]]) -> None:
    run_dir = csv_path.parent.parent
    title = csv_path.stem
    # split by (graph, license)
    combos = set()
    for r in rows:
        g = str(r.get('graph','')) ; lic = str(r.get('license_config',''))
        combos.add((g, lic))
    for g, lic in sorted(combos):
        sub = [r for r in rows if str(r.get('graph','')) == g and str(r.get('license_config','')) == lic]
        out_dir = run_dir / 'analysis' / lic / g
        plot_dynamic_warm_cold(sub, title_prefix=f"{title} — {lic} — {g}", out_dir=out_dir)


def main() -> None:
    runs_dir = Path("runs")
    csvs = sorted(runs_dir.glob("*/csv/*.csv"))
    if not csvs:
        print("no CSVs found under runs/*/csv")
        return
    for csv_path in csvs:
        rows = load_rows(csv_path)
        run_dir = csv_path.parent.parent
        name = run_dir.name
        print(f"analyzing {csv_path}")
        if name.endswith('_benchmark') or csv_path.stem.endswith('_benchmark') or name.endswith('_benchmark_real'):
            analyze_benchmark(csv_path, rows)
        elif name.endswith('_dynamic') or name.endswith('_dynamic_real'):
            analyze_dynamic(csv_path, rows)
        else:
            # default: try static analyses
            analyze_benchmark(csv_path, rows)


if __name__ == "__main__":
    main()


```

### scripts/analysis/plots_cost_time.py

```python
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev, pstdev
from typing import Any

import matplotlib.pyplot as plt

from .commons import ensure_dir, GENERATE_PDF, group_cost_by_n


def plot_cost_vs_n(rows: list[dict[str, Any]], title: str, out_path: Path) -> None:
    series = group_cost_by_n(rows)
    plt.figure(figsize=(8, 5))
    for alg, pts in series.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if xs and ys:
            plt.plot(xs, ys, marker="o", label=alg)
    plt.xlabel("n_nodes")
    plt.ylabel("total_cost")
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=220)
    if GENERATE_PDF:
        plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def plot_time_vs_n(rows: list[dict[str, Any]], title: str, out_path: Path) -> None:
    series_t: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        try:
            alg = str(r["algorithm"])
            n = int(float(r.get("n_nodes", 0)))
            t = float(r.get("time_ms", 0.0)) + 1e-9
        except Exception:  # robust parsing
            continue
        series_t[alg][n].append(t)
    plt.figure(figsize=(8, 5))
    for alg, dn in series_t.items():
        xs = sorted(dn.keys())
        means = [mean(dn[n]) for n in xs]
        cis = [1.96 * ((pstdev(dn[n]) if len(dn[n]) <= 1 else stdev(dn[n])) / (len(dn[n]) ** 0.5)) if len(dn[n]) > 1 else 0.0 for n in xs]
        if xs:
            plt.plot(xs, means, marker="o", label=alg)
            lower = [m - ci for m, ci in zip(means, cis)]
            upper = [m + ci for m, ci in zip(means, cis)]
            plt.fill_between(xs, lower, upper, alpha=0.15)
    plt.yscale("log")
    plt.xlabel("n_nodes")
    plt.ylabel("time_ms (log)")
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=220)
    if GENERATE_PDF:
        plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()

```

### scripts/analysis/plots_density.py

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .commons import ensure_dir, GENERATE_PDF


def plot_density_vs_time(rows: list[dict[str, Any]], title: str, out_path: Path) -> None:
    plt.figure(figsize=(6.5, 5))
    colors = {}
    import itertools
    palette = itertools.cycle([f"C{k}" for k in range(10)])
    for r in rows:
        try:
            alg = str(r['algorithm']) ; d = float(r.get('density', 0.0)) ; t = float(r.get('time_ms', 0.0))
        except Exception:  # robust parsing
            continue
        if alg not in colors:
            colors[alg] = next(palette)
        plt.scatter(d, t, s=18, alpha=0.8, c=colors[alg], label=alg if alg not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.yscale('log')
    plt.xlabel('density'); plt.ylabel('time_ms (log)'); plt.title(title)
    ensure_dir(out_path.parent)
    plt.tight_layout(); plt.savefig(out_path.with_suffix('.png'), dpi=220)
    if GENERATE_PDF:
        plt.savefig(out_path.with_suffix('.pdf'))
    plt.close()

```

### scripts/analysis/plots_heatmap.py

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .commons import ensure_dir, GENERATE_PDF


def plot_cost_heatmap(rows: list[dict[str, Any]], title: str, out_path: Path) -> None:
    dens = [] ; deg = [] ; cost = []
    for r in rows:
        try:
            d = float(r.get('density', 0.0)) ; g = float(r.get('avg_degree', 0.0)) ; c = float(r.get('total_cost', 0.0))
        except Exception:  # robust parsing of possibly incomplete rows
            continue
        dens.append(d) ; deg.append(g) ; cost.append(c)
    if not dens:
        return
    dens = np.array(dens) ; deg = np.array(deg) ; cost = np.array(cost)
    dbins = np.linspace(float(dens.min()), float(dens.max()) + 1e-12, 10)
    gbins = np.linspace(float(deg.min()), float(deg.max()) + 1e-12, 10)
    H = np.full((len(dbins)-1, len(gbins)-1), np.nan)
    for i in range(len(dbins)-1):
        for j in range(len(gbins)-1):
            mask = (dens>=dbins[i]) & (dens<dbins[i+1]) & (deg>=gbins[j]) & (deg<gbins[j+1])
            if mask.any():
                H[i,j] = float(np.mean(cost[mask]))
    plt.figure(figsize=(6.5,5))
    im = plt.imshow(H, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(im, label='mean cost')
    plt.xlabel('avg_degree bins') ; plt.ylabel('density bins') ; plt.title(title)
    ensure_dir(out_path.parent)
    plt.tight_layout() ; plt.savefig(out_path.with_suffix('.png'), dpi=220)
    if GENERATE_PDF:
        plt.savefig(out_path.with_suffix('.pdf'))
    plt.close()

```

### scripts/analysis/plots_license_mix.py

```python
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .commons import ensure_dir, GENERATE_PDF


def plot_license_mix(rows: list[dict[str, Any]], title: str, out_path: Path) -> None:
    lic_agg: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        try:
            alg = str(r["algorithm"]) ; js = r.get("license_counts_json", "{}")
            counts = json.loads(js) if isinstance(js, str) else (js or {})
            for k, v in counts.items():
                lic_agg[alg][k] += int(v)
        except Exception:  # robust parsing
            continue
    if not lic_agg:
        return
    algs = sorted(lic_agg.keys())
    all_licenses = sorted({lic for c in lic_agg.values() for lic in c.keys()})
    W = len(algs)
    vals = np.zeros((len(all_licenses), W), dtype=float)
    for j, alg in enumerate(algs):
        total = sum(lic_agg[alg].values()) or 1
        for i, lic in enumerate(all_licenses):
            vals[i, j] = lic_agg[alg].get(lic, 0) / total
    plt.figure(figsize=(max(6.5, 0.7 * W), 5))
    bottom = np.zeros(W)
    for i, lic in enumerate(all_licenses):
        plt.bar(range(W), vals[i], bottom=bottom, label=lic)
        bottom += vals[i]
    plt.xticks(range(W), algs, rotation=30, ha='right')
    plt.ylim(0, 1)
    plt.ylabel('share of license types')
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    ensure_dir(out_path.parent)
    plt.tight_layout(); plt.savefig(out_path.with_suffix('.png'), dpi=220)
    if GENERATE_PDF:
        plt.savefig(out_path.with_suffix('.pdf'))
    plt.close()

```

### scripts/analysis/plots_pareto.py

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .commons import ensure_dir, GENERATE_PDF


def plot_pareto(rows: list[dict[str, Any]], title: str, out_path: Path) -> None:
    pts = []
    for r in rows:
        try:
            alg = str(r["algorithm"])
            c = float(r["total_cost"]) ; t = float(r["time_ms"]) ; d = float(r.get("density", 0.0))
        except Exception:  # robust parsing
            continue
        pts.append((t, c, alg, d))
    pareto = []
    for i, (ti, ci, *_ ) in enumerate(pts):
        dominated = False
        for j, (tj, cj, *_ ) in enumerate(pts):
            if j != i and tj <= ti and cj <= ci and (tj < ti or cj < ci):
                dominated = True
                break
        if not dominated:
            pareto.append((ti, ci))
    plt.figure(figsize=(6.5, 5))
    colors = {}
    import itertools
    palette = itertools.cycle([f"C{k}" for k in range(10)])
    for t, c, alg, _ in pts:
        if alg not in colors:
            colors[alg] = next(palette)
        plt.scatter(t, c, s=18, alpha=0.8, c=colors[alg], label=alg if alg not in plt.gca().get_legend_handles_labels()[1] else "")
    if pareto:
        xs = [p[0] for p in pareto] ; ys = [p[1] for p in pareto]
        order = sorted(range(len(pareto)), key=lambda k: xs[k])
        xs = [xs[k] for k in order] ; ys = [ys[k] for k in order]
        plt.plot(xs, ys, color="k", linewidth=2, alpha=0.6)
    plt.xlabel("time_ms") ; plt.ylabel("total_cost") ; plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    ensure_dir(out_path.parent)
    plt.tight_layout() ; plt.savefig(out_path.with_suffix(".png"), dpi=220)
    if GENERATE_PDF:
        plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()

```

### scripts/analysis/plots_profiles.py

```python
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt

from .commons import ensure_dir, GENERATE_PDF


def plot_performance_profiles(rows: list[dict[str, Any]], title_prefix: str, out_dir: Path) -> None:
    ensure_dir(out_dir)
    # Instances based on (graph, n_nodes) if present; fallback to index
    insts = sorted({(str(r.get('graph','')), int(float(r.get('n_nodes', 0)))) for r in rows if r.get('n_nodes')})
    # Cost profiles
    perf_cost: dict[str, list[float]] = defaultdict(list)
    for g, n in insts:
        per_alg = defaultdict(list)
        for r in rows:
            try:
                if str(r.get('graph','')) == g and int(float(r.get('n_nodes', 0))) == n:
                    per_alg[str(r['algorithm'])].append(float(r['total_cost']))
            except Exception:  # robust parsing
                pass
        means = {alg: (mean(vs) if vs else float('inf')) for alg, vs in per_alg.items()}
        best = min((v for v in means.values() if v > 0), default=None)
        if not best:
            continue
        for alg, v in means.items():
            if v > 0:
                perf_cost[alg].append(v / best)
    if perf_cost:
        taus = [1.0 + i * 0.05 for i in range(0, 61)]
        plt.figure(figsize=(6.5, 5))
        for alg, ratios in sorted(perf_cost.items()):
            ratios = sorted(ratios)
            ys = []
            for tau in taus:
                c = sum(1 for r in ratios if r <= tau)
                ys.append(c / len(ratios) if ratios else 0.0)
            plt.plot(taus, ys, label=alg)
        plt.xlabel('tau') ; plt.ylabel('fraction of instances')
        plt.title(f"{title_prefix} Performance profile (cost)")
        plt.legend(ncol=2, fontsize=8)
        out = out_dir / f"perf_profile_cost"
        plt.tight_layout(); plt.savefig(out.with_suffix('.png'), dpi=220)
        if GENERATE_PDF:
            plt.savefig(out.with_suffix('.pdf'))
        plt.close()

    # Time profiles
    perf_time: dict[str, list[float]] = defaultdict(list)
    for g, n in insts:
        per_alg = defaultdict(list)
        for r in rows:
            try:
                if str(r.get('graph','')) == g and int(float(r.get('n_nodes', 0))) == n:
                    per_alg[str(r['algorithm'])].append(float(r['time_ms']))
            except Exception:  # robust parsing
                pass
        means = {alg: (mean(vs) if vs else float('inf')) for alg, vs in per_alg.items()}
        best = min((v for v in means.values() if v > 0), default=None)
        if not best:
            continue
        for alg, v in means.items():
            if v > 0:
                perf_time[alg].append(v / best)
    if perf_time:
        taus = [1.0 + i * 0.1 for i in range(0, 61)]
        plt.figure(figsize=(6.5, 5))
        for alg, ratios in sorted(perf_time.items()):
            ratios = sorted(ratios)
            ys = []
            for tau in taus:
                c = sum(1 for r in ratios if r <= tau)
                ys.append(c / len(ratios) if ratios else 0.0)
            plt.plot(taus, ys, label=alg)
        plt.xlabel('tau') ; plt.ylabel('fraction of instances')
        plt.title(f"{title_prefix} Performance profile (time)")
        plt.legend(ncol=2, fontsize=8)
        out = out_dir / f"perf_profile_time"
        plt.tight_layout(); plt.savefig(out.with_suffix('.png'), dpi=220)
        if GENERATE_PDF:
            plt.savefig(out.with_suffix('.pdf'))
        plt.close()

```

### scripts/analysis/tables_aggregates.py

```python
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev, pstdev
from typing import Any

from .commons import ensure_dir


def write_aggregates(rows: list[dict[str, Any]], out_path: Path) -> None:
    groups = defaultdict(list)
    for r in rows:
        try:
            key = (str(r['algorithm']), str(r.get('graph','')), int(float(r.get('n_nodes', 0))))
            groups[key].append(r)
        except Exception:  # robust parsing
            continue
    lines = []
    for (alg, gname, n), rs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        costs = [] ; times = []
        for r in rs:
            try:
                costs.append(float(r['total_cost'])) ; times.append(float(r['time_ms']))
            except Exception:  # robust parsing
                pass
        if not costs:
            continue
        m_c = mean(costs)
        s_c = pstdev(costs) if len(costs) <= 1 else stdev(costs)
        m_t = mean(times)
        s_t = pstdev(times) if len(times) <= 1 else stdev(times)
        nrep = len(costs)
        ci_c = 1.96 * (s_c / (nrep ** 0.5)) if nrep > 1 else 0.0
        ci_t = 1.96 * (s_t / (nrep ** 0.5)) if nrep > 1 else 0.0
        lines.append({
            'algorithm': alg,
            'graph': gname,
            'n_nodes': n,
            'rep': nrep,
            'cost_mean': f"{m_c:.6f}",
            'cost_std': f"{s_c:.6f}",
            'cost_ci95': f"{ci_c:.6f}",
            'time_ms_mean': f"{m_t:.6f}",
            'time_ms_std': f"{s_t:.6f}",
            'time_ms_ci95': f"{ci_t:.6f}",
        })
    if lines:
        ensure_dir(out_path.parent)
        with out_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(lines[0].keys()))
            writer.writeheader()
            writer.writerows(lines)

```

### scripts/analyze.py

```python
#!/usr/bin/env python3
from analysis.main import main

if __name__ == "__main__":
    main()


```

### src/glopt/__init__.py

```python
from . import algorithms
from .core import (
    Algorithm,
    LicenseGroup,
    LicenseType,
    MutationOperators,
    RunResult,
    Solution,
    SolutionBuilder,
    SolutionValidator,
    generate_graph,
    instantiate_algorithms,
    run_once,
)
from .dynamic_simulator import DynamicNetworkSimulator
from .io.csv_writer import BenchmarkCSVWriter
from .io.data_loader import RealWorldDataLoader
from .io.graph_generator import GraphGeneratorFactory
from .io.graph_visualizer import GraphVisualizer
from .license_config import LicenseConfigFactory

__all__ = [
    "Algorithm",
    "BenchmarkCSVWriter",
    "DynamicNetworkSimulator",
    "GraphGeneratorFactory",
    "GraphVisualizer",
    "LicenseConfigFactory",
    "LicenseGroup",
    "LicenseType",
    "MutationOperators",
    "RealWorldDataLoader",
    "RunResult",
    "Solution",
    "SolutionBuilder",
    "SolutionValidator",
    "algorithms",
    "generate_graph",
    "instantiate_algorithms",
    "run_once",
]

```

### src/glopt/algorithms/__init__.py

```python
from .ant_colony import AntColonyOptimization
from .dominating_set import DominatingSetAlgorithm
from .genetic import GeneticAlgorithm
from .greedy import GreedyAlgorithm
from .ilp import ILPSolver
from .naive import NaiveAlgorithm
from .randomized import RandomizedAlgorithm
from .simulated_annealing import SimulatedAnnealing
from .tabu_search import TabuSearch
from .tree_dp import TreeDynamicProgramming

__all__ = [
    "AntColonyOptimization",
    "DominatingSetAlgorithm",
    "GeneticAlgorithm",
    "GreedyAlgorithm",
    "ILPSolver",
    "NaiveAlgorithm",
    "RandomizedAlgorithm",
    "SimulatedAnnealing",
    "TabuSearch",
    "TreeDynamicProgramming",
]

```

### src/glopt/algorithms/ant_colony.py

```python
from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from glopt.algorithms.greedy import GreedyAlgorithm
from glopt.core import Algorithm, LicenseGroup, LicenseType, Solution
from glopt.core.solution_validator import SolutionValidator

if TYPE_CHECKING:
    import networkx as nx

PKey = tuple[Any, str]


class AntColonyOptimization(Algorithm):
    @property
    def name(self) -> str:
        return "ant_colony_optimization"

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation: float = 0.5,
        q0: float = 0.9,
        num_ants: int = 20,
        max_iterations: int = 100,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.evap = evaporation
        self.q0 = q0
        self.num_ants = num_ants
        self.max_iter = max_iterations
        self.validator = SolutionValidator(debug=False)

    def solve(self, graph: nx.Graph, license_types: list[LicenseType], **kwargs: Any) -> Solution:
        seed = kwargs.get("seed")
        if isinstance(seed, int):
            random.seed(seed)
        deadline = kwargs.get("deadline")
        max_iter = int(kwargs.get("max_iterations", self.max_iter))
        num_ants = int(kwargs.get("num_ants", self.num_ants))
        initial: Solution | None = kwargs.get("initial_solution")

        pher = self._init_pher(graph, license_types)
        heur = self._init_heur(graph, license_types)

        # Warm start or greedy baseline
        if initial is not None and self.validator.is_valid_solution(initial, graph):
            best = initial
        else:
            best = GreedyAlgorithm().solve(graph, license_types)
        ok, _ = self.validator.validate(best, graph)
        if not ok:
            best = self._fallback_singletons(graph, license_types)
        best_cost = best.total_cost
        self._deposit(pher, best)

        from time import perf_counter as _pc
        for _ in range(max_iter):
            if deadline is not None and _pc() >= float(deadline):
                break
            improved = False
            for _ in range(num_ants):
                cand = self._construct(graph, license_types, pher, heur)
                ok, _ = self.validator.validate(cand, graph)
                if not ok:
                    continue
                if cand.total_cost < best_cost:
                    best, best_cost, improved = cand, cand.total_cost, True
            self._evaporate(pher)
            self._deposit(pher, best)
            if not improved:
                continue
        # Final safety: ensure we never return an invalid solution.
        ok, _ = self.validator.validate(best, graph)
        if not ok:
            return self._fallback_singletons(graph, license_types)
        return best

    def _construct(self, graph: nx.Graph, lts: list[LicenseType], pher: dict[PKey, float], heur: dict[PKey, float]) -> Solution:
        uncovered: set[Any] = set(graph.nodes())
        groups: list[LicenseGroup] = []
        while uncovered:
            owner = self._select_owner(uncovered, lts, pher, heur)
            owner = owner if owner is not None else next(iter(uncovered))
            lt = self._select_license(owner, lts, pher, heur) or min(lts, key=lambda x: x.cost)

            pool = (set(graph.neighbors(owner)) | {owner}) & uncovered
            if len(pool) < lt.min_capacity:
                # Try to place a feasible group anyway. Prefer singles if available; otherwise, pick
                # the cheapest compatible license for the available pool size.
                if lt.min_capacity == 1:
                    groups.append(LicenseGroup(lt, owner, frozenset()))
                    uncovered.remove(owner)
                else:
                    feas = [x for x in lts if x.min_capacity <= len(pool) <= x.max_capacity]
                    if not feas:
                        # Explicitly try singles if license set supports them.
                        singles = [x for x in lts if x.min_capacity <= 1 <= x.max_capacity]
                        if singles:
                            lt1 = min(singles, key=lambda x: x.cost)
                            groups.append(LicenseGroup(lt1, owner, frozenset()))
                            uncovered.remove(owner)
                        else:
                            # No feasible assignment for this owner with current license set; skip covering here.
                            uncovered.remove(owner)
                    else:
                        lt2 = min(feas, key=lambda x: x.cost)
                        add_need = max(0, lt2.min_capacity - 1)
                        add = sorted((pool - {owner}), key=lambda n: graph.degree(n), reverse=True)[:add_need]
                        groups.append(LicenseGroup(lt2, owner, frozenset(add)))
                        uncovered -= {owner} | set(add)
                continue

            k = max(0, lt.max_capacity - 1)
            add = sorted((pool - {owner}), key=lambda n: graph.degree(n), reverse=True)[:k]
            groups.append(LicenseGroup(lt, owner, frozenset(add)))
            uncovered -= {owner} | set(add)
        return Solution(groups=tuple(groups))

    def _select_owner(self, uncovered: set[Any], lts: list[LicenseType], pher: dict[PKey, float], heur: dict[PKey, float]) -> Any | None:
        if not uncovered:
            return None
        scores: dict[Any, float] = {}
        for n in uncovered:
            acc = 0.0
            for lt in lts:
                tau = pher.get((n, lt.name), 1.0)
                eta = heur.get((n, lt.name), 1.0)
                acc += (tau**self.alpha) * (eta**self.beta)
            scores[n] = acc / max(1, len(lts))
        return self._roulette_or_best(list(uncovered), scores)

    def _select_license(self, owner: Any, lts: list[LicenseType], pher: dict[PKey, float], heur: dict[PKey, float]) -> LicenseType | None:
        if not lts:
            return None
        scores = {lt: (pher.get((owner, lt.name), 1.0) ** self.alpha) * (heur.get((owner, lt.name), 1.0) ** self.beta) for lt in lts}
        return self._roulette_or_best(lts, scores)

    def _roulette_or_best(self, choices: list[Any], scores: dict[Any, float]) -> Any:
        if not choices:
            return None
        if random.random() < self.q0:
            return max(choices, key=lambda c: scores.get(c, 0.0))
        total = sum(max(0.0, scores.get(c, 0.0)) for c in choices)
        if total <= 0:
            return random.choice(choices)
        r = random.uniform(0, total)
        acc = 0.0
        for c in choices:
            acc += max(0.0, scores.get(c, 0.0))
            if acc >= r:
                return c
        return random.choice(choices)

    def _init_pher(self, graph: nx.Graph, lts: list[LicenseType]) -> dict[PKey, float]:
        return {(n, lt.name): 1.0 for n in graph.nodes() for lt in lts}

    def _init_heur(self, graph: nx.Graph, lts: list[LicenseType]) -> dict[PKey, float]:
        h: dict[PKey, float] = {}
        for n in graph.nodes():
            deg = graph.degree(n)
            for lt in lts:
                cap_eff = (lt.max_capacity / lt.cost) if lt.cost > 0 else 1e9
                h[n, lt.name] = cap_eff * (1.0 + deg)
        return h

    def _evaporate(self, pher: dict[PKey, float]) -> None:
        f = max(0.0, min(1.0, self.evap))
        for k in pher:
            pher[k] *= 1.0 - f

    def _deposit(self, pher: dict[PKey, float], sol: Solution) -> None:
        if sol.total_cost <= 0:
            return
        q = 1.0 / sol.total_cost
        for g in sol.groups:
            for n in g.all_members:
                k = (n, g.license_type.name)
                if k in pher:
                    pher[k] += q

    def _fallback_singletons(self, graph: nx.Graph, lts: list[LicenseType]) -> Solution:
        lt1 = min([x for x in lts if x.min_capacity <= 1] or lts, key=lambda x: x.cost)
        groups = [LicenseGroup(lt1, n, frozenset()) for n in graph.nodes()]
        return Solution(groups=tuple(groups))

```

### src/glopt/algorithms/dominating_set.py

```python
from typing import Any

import networkx as nx

from glopt.core import Algorithm, LicenseGroup, LicenseType, Solution
from glopt.core.solution_builder import SolutionBuilder


class DominatingSetAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "dominating_set_algorithm"

    def solve(self, graph: nx.Graph, license_types: list[LicenseType], **kwargs: Any) -> Solution:
        if len(graph.nodes()) == 0:
            return Solution(groups=())

        dominating_set = self._find_cost_effective_dominating_set(graph, license_types)

        remaining_nodes = set(graph.nodes())
        groups = []

        sorted_dominators = sorted(dominating_set, key=lambda n: graph.degree(n), reverse=True)

        for dominator in sorted_dominators:
            if dominator not in remaining_nodes:
                continue

            neighbors = set(graph.neighbors(dominator)) & remaining_nodes
            available_nodes = neighbors | {dominator}

            best_assignment = self._find_best_cost_assignment(dominator, available_nodes, license_types)

            if best_assignment:
                license_type, group_members = best_assignment
                additional_members = group_members - {dominator}
                group = LicenseGroup(license_type, dominator, frozenset(additional_members))
                groups.append(group)
                remaining_nodes -= group_members

        remaining_sorted = sorted(remaining_nodes, key=lambda n: graph.degree(n), reverse=True)

        for node in remaining_sorted:
            if node not in remaining_nodes:
                continue

            neighbors = set(graph.neighbors(node)) & remaining_nodes
            available_nodes = neighbors | {node}

            best_assignment = self._find_best_cost_assignment(node, available_nodes, license_types)

            if best_assignment:
                license_type, group_members = best_assignment
                additional_members = group_members - {node}
                group = LicenseGroup(license_type, node, frozenset(additional_members))
                groups.append(group)
                remaining_nodes -= group_members
            else:
                cheapest_single = self._find_cheapest_single_license(license_types)
                group = LicenseGroup(cheapest_single, node, frozenset())
                groups.append(group)
                remaining_nodes.remove(node)

        return SolutionBuilder.create_solution_from_groups(groups)

    def _find_cost_effective_dominating_set(self, graph: nx.Graph, license_types: list[LicenseType]) -> set[Any]:
        nodes = set(graph.nodes())
        uncovered = nodes.copy()
        dominating_set = set()

        while uncovered:
            best_node = None
            best_score = -1

            for node in nodes:
                if node in dominating_set:
                    continue

                neighbors = set(graph.neighbors(node))
                coverage = (neighbors | {node}) & uncovered

                if len(coverage) == 0:
                    continue

                min_cost_per_node = self._calculate_min_cost_per_node(len(coverage), license_types)

                score = len(coverage) / min_cost_per_node if min_cost_per_node > 0 else len(coverage)

                if score > best_score:
                    best_score = score
                    best_node = node

            if best_node is None:
                best_node = next(iter(uncovered))

            dominating_set.add(best_node)
            neighbors = set(graph.neighbors(best_node))
            covered_by_node = (neighbors | {best_node}) & uncovered
            uncovered -= covered_by_node

        return dominating_set

    def _calculate_min_cost_per_node(self, group_size: int, license_types: list[LicenseType]) -> float:
        min_cost = float("inf")

        for license_type in license_types:
            if license_type.min_capacity <= group_size <= license_type.max_capacity:
                cost_per_node = license_type.cost / group_size
                min_cost = min(min_cost, cost_per_node)

        return min_cost if min_cost != float("inf") else 0

    def _find_best_cost_assignment(self, owner: Any, available_nodes: set[Any], license_types: list[LicenseType]) -> tuple[LicenseType, set[Any]]:
        best_assignment = None
        best_efficiency = float("inf")

        for license_type in license_types:
            max_possible_size = min(len(available_nodes), license_type.max_capacity)

            for group_size in range(license_type.min_capacity, max_possible_size + 1):
                if group_size > len(available_nodes):
                    break

                group_members = self._select_best_group_members(owner, available_nodes, group_size)

                if len(group_members) == group_size:
                    cost_per_node = license_type.cost / group_size

                    if cost_per_node < best_efficiency:
                        best_efficiency = cost_per_node
                        best_assignment = (license_type, group_members)

        return best_assignment

    def _select_best_group_members(self, owner: Any, available_nodes: set[Any], target_size: int) -> set[Any]:
        if target_size <= 0:
            return set()

        group_members = {owner}
        remaining_slots = target_size - 1

        if remaining_slots <= 0:
            return group_members

        candidates = list(available_nodes - {owner})

        candidates.sort(key=lambda _: len(available_nodes), reverse=True)

        group_members.update(candidates[:remaining_slots])

        return group_members

    def _find_cheapest_single_license(self, license_types: list[LicenseType]) -> LicenseType:
        single_licenses = [lt for lt in license_types if lt.min_capacity <= 1 <= lt.max_capacity]

        if not single_licenses:
            return min(license_types, key=lambda lt: lt.cost)

        return min(single_licenses, key=lambda lt: lt.cost)

```

### src/glopt/algorithms/genetic.py

```python
from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from glopt.algorithms.randomized import RandomizedAlgorithm
from glopt.algorithms.greedy import GreedyAlgorithm
from glopt.core import Algorithm, LicenseType, Solution
from glopt.core.solution_builder import SolutionBuilder
from glopt.core.mutations import MutationOperators
from glopt.core.solution_validator import SolutionValidator

if TYPE_CHECKING:
    from collections.abc import Sequence

    import networkx as nx


class GeneticAlgorithm(Algorithm):
    def __init__(
        self,
        population_size: int = 30,
        generations: int = 40,
        elite_fraction: float = 0.2,
        crossover_rate: float = 0.6,
        seed: int | None = None,
    ) -> None:
        self.population_size = max(2, population_size)
        self.generations = max(1, generations)
        self.elite_fraction = max(0.0, min(1.0, elite_fraction))
        self.crossover_rate = max(0.0, min(1.0, crossover_rate))
        self.seed = seed
        self.validator = SolutionValidator()

    @property
    def name(self) -> str:
        return "genetic"

    def solve(
        self,
        graph: nx.Graph,
        license_types: Sequence[LicenseType],
        **kwargs: Any,
    ) -> Solution:
        seed = kwargs.get("seed", self.seed)
        if isinstance(seed, int):
            random.seed(seed)
        deadline = kwargs.get("deadline")
        initial: Solution | None = kwargs.get("initial_solution")
        generations = int(kwargs.get("generations", self.generations))
        if graph.number_of_nodes() == 0:
            return Solution()

        population = self._init_population(graph, license_types, initial)
        best = min(population, key=lambda s: s.total_cost)

        from time import perf_counter as _pc
        for _ in range(generations):
            if deadline is not None and _pc() >= float(deadline):
                break
            population.sort(key=lambda s: s.total_cost)
            elite_count = max(1, int(self.elite_fraction * self.population_size))
            new_pop: list[Solution] = population[:elite_count]

            while len(new_pop) < self.population_size:
                # Crossover with probability; otherwise mutate
                if random.random() < self.crossover_rate and len(population) >= 2:
                    p1 = self._tournament_selection(population)
                    p2 = self._tournament_selection(population)
                    if p2 is p1 and len(population) > 1:
                        # ensure two (likely) different parents
                        p2 = random.choice(population)
                    child = self._crossover(p1, p2, graph, license_types)
                    if not self.validator.is_valid_solution(child, graph):
                        # fallback to mutation when crossover yields invalid
                        base = min([p1, p2], key=lambda s: s.total_cost)
                        child = self._mutate(base, graph, license_types)
                else:
                    parent = self._tournament_selection(population)
                    child = self._mutate(parent, graph, license_types)
                new_pop.append(child)

            population = new_pop
            current_best = min(population, key=lambda s: s.total_cost)
            if current_best.total_cost < best.total_cost:
                best = current_best

        return best

    def _init_population(self, graph: nx.Graph, license_types: Sequence[LicenseType], initial: Solution | None = None) -> list[Solution]:
        # Seed population with (optional) warm-start and a strong greedy baseline; fill the rest randomly
        pop: list[Solution] = []
        if initial is not None and self.validator.is_valid_solution(initial, graph):
            pop.append(initial)
        try:
            greedy = GreedyAlgorithm().solve(graph, list(license_types))
            pop.append(greedy)
        except Exception:  # defensive: fallback if greedy fails unexpectedly
            pass
        rand_algo = RandomizedAlgorithm()
        while len(pop) < self.population_size:
            pop.append(rand_algo.solve(graph, list(license_types)))
        return pop

    def _tournament_selection(self, population: list[Solution], k: int = 3) -> Solution:
        k = max(1, min(k, len(population)))
        contenders = random.sample(population, k)
        return min(contenders, key=lambda s: s.total_cost)

    def _mutate(
        self,
        solution: Solution,
        graph: nx.Graph,
        license_types: Sequence[LicenseType],
    ) -> Solution:
        neighbors = MutationOperators.generate_neighbors(solution, graph, license_types, k=5)
        valid_neighbors = [s for s in neighbors if self.validator.is_valid_solution(s, graph)]
        if not valid_neighbors:
            # Fallback: try greedy improvement
            try:
                greedy = GreedyAlgorithm().solve(graph, list(license_types))
                if self.validator.is_valid_solution(greedy, graph) and greedy.total_cost <= solution.total_cost:
                    return greedy
            except Exception:  # defensive: ignore greedy failure in fallback
                pass
            return solution
        return min(valid_neighbors, key=lambda s: s.total_cost)

    def _crossover(
        self,
        p1: Solution,
        p2: Solution,
        graph: nx.Graph,
        license_types: Sequence[LicenseType],
    ) -> Solution:
        # Greedy merge of parent groups by efficiency, then fill uncovered with greedy on subgraph
        def eff(g):
            return (g.license_type.cost / max(1, g.size), -g.size)

        candidates = list(p1.groups) + list(p2.groups)
        candidates.sort(key=eff)

        used = set()
        chosen: list = []
        for g in candidates:
            if used.isdisjoint(g.all_members):
                chosen.append(g)
                used.update(g.all_members)

        uncovered = set(graph.nodes()) - used
        if uncovered:
            H = graph.subgraph(uncovered)
            filler = GreedyAlgorithm().solve(H, list(license_types))
            # Accept only members entirely from 'uncovered' and add to chosen
            for fg in filler.groups:
                if set(fg.all_members).issubset(uncovered):
                    chosen.append(fg)
            # No need to update 'used' further; final validation will check coverage

        child = SolutionBuilder.create_solution_from_groups(chosen)
        # Ensure validity; if not valid, fall back to greedy global
        if not self.validator.is_valid_solution(child, graph):
            return GreedyAlgorithm().solve(graph, list(license_types))
        return child

```

### src/glopt/algorithms/greedy.py

```python
from typing import Any

import networkx as nx

from glopt.core import Algorithm, LicenseGroup, LicenseType, Solution
from glopt.core.solution_builder import SolutionBuilder


class GreedyAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "greedy"

    def solve(
        self,
        graph: nx.Graph,
        license_types: list[LicenseType],
        **_: Any,
    ) -> Solution:
        licenses = sorted(license_types, key=lambda lt: (-lt.max_capacity, lt.cost))

        nodes: list[Any] = list(graph.nodes())
        uncovered: set[Any] = set(nodes)
        groups: list[LicenseGroup] = []

        for owner in sorted(nodes, key=lambda n: graph.degree(n), reverse=True):
            if owner not in uncovered:
                continue

            avail = SolutionBuilder.get_owner_neighbors_with_self(graph, owner) & uncovered
            if not avail:
                continue

            best_group = self._best_group_for_owner(owner, avail, graph, licenses)
            if best_group is None:
                continue

            groups.append(best_group)
            uncovered -= best_group.all_members

        while uncovered:
            owner = next(iter(uncovered))
            avail = SolutionBuilder.get_owner_neighbors_with_self(graph, owner) & uncovered

            fallback = self._cheapest_feasible_group(owner, avail, graph, license_types)
            if fallback is not None:
                groups.append(fallback)
                uncovered -= fallback.all_members
                continue

            cheapest = min(license_types, key=lambda lt: lt.cost)
            if cheapest.min_capacity == 1:
                groups.append(LicenseGroup(license_type=cheapest, owner=owner, additional_members=frozenset()))
                uncovered.remove(owner)
            else:
                break

        return SolutionBuilder.create_solution_from_groups(groups)

    def _best_group_for_owner(
        self,
        owner: Any,
        avail: set[Any],
        graph: nx.Graph,
        licenses: list[LicenseType],
    ) -> LicenseGroup | None:
        ordered = sorted(avail, key=lambda n: graph.degree(n), reverse=True)

        best: LicenseGroup | None = None
        best_eff = float("inf")

        for lt in licenses:
            cap_additional = max(0, lt.max_capacity - 1)
            pool = [n for n in ordered if n != owner]
            take = min(len(pool), cap_additional)

            additional = pool[:take]
            size_with_owner = 1 + len(additional)
            if size_with_owner < lt.min_capacity:
                continue

            grp = LicenseGroup(license_type=lt, owner=owner, additional_members=frozenset(additional))
            eff = lt.cost / grp.size
            if eff < best_eff:
                best_eff = eff
                best = grp

        return best

    def _cheapest_feasible_group(
        self,
        owner: Any,
        avail: set[Any],
        graph: nx.Graph,
        license_types: list[LicenseType],
    ) -> LicenseGroup | None:
        for lt in sorted(license_types, key=lambda x: (x.cost, -x.max_capacity)):
            if len(avail) < lt.min_capacity:
                continue

            need_additional = max(0, lt.min_capacity - 1)
            pool = sorted((n for n in avail if n != owner), key=lambda n: graph.degree(n), reverse=True)
            chosen = pool[:need_additional]

            return LicenseGroup(license_type=lt, owner=owner, additional_members=frozenset(chosen))

        return None

```

### src/glopt/algorithms/ilp.py

```python
from typing import Any

import networkx as nx
import pulp

from glopt.core import Algorithm, LicenseGroup, LicenseType, Solution

VAR_TRUE_THRESHOLD = 0.5


class ILPSolver(Algorithm):
    @property
    def name(self) -> str:
        return "ilp"

    def solve(
        self,
        graph: nx.Graph,
        license_types: list[LicenseType],
        **kwargs: Any,
    ) -> Solution:
        time_limit: int | None = kwargs.get("time_limit")

        nodes: list[Any] = list(graph.nodes())
        model = pulp.LpProblem("graph_licensing_optimization", pulp.LpMinimize)

        assign_vars: dict[tuple[Any, Any, int], pulp.LpVariable] = {}
        for i in nodes:
            neighborhood_i: set[Any] = set(graph.neighbors(i)) | {i}
            for j in neighborhood_i:
                for t_idx, _lt in enumerate(license_types):
                    assign_vars[i, j, t_idx] = pulp.LpVariable(f"x_{i}_{j}_{t_idx}", cat="Binary")

        active_vars: dict[tuple[Any, int], pulp.LpVariable] = {}
        for i in nodes:
            for t_idx, _lt in enumerate(license_types):
                active_vars[i, t_idx] = pulp.LpVariable(f"group_active_{i}_{t_idx}", cat="Binary")

        model += pulp.lpSum(active_vars[i, t_idx] * lt.cost for i in nodes for t_idx, lt in enumerate(license_types))

        # Each owner can activate at most one license type
        for i in nodes:
            model += pulp.lpSum(active_vars[i, t_idx] for t_idx in range(len(license_types))) <= 1

        for j in nodes:
            neighborhood_j: set[Any] = set(graph.neighbors(j)) | {j}
            model += pulp.lpSum(assign_vars.get((i, j, t_idx), 0) for i in neighborhood_j for t_idx in range(len(license_types))) == 1

        for i in nodes:
            neighborhood_i = set(graph.neighbors(i)) | {i}
            for t_idx, lt in enumerate(license_types):
                group_size = pulp.lpSum(assign_vars.get((i, j, t_idx), 0) for j in neighborhood_i)
                # Capacity bounds bind only when the group is active
                model += group_size <= active_vars[i, t_idx] * lt.max_capacity
                model += group_size >= active_vars[i, t_idx] * lt.min_capacity

        for i in nodes:
            for t_idx, _lt in enumerate(license_types):
                var = assign_vars.get((i, i, t_idx))
                if var is not None:
                    model += var >= active_vars[i, t_idx]

        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit) if time_limit else pulp.PULP_CBC_CMD(msg=0)
        model.solve(solver)

        if model.status != pulp.LpStatusOptimal:
            msg = f"ilp solver failed with status {pulp.LpStatus[model.status]}"
            raise RuntimeError(msg)

        groups: list[LicenseGroup] = []
        for i in nodes:
            for t_idx, lt in enumerate(license_types):
                if active_vars[i, t_idx].varValue and active_vars[i, t_idx].varValue > VAR_TRUE_THRESHOLD:
                    members: set[Any] = set()
                    for j in set(graph.neighbors(i)) | {i}:
                        var = assign_vars.get((i, j, t_idx))
                        if var and var.varValue and var.varValue > VAR_TRUE_THRESHOLD:
                            members.add(j)
                    if members:
                        groups.append(
                            LicenseGroup(
                                license_type=lt,
                                owner=i,
                                additional_members=frozenset(members - {i}),
                            ),
                        )

        return Solution(groups=tuple(groups))

```

### src/glopt/algorithms/naive.py

```python
from collections.abc import Iterator, Sequence
from itertools import product
from typing import Any

import networkx as nx

from glopt.core import Algorithm, LicenseGroup, LicenseType, Solution
from glopt.core.solution_builder import SolutionBuilder

Assignment = list[tuple[LicenseType, Any, set[Any]]]


class NaiveAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "naive_algorithm"

    def solve(
        self,
        graph: nx.Graph,
        license_types: Sequence[LicenseType],
        **kwargs: Any,
    ) -> Solution:
        nodes: list[Any] = list(graph.nodes())
        n = len(nodes)

        max_n = 10
        if n > max_n:
            msg = f"graph too large for naive algorithm: {n} nodes > {max_n}"
            raise ValueError(msg)

        if n == 0:
            return Solution(groups=())

        best: Assignment | None = None
        best_cost: float = float("inf")

        for assignment in self._generate_all_assignments(nodes, graph, license_types):
            if not assignment:
                continue
            if self._is_valid_assignment(assignment, nodes, graph):
                cost = self._calculate_cost(assignment)
                if cost < best_cost:
                    best_cost = cost
                    best = assignment

        if best is None:
            cheapest_single = min(
                license_types,
                key=lambda lt: lt.cost if lt.min_capacity <= 1 <= lt.max_capacity else float("inf"),
            )
            groups = [LicenseGroup(cheapest_single, node, frozenset()) for node in nodes]
            return SolutionBuilder.create_solution_from_groups(groups)

        return self._create_solution_from_assignment(best)

    def _generate_all_assignments(
        self,
        nodes: list[Any],
        graph: nx.Graph,
        license_types: Sequence[LicenseType],
    ) -> Iterator[Assignment]:
        for partition in self._generate_partitions(nodes):
            yield from self._generate_assignments_for_partition(partition, graph, license_types)

    def _generate_partitions(self, nodes: list[Any]) -> Iterator[list[set[Any]]]:
        n = len(nodes)
        if n == 0:
            yield []
            return
        if n == 1:
            yield [{nodes[0]}]
            return

        first, rest = nodes[0], nodes[1:]
        for smaller in self._generate_partitions(rest):
            yield [{first}, *smaller]

            for i, block in enumerate(smaller):
                new_part = list(smaller)
                new_part[i] = set(block) | {first}
                yield new_part

    def _generate_assignments_for_partition(
        self,
        partition: list[set[Any]],
        graph: nx.Graph,
        license_types: Sequence[LicenseType],
    ) -> Iterator[Assignment]:
        if not partition:
            yield []
            return

        per_block: list[list[tuple[LicenseType, Any, set[Any]]]] = []
        for block in partition:
            block_choices: list[tuple[LicenseType, Any, set[Any]]] = []
            bsize = len(block)
            for lt in license_types:
                if not (lt.min_capacity <= bsize <= lt.max_capacity):
                    continue
                for owner in block:
                    members = block - {owner}
                    if self._is_valid_group(owner, members, graph):
                        block_choices.append((lt, owner, members))
            per_block.append(block_choices)

        if all(per_block):
            for combo in product(*per_block):
                yield list(combo)

    def _is_valid_group(self, owner: Any, members: set[Any], graph: nx.Graph) -> bool:
        owner_neighbors = set(graph.neighbors(owner))
        return all(m in owner_neighbors for m in members)

    def _is_valid_assignment(self, assignment: Assignment, nodes: list[Any], graph: nx.Graph) -> bool:
        covered: set[Any] = set()

        for lt, owner, members in assignment:
            group_nodes = {owner} | members

            if not self._is_valid_group(owner, members, graph):
                return False
            if not (lt.min_capacity <= len(group_nodes) <= lt.max_capacity):
                return False

            if covered & group_nodes:
                return False
            covered.update(group_nodes)

        return covered == set(nodes)

    def _calculate_cost(self, assignment: Assignment) -> float:
        return sum(lt.cost for lt, _, _ in assignment)

    def _create_solution_from_assignment(self, assignment: Assignment) -> Solution:
        groups = [LicenseGroup(lt, owner, frozenset(members)) for lt, owner, members in assignment]
        return SolutionBuilder.create_solution_from_groups(groups)

```

### src/glopt/algorithms/randomized.py

```python
import random
from typing import Any

import networkx as nx

from glopt.core import Algorithm, LicenseGroup, LicenseType, Solution
from glopt.core.solution_builder import SolutionBuilder


class RandomizedAlgorithm(Algorithm):
    @property
    def name(self) -> str:
        return "randomized_algorithm"

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def solve(self, graph: nx.Graph, license_types: list[LicenseType], **kwargs: Any) -> Solution:
        if len(graph.nodes()) == 0:
            return Solution(groups=())

        runtime_seed = kwargs.get("seed", self.seed)
        if runtime_seed is not None:
            random.seed(runtime_seed)

        nodes = list(graph.nodes())
        uncovered_nodes = set(nodes)
        groups = []

        random.shuffle(nodes)

        for node in nodes:
            if node not in uncovered_nodes:
                continue

            assignment = self._random_assignment(node, uncovered_nodes, graph, license_types)

            if assignment:
                license_type, group_members = assignment
                additional_members = group_members - {node}
                group = LicenseGroup(license_type, node, frozenset(additional_members))
                groups.append(group)
                uncovered_nodes -= group_members

        while uncovered_nodes:
            node = uncovered_nodes.pop()
            cheapest_single = self._find_cheapest_single_license(license_types)
            group = LicenseGroup(cheapest_single, node, frozenset())
            groups.append(group)

        return SolutionBuilder.create_solution_from_groups(groups)

    def _random_assignment(
        self,
        node: Any,
        uncovered_nodes: set[Any],
        graph: nx.Graph,
        license_types: list[LicenseType],
    ) -> tuple[LicenseType, set[Any]] | None:
        neighbors = set(graph.neighbors(node)) & uncovered_nodes
        available_nodes = neighbors | {node}

        compatible_licenses = [lt for lt in license_types if lt.min_capacity <= len(available_nodes)]

        if not compatible_licenses:
            return self._greedy_assignment(node, uncovered_nodes, graph, license_types)

        random.shuffle(compatible_licenses)

        for license_type in compatible_licenses:
            max_possible_size = min(len(available_nodes), license_type.max_capacity)

            if max_possible_size < license_type.min_capacity:
                continue

            group_size = random.randint(license_type.min_capacity, max_possible_size)

            group_members = self._select_random_group_members(node, available_nodes, group_size)

            if len(group_members) >= license_type.min_capacity:
                return (license_type, group_members)

        return self._greedy_assignment(node, uncovered_nodes, graph, license_types)

    def _select_random_group_members(self, owner: Any, available_nodes: set[Any], target_size: int) -> set[Any]:
        if target_size <= 0:
            return set()

        group_members = {owner}
        remaining_slots = target_size - 1

        if remaining_slots <= 0:
            return group_members

        candidates = list(available_nodes - {owner})

        if len(candidates) >= remaining_slots:
            selected_candidates = random.sample(candidates, remaining_slots)
            group_members.update(selected_candidates)
        else:
            group_members.update(candidates)

        return group_members

    def _find_cheapest_single_license(self, license_types: list[LicenseType]) -> LicenseType:
        single_licenses = [lt for lt in license_types if lt.min_capacity <= 1 <= lt.max_capacity]

        if not single_licenses:
            return min(license_types, key=lambda lt: lt.cost)

        return min(single_licenses, key=lambda lt: lt.cost)

```

### src/glopt/algorithms/simulated_annealing.py

```python
import math
import random
from typing import Any

import networkx as nx

from glopt.core import Algorithm, LicenseGroup, LicenseType, Solution
from glopt.core.solution_builder import SolutionBuilder
from glopt.core.solution_validator import SolutionValidator

from .greedy import GreedyAlgorithm


class SimulatedAnnealing(Algorithm):
    @property
    def name(self) -> str:
        return "simulated_annealing"

    def __init__(
        self,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.995,
        min_temperature: float = 0.001,
        max_iterations: int = 20_000,
        max_stall: int = 2_000,
    ) -> None:
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.max_stall = max_stall
        self.validator = SolutionValidator(debug=False)

    def solve(self, graph: nx.Graph, license_types: list[LicenseType], **kwargs: Any) -> Solution:
        # Optional controls
        seed = kwargs.get("seed")
        if isinstance(seed, int):
            random.seed(seed)
        deadline = kwargs.get("deadline")
        initial: Solution | None = kwargs.get("initial_solution")
        max_iterations = int(kwargs.get("max_iterations", self.max_iterations))
        max_stall = int(kwargs.get("max_stall", self.max_stall))

        # Warm start if available and valid; otherwise greedy baseline
        if initial is not None and self.validator.is_valid_solution(initial, graph):
            current = initial
        else:
            current = GreedyAlgorithm().solve(graph, license_types)
        ok, _ = self.validator.validate(current, graph)
        if not ok:
            current = self._fallback_singletons(graph, license_types)

        best = current
        temperature = self.initial_temperature
        stall = 0

        from time import perf_counter as _pc
        for _ in range(max_iterations):
            if deadline is not None and _pc() >= float(deadline):
                break
            if self.min_temperature > temperature:
                break

            neighbor = self._neighbor(current, graph, license_types)
            if neighbor is None:
                stall += 1
            else:
                d = neighbor.total_cost - current.total_cost
                if d < 0 or random.random() < math.exp(-d / max(temperature, 1e-12)):
                    current = neighbor
                    if current.total_cost < best.total_cost:
                        best = current
                        stall = 0
                    else:
                        stall += 1
                else:
                    stall += 1

            if stall >= max_stall:
                stall = 0
                temperature = max(self.min_temperature, temperature * 0.5)

            temperature *= self.cooling_rate

        return best

    def _fallback_singletons(self, graph: nx.Graph, lts: list[LicenseType]) -> Solution:
        lt1 = SolutionBuilder.find_cheapest_single_license(lts)
        groups = [LicenseGroup(lt1, n, frozenset()) for n in graph.nodes()]
        return Solution(groups=tuple(groups))

    def _neighbor(self, solution: Solution, graph: nx.Graph, lts: list[LicenseType]) -> Solution | None:
        moves = [
            self._mv_change_license,
            self._mv_move_member,
            self._mv_swap_members,
            self._mv_merge_groups,
            self._mv_split_group,
        ]
        for _ in range(12):
            mv = random.choice(moves)
            try:
                cand = mv(solution, graph, lts)
            except Exception:
                cand = None
            if cand:
                ok, _ = self.validator.validate(cand, graph)
                if ok:
                    return cand
        return None

    def _mv_change_license(self, solution: Solution, graph: nx.Graph, lts: list[LicenseType]) -> Solution | None:
        if not solution.groups:
            return None
        g = random.choice(solution.groups)
        compat = SolutionBuilder.get_compatible_license_types(g.size, lts, exclude=g.license_type)
        cheaper = [lt for lt in compat if lt.cost < g.license_type.cost]
        if not cheaper:
            return None
        new_lt = random.choice(cheaper)

        new_groups = [LicenseGroup(new_lt, g.owner, g.additional_members) if x is g else x for x in solution.groups]
        return Solution(groups=tuple(new_groups))

    def _mv_move_member(self, solution: Solution, graph: nx.Graph, lts: list[LicenseType]) -> Solution | None:
        donors = [g for g in solution.groups if g.additional_members and g.size > g.license_type.min_capacity]
        if not donors:
            return None
        from_g = random.choice(donors)
        member = random.choice(list(from_g.additional_members))

        receivers = [g for g in solution.groups if g is not from_g and g.size < g.license_type.max_capacity]
        if not receivers:
            return None
        to_g = random.choice(receivers)

        allowed = SolutionBuilder.get_owner_neighbors_with_self(graph, to_g.owner)
        if member not in allowed:
            return None

        new_groups = []
        for g in solution.groups:
            if g is from_g:
                new_groups.append(LicenseGroup(g.license_type, g.owner, g.additional_members - {member}))
            elif g is to_g:
                new_groups.append(LicenseGroup(g.license_type, g.owner, g.additional_members | {member}))
            else:
                new_groups.append(g)
        return Solution(groups=tuple(new_groups))

    def _mv_swap_members(self, solution: Solution, graph: nx.Graph, lts: list[LicenseType]) -> Solution | None:
        if len(solution.groups) < 2:
            return None
        g1, g2 = random.sample(list(solution.groups), 2)

        cand1 = list(g1.all_members)
        cand2 = list(g2.all_members)
        if not cand1 or not cand2:
            return None
        n1 = random.choice(cand1)
        n2 = random.choice(cand2)

        if n1 not in SolutionBuilder.get_owner_neighbors_with_self(graph, g2.owner):
            return None
        if n2 not in SolutionBuilder.get_owner_neighbors_with_self(graph, g1.owner):
            return None

        new_groups: list[LicenseGroup] = []
        for g in solution.groups:
            if g is g1:
                mem = (g.all_members - {n1}) | {n2}
                owner = g.owner if g.owner in mem else n2
                new_groups.append(LicenseGroup(g.license_type, owner, frozenset(mem - {owner})))
            elif g is g2:
                mem = (g.all_members - {n2}) | {n1}
                owner = g.owner if g.owner in mem else n1
                new_groups.append(LicenseGroup(g.license_type, owner, frozenset(mem - {owner})))
            else:
                new_groups.append(g)
        return Solution(groups=tuple(new_groups))

    def _mv_merge_groups(self, solution: Solution, graph: nx.Graph, lts: list[LicenseType]) -> Solution | None:
        if len(solution.groups) < 2:
            return None
        g1, g2 = random.sample(list(solution.groups), 2)
        merged = SolutionBuilder.merge_groups(g1, g2, graph, lts)
        if merged is None:
            return None
        new_groups = [g for g in solution.groups if g not in (g1, g2)]
        new_groups.append(merged)
        return Solution(groups=tuple(new_groups))

    def _mv_split_group(self, solution: Solution, graph: nx.Graph, lts: list[LicenseType]) -> Solution | None:
        splittable = [g for g in solution.groups if g.size >= 3]
        if not splittable:
            return None
        g = random.choice(splittable)
        members = list(g.all_members)

        for _ in range(4):
            random.shuffle(members)
            cut = random.randint(1, len(members) - 1)
            part1, part2 = members[:cut], members[cut:]

            lt1 = SolutionBuilder.find_cheapest_license_for_size(len(part1), lts)
            lt2 = SolutionBuilder.find_cheapest_license_for_size(len(part2), lts)
            if not lt1 or not lt2:
                continue

            owner1 = random.choice(part1)
            owner2 = random.choice(part2)

            neigh1 = SolutionBuilder.get_owner_neighbors_with_self(graph, owner1)
            neigh2 = SolutionBuilder.get_owner_neighbors_with_self(graph, owner2)
            if not set(part1).issubset(neigh1) or not set(part2).issubset(neigh2):
                continue

            new_groups = [x for x in solution.groups if x is not g]
            new_groups.append(LicenseGroup(lt1, owner1, frozenset(set(part1) - {owner1})))
            new_groups.append(LicenseGroup(lt2, owner2, frozenset(set(part2) - {owner2})))
            return Solution(groups=tuple(new_groups))

        return None

```

### src/glopt/algorithms/tabu_search.py

```python
from collections import deque
from typing import Any

import networkx as nx

from glopt.core import Algorithm, LicenseType, Solution
from glopt.core.mutations import MutationOperators
from glopt.core.solution_validator import SolutionValidator

from .greedy import GreedyAlgorithm


class TabuSearch(Algorithm):
    @property
    def name(self) -> str:
        return "tabu_search"

    def solve(
        self,
        graph: nx.Graph,
        license_types: list[LicenseType],
        **kwargs: Any,
    ) -> Solution:
        import random
        seed = kwargs.get("seed")
        if isinstance(seed, int):
            random.seed(seed)
        max_iterations: int = kwargs.get("max_iterations", 1000)
        tabu_tenure: int = kwargs.get("tabu_tenure", 20)
        neighbors_per_iter: int = kwargs.get("neighbors_per_iter", 10)
        deadline = kwargs.get("deadline")
        initial: Solution | None = kwargs.get("initial_solution")

        validator = SolutionValidator(debug=False)

        # Warm start if provided and valid; else greedy baseline
        greedy = GreedyAlgorithm()
        if initial is not None and validator.is_valid_solution(initial, graph):
            current = initial
        else:
            current = greedy.solve(graph, license_types)
        best = current

        tabu: deque[str] = deque(maxlen=max(1, tabu_tenure))
        tabu.append(self._hash(current))

        from time import perf_counter as _pc
        for _ in range(max_iterations):
            if deadline is not None and _pc() >= float(deadline):
                break
            neighborhood: list[Solution] = MutationOperators.generate_neighbors(base=current, graph=graph, license_types=license_types, k=neighbors_per_iter)
            if not neighborhood:
                break

            chosen: Solution | None = None
            chosen_cost = float("inf")

            for cand in neighborhood:
                ok, _ = validator.validate(cand, graph)
                if not ok:
                    continue
                h = self._hash(cand)

                if h in tabu and cand.total_cost >= best.total_cost:
                    continue

                if cand.total_cost < chosen_cost:
                    chosen = cand
                    chosen_cost = cand.total_cost

            if chosen is None:
                break

            current = chosen
            if current.total_cost < best.total_cost:
                best = current

            tabu.append(self._hash(current))

        return best

    def _hash(self, solution: Solution) -> str:
        parts: list[str] = []
        for g in sorted(solution.groups, key=lambda gg: (str(gg.owner), gg.license_type.name)):
            members = ",".join(map(str, sorted(g.all_members)))
            parts.append(f"{g.license_type.name}:{g.owner}:{members}")
        return "|".join(parts)

```

### src/glopt/algorithms/tree_dp.py

```python
from itertools import combinations
from typing import Any

import networkx as nx

from glopt.core import Algorithm, LicenseGroup, LicenseType, Solution
from glopt.core.solution_builder import SolutionBuilder


class TreeDynamicProgramming(Algorithm):
    @property
    def name(self) -> str:
        return "tree_dp"

    def solve(
        self,
        graph: nx.Graph,
        license_types: list[LicenseType],
        **_: Any,
    ) -> Solution:
        if not nx.is_tree(graph):
            msg = "TreeDynamicProgramming requires a tree graph"
            raise ValueError(msg)

        if len(graph.nodes()) == 0:
            return Solution(groups=())

        if len(graph.nodes()) == 1:
            node = next(iter(graph.nodes()))
            cheapest = min(license_types, key=lambda lt: lt.cost if lt.min_capacity <= 1 <= lt.max_capacity else float("inf"))
            group = LicenseGroup(cheapest, node, frozenset())
            return SolutionBuilder.create_solution_from_groups([group])

        root = next(iter(graph.nodes()))
        memo = {}
        cost, groups = self._solve_subtree(graph, root, None, license_types, memo)
        return SolutionBuilder.create_solution_from_groups(groups)

    def _solve_subtree(self, graph: nx.Graph, node: Any, parent: Any, license_types: list[LicenseType], memo: dict) -> tuple[float, list[LicenseGroup]]:
        children = [child for child in graph.neighbors(node) if child != parent]

        state_key = (node, tuple(sorted(children)))
        if state_key in memo:
            return memo[state_key]

        if not children:
            cheapest = min(
                license_types,
                key=lambda lt: lt.cost if lt.min_capacity <= 1 <= lt.max_capacity else float("inf"),
            )
            group = LicenseGroup(cheapest, node, frozenset())
            result = (cheapest.cost, [group])
            memo[state_key] = result
            return result

        child_solutions = {}
        for child in children:
            child_solutions[child] = self._solve_subtree(graph, child, node, license_types, memo)

        best_cost = float("inf")
        best_groups = []

        for license_type in license_types:
            min_capacity = license_type.min_capacity
            max_capacity = license_type.max_capacity

            if max_capacity < 1:
                continue

            for num_children in range(min(len(children), max_capacity - 1) + 1):
                if min_capacity > num_children + 1:
                    continue

                child_combinations = [()] if num_children == 0 else combinations(children, num_children)

                for child_combination in child_combinations:
                    included_children = set(child_combination)
                    remaining_children = [c for c in children if c not in included_children]

                    cost = license_type.cost
                    groups = [LicenseGroup(license_type, node, frozenset(included_children))]

                    for child in remaining_children:
                        child_cost, child_groups = child_solutions[child]
                        cost += child_cost
                        groups.extend(child_groups)

                    for child in included_children:
                        subtree_cost = self._solve_child_subtree(graph, child, node, license_types, memo)
                        cost += subtree_cost[0]
                        groups.extend(subtree_cost[1])

                    if cost < best_cost:
                        best_cost = cost
                        best_groups = groups

        result = (best_cost, best_groups)
        memo[state_key] = result
        return result

    def _solve_child_subtree(self, graph: nx.Graph, child: Any, parent: Any, license_types: list[LicenseType], memo: dict) -> tuple[float, list[LicenseGroup]]:
        grandchildren = [gc for gc in graph.neighbors(child) if gc != parent]

        if not grandchildren:
            return (0.0, [])

        total_cost = 0.0
        all_groups = []

        for grandchild in grandchildren:
            gc_cost, gc_groups = self._solve_subtree(graph, grandchild, child, license_types, memo)
            total_cost += gc_cost
            all_groups.extend(gc_groups)

        return (total_cost, all_groups)

```

### src/glopt/cli/__init__.py

```python

```

### src/glopt/cli/all.py

```python
from datetime import datetime
from typing import Any

from glopt import algorithms
from glopt.core import RunResult, generate_graph, instantiate_algorithms, run_once
from glopt.io import build_paths, ensure_dir, write_csv
from glopt.license_config import LicenseConfigFactory

# Configuration
N_NODES: int = 100
GRAPH_NAMES: list[str] = ["random", "scale_free", "small_world"]
DEFAULT_GRAPH_PARAMS: dict[str, dict[str, Any]] = {
    "random": {"p": 0.1, "seed": 42},
    "scale_free": {"m": 2, "seed": 42},
    "small_world": {"k": 4, "p": 0.1, "seed": 42},
}
LICENSE_CONFIGS: list[str] = ["spotify", "duolingo_super", "roman_domination"]
ALGORITHMS: list[str] = list(algorithms.__all__)


def main() -> int:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_all"
    _, graphs_dir_root, csv_dir = build_paths(run_id)

    print("== glopt all ==")
    print(f"run_id: {run_id}")
    print(f"graphs: {', '.join(GRAPH_NAMES)} n={N_NODES}")
    print(f"licenses: {', '.join(LICENSE_CONFIGS)}")
    print(f"algorithms: {', '.join(ALGORITHMS)}")

    results: list[RunResult] = []
    for graph_name in GRAPH_NAMES:
        params = DEFAULT_GRAPH_PARAMS.get(graph_name, {})
        graph = generate_graph(graph_name, N_NODES, params)

        for lic_name in LICENSE_CONFIGS:
            license_types = LicenseConfigFactory.get_config(lic_name)
            g_dir = f"{graphs_dir_root}/{graph_name}/{lic_name}"
            ensure_dir(g_dir)
            print(f"-> {graph_name} {lic_name}")

            for algo_name in ALGORITHMS:
                algo = instantiate_algorithms([algo_name])[0]
                print(f"   running {algo.name}...")
                r = run_once(
                    algo=algo,
                    graph=graph,
                    license_types=license_types,
                    run_id=run_id,
                    graphs_dir=g_dir,
                    print_issue_limit=10,
                )
                print(f"     cost={r.total_cost:.2f} time_ms={r.time_ms:.2f} valid={r.valid} issues={r.issues}")
                r = RunResult(
                    **{
                        **r.__dict__,
                        "graph": graph_name,
                        "graph_params": str(params),
                        "license_config": lic_name,
                    },
                )
                results.append(r)

    csv_path = write_csv(csv_dir, run_id, results)
    print("== summary ==")
    print(f"runs: {len(results)}")
    print(f"csv: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

### src/glopt/cli/benchmark.py

```python
from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import networkx as nx
import json
import pickle
import multiprocessing as mp

from glopt.core import generate_graph, instantiate_algorithms
from glopt.core.solution_validator import SolutionValidator
from glopt.io import ensure_dir
from glopt.license_config import LicenseConfigFactory

# ==============================================
# Simple, tweakable configuration (no CLI/env)
# ==============================================

# Optional custom run id suffix. If None, timestamp is used.
RUN_ID: str | None = None

# Graph families and default params
GRAPH_NAMES: list[str] = ["random", "small_world", "scale_free"]
GRAPH_DEFAULTS: dict[str, dict[str, Any]] = {
    "random": {"p": 0.10, "seed": 42},
    "small_world": {"k": 6, "p": 0.05, "seed": 42},
    "scale_free": {"m": 2, "seed": 42},
}

# Sizes: dense grid for small n, coarser for larger n
SIZES_SMALL: list[int] = list(range(20, 201, 20))
SIZES_LARGE: list[int] = [300, 400, 600, 800, 1000, 1500, 2000, 2500, 3000]
SIZES: list[int] = SIZES_SMALL + SIZES_LARGE

# Experiments: number of independent graph samples per (graph, n)
SAMPLES_PER_SIZE: int = 3  # increase for more robust averages

# Repeated runs of stochastic solvers on the same graph
REPEATS_PER_GRAPH: int = 2  # e.g., different algorithm seeds

# Per-run time budget (hard cap)
TIMEOUT_SECONDS: float = 90.0

# License configurations and algorithms under test
LICENSE_CONFIG_NAMES: list[str] = ["spotify", "duolingo_super", "roman_domination"]
DYNAMIC_ROMAN_PS: list[float] = [1.5, 2.0, 2.5, 3.0]
LICENSE_CONFIG_NAMES.extend([f"roman_p_{str(p).replace('.', '_')}" for p in DYNAMIC_ROMAN_PS])
ALGORITHM_CLASSES: list[str] = [
    # "ILPSolver",  # exact (small n only)
    "GreedyAlgorithm",
    "RandomizedAlgorithm",
    "DominatingSetAlgorithm",
    "AntColonyOptimization",
    "SimulatedAnnealing",
    "TabuSearch",
    "GeneticAlgorithm",
    # "NaiveAlgorithm",          # enable for very small n only
    # "TreeDynamicProgramming",  # only for trees
]

# Optional cap for ILP to avoid extreme runs (set to None to disable)
ILP_MAX_N: int | None = None

# Graph cache directory
GRAPH_CACHE_DIR: str = "data/graphs_cache"


def _adjust_params(name: str, n: int, base: dict[str, Any]) -> dict[str, Any]:
    p = dict(base)
    # scale_free: keep m reasonable vs n
    if name == "scale_free":
        m = int(p.get("m", 2))
        p["m"] = max(1, min(m, max(1, n - 1)))
    # small_world: enforce even k within [2, n-1]
    if name == "small_world":
        k = int(p.get("k", 6))
        if n > 2:
            k = max(2, min(k, n - 1))
            if k % 2 == 1:
                k = k + 1 if k + 1 < n else k - 1
        else:
            k = 2
        p["k"] = k
    return p


def _write_row(csv_path: Path, row: dict[str, object]) -> None:
    import csv as _csv

    is_new = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            w.writeheader()
        w.writerow(row)


def _json_dumps(obj: Any) -> str:
    import json as _json

    try:
        return _json.dumps(obj, ensure_ascii=False)
    except Exception:
        return "{}"


def _cache_paths(cache_dir: str, gname: str, n: int, sample: int) -> tuple[Path, Path]:
    base = Path(cache_dir) / gname / f"n{n:04d}"
    gpath = base / f"s{sample}.gpickle"
    mpath = gpath.with_suffix(".json")
    return gpath, mpath


def _ensure_cache_for_all() -> None:
    ensure_dir(GRAPH_CACHE_DIR)
    created = 0
    for gname in GRAPH_NAMES:
        for n in SIZES:
            base_params = GRAPH_DEFAULTS.get(gname, {})
            base_params = _adjust_params(gname, n, base_params)
            for s_idx in range(SAMPLES_PER_SIZE):
                gpath, mpath = _cache_paths(GRAPH_CACHE_DIR, gname, n, s_idx)
                if gpath.exists() and mpath.exists():
                    continue
                seed = int((base_params.get("seed", 42) or 42) + s_idx * 1009)
                params = dict(base_params)
                params["seed"] = seed
                G = generate_graph(gname, n, params)
                Path(gpath).parent.mkdir(parents=True, exist_ok=True)
                with gpath.open("wb") as f:
                    pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
                meta = {"type": gname, "n": n, "params": params, "sample": s_idx}
                with mpath.open("w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False)
                created += 1
    if created:
        print(f"graph cache: generated {created} new graphs under {GRAPH_CACHE_DIR}")
    else:
        print(f"graph cache: up-to-date at {GRAPH_CACHE_DIR}")


def _load_cached_graph(gname: str, n: int, sample_idx: int) -> tuple[nx.Graph, dict[str, Any]]:
    gpath, mpath = _cache_paths(GRAPH_CACHE_DIR, gname, n, sample_idx)
    with gpath.open("rb") as f:
        G: nx.Graph = pickle.load(f)
    params: dict[str, Any] = {}
    if mpath.exists():
        try:
            with mpath.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            params = dict(meta.get("params", {}))
        except Exception:
            params = {}
    return G, params


def _worker_solve(algo_name: str, graph: nx.Graph, license_config: str, seed: int, conn) -> None:  # type: ignore[no-redef]
    """Child process: run solve once and return metrics via pipe."""
    try:
        validator = SolutionValidator(debug=False)
        algo = instantiate_algorithms([algo_name])[0]
        lts = LicenseConfigFactory.get_config(license_config)
        t0 = perf_counter()
        sol = algo.solve(graph, lts, seed=seed)
        elapsed_ms = (perf_counter() - t0) * 1000.0

        ok, issues = validator.validate(sol, graph)
        sizes = [g.size for g in sol.groups]
        sizes_sorted = sorted(sizes)
        groups = len(sizes)
        mean_sz = (sum(sizes) / groups) if groups else 0.0
        if groups:
            mid = groups // 2
            if groups % 2 == 1:
                median_sz = float(sizes_sorted[mid])
            else:
                median_sz = (sizes_sorted[mid - 1] + sizes_sorted[mid]) / 2.0
            p90 = float(sizes_sorted[min(groups - 1, int(0.9 * (groups - 1)))])
        else:
            median_sz = 0.0
            p90 = 0.0
        lic_counts = Counter(g.license_type.name for g in sol.groups)
        res = {
            "success": True,
            "total_cost": float(sol.total_cost),
            "time_ms": float(elapsed_ms),
            "valid": bool(ok),
            "issues": int(len(issues)),
            "groups": int(groups),
            "group_size_mean": float(mean_sz),
            "group_size_median": float(median_sz),
            "group_size_p90": float(p90),
            "license_counts_json": _json_dumps(lic_counts),
            "cost_per_node": float(sol.total_cost) / max(1, graph.number_of_nodes()),
        }
    except Exception as e:  # defensive: return error to parent instead of crashing worker
        res = {"success": False, "error": str(e)}
    try:
        conn.send(res)
    finally:
        conn.close()


def _run_one(
    algo_name: str,
    graph: nx.Graph,
    license_config: str,
    seed: int,
) -> tuple[dict[str, object], bool]:
    parent_conn, child_conn = mp.Pipe(duplex=False)
    p = mp.Process(target=_worker_solve, args=(algo_name, graph, license_config, seed, child_conn))
    p.start()
    timed_out = False
    res: dict[str, object]
    if parent_conn.poll(TIMEOUT_SECONDS):
        try:
            msg = parent_conn.recv()
        except EOFError:
            msg = {"success": False, "error": "no-data"}
        p.join()
        if msg.get("success"):
            res = {
                "total_cost": float(msg.get("total_cost", float("nan"))),
                "time_ms": float(msg.get("time_ms", 0.0)),
                "valid": bool(msg.get("valid", False)),
                "issues": int(msg.get("issues", 0)),
                "groups": int(msg.get("groups", 0)),
                "group_size_mean": float(msg.get("group_size_mean", 0.0)),
                "group_size_median": float(msg.get("group_size_median", 0.0)),
                "group_size_p90": float(msg.get("group_size_p90", 0.0)),
                "license_counts_json": str(msg.get("license_counts_json", "{}")),
                "cost_per_node": float(msg.get("cost_per_node", 0.0)),
                "notes": "",
            }
        else:
            res = {
                "total_cost": float("nan"),
                "time_ms": float(0.0),
                "valid": False,
                "issues": 0,
                "groups": 0,
                "group_size_mean": 0.0,
                "group_size_median": 0.0,
                "group_size_p90": 0.0,
                "license_counts_json": "{}",
                "notes": "error",
            }
    else:
        # Timeout: kill child and report
        timed_out = True
        try:
            p.terminate()
        finally:
            p.join()
        res = {
            "total_cost": float("nan"),
            "time_ms": float(TIMEOUT_SECONDS * 1000.0),
            "valid": False,
            "issues": 0,
            "groups": 0,
            "group_size_mean": 0.0,
            "group_size_median": 0.0,
            "group_size_p90": 0.0,
            "license_counts_json": "{}",
            "notes": "timeout",
        }

    return res, timed_out or (res.get("notes") == "error")


def main() -> None:
    run_id = (RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")) + "_benchmark"
    base = Path("runs") / run_id
    csv_dir = base / "csv"
    ensure_dir(str(csv_dir))
    out_path = csv_dir / f"{run_id}.csv"

    print("== glopt benchmark ==")
    print(f"run_id: {run_id}")
    print(f"graphs: {', '.join(GRAPH_NAMES)}")
    print(f"sizes: {SIZES[0]}..{SIZES[-1]} ({len(SIZES)} points)")
    print(f"samples/size: {SAMPLES_PER_SIZE} repeats/graph: {REPEATS_PER_GRAPH}")
    print(f"licenses: {', '.join(LICENSE_CONFIG_NAMES)}")
    print(f"algorithms: {', '.join(ALGORITHM_CLASSES)}")
    print(f"timeout limit: {TIMEOUT_SECONDS:.0f}s (kills run and stops larger sizes)")
    print("warming up graph cache …")
    _ensure_cache_for_all()

    for lic_name in LICENSE_CONFIG_NAMES:
        for algo_name in ALGORITHM_CLASSES:
            print(f"-> {lic_name} / {algo_name}")
            for gname in GRAPH_NAMES:
                stop_sizes = False
                for n in SIZES:
                    if stop_sizes:
                        break
                    if ILP_MAX_N is not None and algo_name == "ILPSolver" and n > ILP_MAX_N:
                        print(f"   {gname:12s} n={n:4d} ILP capped at {ILP_MAX_N} — skipping remaining sizes")
                        stop_sizes = True
                        break

                    # Load SAMPLES_PER_SIZE graph instances from cache
                    for s_idx in range(SAMPLES_PER_SIZE):
                        G, params = _load_cached_graph(gname, n, s_idx)
                        graph_seed = int(params.get("seed", 0) or 0)

                        # Graph metrics
                        n_nodes = G.number_of_nodes()
                        n_edges = G.number_of_edges()
                        density = (2.0 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
                        avg_deg = (2.0 * n_edges) / n_nodes if n_nodes > 0 else 0.0
                        clustering = nx.average_clustering(G) if n_nodes > 1 else 0.0
                        components = nx.number_connected_components(G)

                        over_here = False
                        for rep in range(REPEATS_PER_GRAPH):
                            algo_seed = 12345 + s_idx * 1000 + rep
                            result, is_over = _run_one(algo_name, G, lic_name, algo_seed)
                            row = {
                                "run_id": run_id,
                                "algorithm": algo_name,
                                "graph": gname,
                                "n_nodes": n_nodes,
                                "n_edges": n_edges,
                                "graph_params": str(params),
                                "license_config": lic_name,
                                "rep": rep,
                                "seed": algo_seed,
                                "sample": s_idx,
                                "graph_seed": int(graph_seed),
                                "density": float(density),
                                "avg_degree": float(avg_deg),
                                "clustering": float(clustering),
                                "components": int(components),
                                "image_path": "",
                                **result,
                            }
                            _write_row(out_path, row)
                            status = "OK"
                            if row.get("notes") == "timeout":
                                status = "TIMEOUT"
                            elif row.get("notes") == "error":
                                status = "ERROR"
                            print(
                                f"   {gname:12s} n={n:4d} s={s_idx} rep={rep} cost={row['total_cost']:.2f} time_ms={row['time_ms']:.2f} valid={row['valid']} {status}"
                            )
                            if is_over:
                                over_here = True
                        if over_here:
                            print(f"   {gname:12s} n={n:4d} TIMEOUT — stopping larger sizes for {algo_name} on this graph")
                            stop_sizes = True
                            break

    if out_path.exists():
        print(f"csv: {out_path}")


if __name__ == "__main__":
    main()

```

### src/glopt/cli/benchmark_real.py

```python
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import networkx as nx
import multiprocessing as mp

from glopt.core import instantiate_algorithms
from glopt.core.solution_validator import SolutionValidator
from glopt.io import ensure_dir
from glopt.license_config import LicenseConfigFactory
from glopt.io.data_loader import RealWorldDataLoader

# Configuration
RUN_ID: str | None = None
ALGORITHM_CLASSES: list[str] = [
    "ILPSolver",
    "GreedyAlgorithm",
    "RandomizedAlgorithm",
    "DominatingSetAlgorithm",
    "AntColonyOptimization",
    "SimulatedAnnealing",
    "TabuSearch",
    "GeneticAlgorithm",
]
LICENSE_CONFIG_NAMES: list[str] = ["spotify", "duolingo_super", "roman_domination"]
DYNAMIC_ROMAN_PS: list[float] = [1.5, 2.0, 2.5, 3.0]
LICENSE_CONFIG_NAMES.extend([f"roman_p_{str(p).replace('.', '_')}" for p in DYNAMIC_ROMAN_PS])

TIMEOUT_SECONDS: float = 60.0


def _json_dumps(obj: Any) -> str:
    import json as _json

    try:
        return _json.dumps(obj, ensure_ascii=False)
    except Exception:
        return "{}"


def _worker_solve(algo_name: str, graph: nx.Graph, license_config: str, seed: int, conn) -> None:  # type: ignore[no-redef]
    try:
        validator = SolutionValidator(debug=False)
        algo = instantiate_algorithms([algo_name])[0]
        lts = LicenseConfigFactory.get_config(license_config)
        t0 = perf_counter()
        sol = algo.solve(graph, lts, seed=seed)
        elapsed_ms = (perf_counter() - t0) * 1000.0

        ok, issues = validator.validate(sol, graph)
        sizes = [g.size for g in sol.groups]
        groups = len(sizes)
        mean_sz = (sum(sizes) / groups) if groups else 0.0
        med_sz = float(sorted(sizes)[groups // 2]) if groups else 0.0
        p90 = float(sorted(sizes)[min(groups - 1, int(0.9 * (groups - 1)))]) if groups else 0.0
        res = {
            "success": True,
            "total_cost": float(sol.total_cost),
            "time_ms": float(elapsed_ms),
            "valid": bool(ok),
            "issues": int(len(issues)),
            "groups": int(groups),
            "group_size_mean": float(mean_sz),
            "group_size_median": float(med_sz),
            "group_size_p90": float(p90),
            "cost_per_node": float(sol.total_cost) / max(1, graph.number_of_nodes()),
        }
    except Exception as e:  # defensive: return error to parent instead of crashing worker
        res = {"success": False, "error": str(e)}
    try:
        conn.send(res)
    finally:
        conn.close()


def _run_one(algo: str, graph: nx.Graph, lic: str, seed: int) -> tuple[dict[str, object], bool]:
    parent_conn, child_conn = mp.Pipe(duplex=False)
    p = mp.Process(target=_worker_solve, args=(algo, graph, lic, seed, child_conn))
    p.start()
    timed_out = False
    if parent_conn.poll(TIMEOUT_SECONDS):
        try:
            msg = parent_conn.recv()
        except EOFError:
            msg = {"success": False, "error": "no-data"}
        p.join()
        if msg.get("success"):
            res = {
                "total_cost": float(msg.get("total_cost", float("nan"))),
                "time_ms": float(msg.get("time_ms", 0.0)),
                "valid": bool(msg.get("valid", False)),
                "issues": int(msg.get("issues", 0)),
                "groups": int(msg.get("groups", 0)),
                "group_size_mean": float(msg.get("group_size_mean", 0.0)),
                "group_size_median": float(msg.get("group_size_median", 0.0)),
                "group_size_p90": float(msg.get("group_size_p90", 0.0)),
                "cost_per_node": float(msg.get("cost_per_node", 0.0)),
                "notes": "",
            }
        else:
            res = {"total_cost": float("nan"), "time_ms": 0.0, "valid": False, "issues": 0, "groups": 0, "group_size_mean": 0.0, "group_size_median": 0.0, "group_size_p90": 0.0, "cost_per_node": 0.0, "notes": "error"}
    else:
        timed_out = True
        try:
            p.terminate()
        finally:
            p.join()
        res = {"total_cost": float("nan"), "time_ms": float(TIMEOUT_SECONDS * 1000.0), "valid": False, "issues": 0, "groups": 0, "group_size_mean": 0.0, "group_size_median": 0.0, "group_size_p90": 0.0, "cost_per_node": 0.0, "notes": "timeout"}
    return res, timed_out or (res.get("notes") == "error")


def main() -> None:
    run_id = (RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")) + "_benchmark_real"
    base = Path("runs") / run_id
    csv_dir = base / "csv"
    ensure_dir(str(csv_dir))
    out_path = csv_dir / f"{run_id}.csv"

    print("== glopt benchmark real ==")
    print(f"run_id: {run_id}")
    print(f"algorithms: {', '.join(ALGORITHM_CLASSES)}")
    print(f"licenses: {', '.join(LICENSE_CONFIG_NAMES)}")
    print(f"timeout: {TIMEOUT_SECONDS:.0f}s")

    loader = RealWorldDataLoader(data_dir="data")
    networks = loader.load_all_facebook_networks()
    ego_ids = sorted(networks.keys())
    print(f"facebook ego networks: {len(ego_ids)}")

    # Write CSV header
    import csv as _csv
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = _csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "graph",
                "graph_params",
                "license_config",
                "algorithm",
                "ego_id",
                "n_nodes",
                "n_edges",
                "density",
                "avg_degree",
                "clustering",
                "components",
                "total_cost",
                "cost_per_node",
                "time_ms",
                "valid",
                "issues",
                "groups",
                "group_size_mean",
                "group_size_median",
                "group_size_p90",
                "notes",
            ],
        )
        w.writeheader()

        for ego_id in ego_ids:
            G = networks[ego_id]
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            density = (2.0 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0.0
            avg_deg = (2.0 * n_edges) / n_nodes if n_nodes > 0 else 0.0
            clustering = nx.average_clustering(G) if n_nodes > 1 else 0.0
            components = nx.number_connected_components(G)

            for lic in LICENSE_CONFIG_NAMES:
                for algo in ALGORITHM_CLASSES:
                    print(f"-> ego={ego_id} lic={lic} algo={algo}")
                    res, _ = _run_one(algo, G, lic, seed=12345)
                    row = {
                        "run_id": run_id,
                        "graph": "facebook_ego",
                        "graph_params": _json_dumps({"ego_id": ego_id}),
                        "license_config": lic,
                        "algorithm": algo,
                        "ego_id": ego_id,
                        "n_nodes": n_nodes,
                        "n_edges": n_edges,
                        "density": float(density),
                        "avg_degree": float(avg_deg),
                        "clustering": float(clustering),
                        "components": int(components),
                        **res,
                    }
                    w.writerow(row)

    print(f"csv: {out_path}")


if __name__ == "__main__":
    main()

```

### src/glopt/cli/custom.py

```python
from __future__ import annotations

from datetime import datetime
from typing import Any

from glopt.core import RunResult, generate_graph, instantiate_algorithms, run_once
from glopt.io import build_paths, write_csv
from glopt.license_config import LicenseConfigFactory

# Configuration
RUN_ID: str | None = None
GRAPH_NAME: str = "small_world"
GRAPH_PARAMS: dict[str, Any] = {"k": 4, "p": 0.1, "seed": 42}
N_NODES: int = 100
LICENSE_CONFIG_NAME: str = "spotify"
ALGORITHMS: list[str] = [
    "ILPSolver",
    # "NaiveAlgorithm",
    "RandomizedAlgorithm",
    "GreedyAlgorithm",
    "DominatingSetAlgorithm",
    "AntColonyOptimization",
    "SimulatedAnnealing",
    "TabuSearch",
    "GeneticAlgorithm",
    # "TreeDynamicProgramming",
]


def main() -> None:
    run_id = (RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")) + "_custom"
    _, graphs_dir, csv_dir = build_paths(run_id)

    # Print header config
    print("== glopt custom run ==")
    print(f"run_id: {run_id}")
    print(f"graph: {GRAPH_NAME} n={N_NODES} params={GRAPH_PARAMS}")
    print(f"license: {LICENSE_CONFIG_NAME}")
    print(f"algorithms: {', '.join(ALGORITHMS)}")

    graph = generate_graph(GRAPH_NAME, N_NODES, GRAPH_PARAMS)
    license_types = LicenseConfigFactory.get_config(LICENSE_CONFIG_NAME)
    algos = instantiate_algorithms(ALGORITHMS)

    results: list[RunResult] = []
    for algo in algos:
        print(f"-> running {algo.name}...")
        r = run_once(
            algo=algo,
            graph=graph,
            license_types=license_types,
            run_id=run_id,
            graphs_dir=graphs_dir,
            print_issue_limit=10,
        )
        print(f"   done: cost={r.total_cost:.2f} time_ms={r.time_ms:.2f} valid={r.valid} issues={r.issues}")
        r = RunResult(
            **{
                **r.__dict__,
                "graph": GRAPH_NAME,
                "graph_params": str(GRAPH_PARAMS),
                "license_config": LICENSE_CONFIG_NAME,
            },
        )
        results.append(r)

    csv_path = write_csv(csv_dir, run_id, results)

    # Summary
    print("== summary ==")
    for r in results:
        print(f"{r.algorithm}: cost={r.total_cost:.2f} time_ms={r.time_ms:.2f} valid={r.valid}")
        if r.image_path:
            print(f" image: {r.image_path}")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()

```

### src/glopt/cli/dynamic.py

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import networkx as nx

from glopt.core import instantiate_algorithms
from glopt.core.solution_validator import SolutionValidator
from glopt.dynamic_simulator import DynamicNetworkSimulator, MutationParams
from glopt.io import build_paths, ensure_dir
from glopt.io.graph_generator import GraphGeneratorFactory
from glopt.license_config import LicenseConfigFactory
from glopt.algorithms.greedy import GreedyAlgorithm
from glopt.core.solution_builder import SolutionBuilder


# ==============================================
# Simple, tweakable configuration
# ==============================================

RUN_ID: str | None = None

GRAPH_NAMES: list[str] = ["random", "small_world", "scale_free"]
GRAPH_DEFAULTS: dict[str, dict[str, Any]] = {
    "random": {"p": 0.10, "seed": 7},
    "small_world": {"k": 6, "p": 0.05, "seed": 7},
    "scale_free": {"m": 2, "seed": 7},
}

N_NODES: int = 60
LICENSE_CONFIGS: list[str] = ["spotify", "duolingo_super", "roman_domination"]

# Dynamic changes
NUM_STEPS: int = 10
# Probabilities per step (light to moderate churn)
ADD_NODES_PROB: float = 0.06
REMOVE_NODES_PROB: float = 0.04
ADD_EDGES_PROB: float = 0.18
REMOVE_EDGES_PROB: float = 0.12
RANDOM_SEED: int = 123

# Algorithms: warm-start capable vs baselines
WARM_ALGOS: list[str] = [
    "GeneticAlgorithm",
    "SimulatedAnnealing",
    "TabuSearch",
    "AntColonyOptimization",
]
BASELINE_ALGOS: list[str] = [
    "GreedyAlgorithm",
    "ILPSolver",
]


def _adjust_params(name: str, n: int, base: dict[str, Any]) -> dict[str, Any]:
    p = dict(base)
    if name == "scale_free":
        m = int(p.get("m", 2))
        p["m"] = max(1, min(m, max(1, n - 1)))
    if name == "small_world":
        k = int(p.get("k", 6))
        if n > 2:
            k = max(2, min(k, n - 1))
            if k % 2 == 1:
                k = k + 1 if k + 1 < n else k - 1
        else:
            k = 2
        p["k"] = k
    return p


def _repair_solution_for_graph(prev_solution, graph: nx.Graph, license_types) -> Any:
    # Keep groups with owner+members present and neighbor constraint satisfied
    ok_groups = []
    nodes = set(graph.nodes())
    for g in prev_solution.groups:
        if g.owner not in nodes:
            continue
        allowed = set(graph.neighbors(g.owner)) | {g.owner}
        allm = set(g.all_members) & nodes
        if not allm:
            continue
        if not allm.issubset(allowed):
            continue
        try:
            new_g = type(g)(g.license_type, g.owner, frozenset(allm - {g.owner}))
        except Exception:
            continue
        ok_groups.append(new_g)

    covered = set().union(*(g.all_members for g in ok_groups)) if ok_groups else set()
    uncovered = set(graph.nodes()) - covered
    if uncovered:
        H = graph.subgraph(uncovered).copy()
        greedy = GreedyAlgorithm().solve(H, list(license_types))
        ok_groups.extend(greedy.groups)
    return SolutionBuilder.create_solution_from_groups(ok_groups)


def _write_csv_header(path: Path, header: list[str]) -> None:
    import csv

    ensure_dir(str(path.parent))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)


def _append_csv_row(path: Path, row: list[Any]) -> None:
    import csv

    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(row)


def main() -> int:
    run_id = (RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")) + "_dynamic"
    _, _, csv_dir = build_paths(run_id)
    out_path = Path(csv_dir) / f"{run_id}.csv"

    print("== glopt dynamic benchmark ==")
    print(f"run_id: {run_id}")
    print(f"graphs: {', '.join(GRAPH_NAMES)} n={N_NODES}")
    print(f"licenses: {', '.join(LICENSE_CONFIGS)}")
    print(f"warm algos: {', '.join(WARM_ALGOS)}")
    print(f"baselines: {', '.join(BASELINE_ALGOS)}")
    print(f"steps: {NUM_STEPS} seed={RANDOM_SEED}")

    header = [
        "run_id",
        "graph",
        "graph_params",
        "license_config",
        "algorithm",
        "warm_start",
        "step",
        "n_nodes",
        "n_edges",
        "total_cost",
        "time_ms",
        "delta_cost",
        "avg_time_ms_per_step",
        "delta_cost_abs",
        "delta_cost_std_so_far",
        "valid",
        "issues",
        "groups",
        "mutations",
    ]
    _write_csv_header(out_path, header)

    validator = SolutionValidator(debug=False)

    for gname in GRAPH_NAMES:
        params = _adjust_params(gname, N_NODES, GRAPH_DEFAULTS.get(gname, {}))
        gen = GraphGeneratorFactory.get(gname)
        graph = gen(n_nodes=N_NODES, **params)

        # Prepare mutation params scaled to graph size and shared across algorithms
        E0 = graph.number_of_edges()
        mut_params = MutationParams(
            add_nodes_prob=ADD_NODES_PROB,
            remove_nodes_prob=REMOVE_NODES_PROB,
            add_edges_prob=ADD_EDGES_PROB,
            remove_edges_prob=REMOVE_EDGES_PROB,
            max_nodes_add=max(1, int(0.03 * N_NODES)),
            max_nodes_remove=max(1, int(0.02 * N_NODES)),
            max_edges_add=max(1, int(0.04 * (E0 or N_NODES))),
            max_edges_remove=max(1, int(0.03 * (E0 or N_NODES))),
        )
        sim = DynamicNetworkSimulator(mutation_params=mut_params, seed=RANDOM_SEED)
        sim.next_node_id = max(graph.nodes()) + 1 if graph.nodes() else 0

        graphs: list[nx.Graph] = [graph.copy()]
        mutations_per_step: list[list[str]] = [[]]
        g_curr = graph.copy()
        for _ in range(NUM_STEPS):
            g_curr, muts = sim._apply_mutations(g_curr)  # type: ignore[attr-defined]
            graphs.append(g_curr.copy())
            mutations_per_step.append(muts)

        for lic in LICENSE_CONFIGS:
            lts = LicenseConfigFactory.get_config(lic)

            # Instantiate algorithms once per combo
            warm_algos = instantiate_algorithms(WARM_ALGOS)
            base_algos = instantiate_algorithms(BASELINE_ALGOS)

            # Initial solutions at step 0 and accumulators
            prev_solutions: dict[tuple[str, bool], Any] = {}
            time_accum: dict[tuple[str, bool], tuple[float, int]] = {}
            delta_stats: dict[tuple[str, bool], tuple[float, float, int]] = {}
            for algo in warm_algos + base_algos:
                G0 = graphs[0]
                t0 = perf_counter()
                sol0 = algo.solve(G0, lts, seed=RANDOM_SEED)
                elapsed = (perf_counter() - t0) * 1000.0
                ok, issues = validator.validate(sol0, G0)
                key = (algo.name, False)
                prev_solutions[key] = sol0
                time_accum[key] = (elapsed, 1)
                delta_stats[key] = (0.0, 0.0, 0)
                print(f"init -> {gname} / {lic} / {algo.name:<24s} cold   cost={sol0.total_cost:.2f} time_ms={elapsed:.2f} valid={ok} groups={len(sol0.groups)}")
                _append_csv_row(
                    out_path,
                    [
                        run_id,
                        gname,
                        str(params),
                        lic,
                        algo.name,
                        False,
                        0,
                        G0.number_of_nodes(),
                        G0.number_of_edges(),
                        float(sol0.total_cost),
                        float(elapsed),
                        0.0,
                        float(elapsed),
                        0.0,
                        0.0,
                        bool(ok),
                        int(len(issues)),
                        int(len(sol0.groups)),
                        "; ".join(mutations_per_step[0]),
                    ],
                )

            # Dynamic steps 1..NUM_STEPS
            for step in range(1, NUM_STEPS + 1):
                Gs = graphs[step]
                muts = "; ".join(mutations_per_step[step])
                print(
                    f"step {step:02d} -> graph={gname} lic={lic} nodes={Gs.number_of_nodes()} edges={Gs.number_of_edges()} mutations=[{muts}]"
                )

                # Warm-start algorithms (warm and cold variants)
                for algo in warm_algos:
                    # WARM RUN
                    prev_warm = prev_solutions.get((algo.name, True))
                    if prev_warm is None:
                        prev_warm = prev_solutions.get((algo.name, False))
                    warm = _repair_solution_for_graph(prev_warm, Gs, lts) if prev_warm is not None else None
                    t0 = perf_counter()
                    sol = algo.solve(Gs, lts, seed=RANDOM_SEED + step, initial_solution=warm)
                    elapsed = (perf_counter() - t0) * 1000.0
                    ok, issues = validator.validate(sol, Gs)
                    key_w = (algo.name, True)
                    prev_cost = float(prev_solutions.get(key_w, sol).total_cost) if isinstance(prev_solutions.get(key_w), type(sol)) else (
                        float(prev_solutions.get((algo.name, False), sol).total_cost) if isinstance(prev_solutions.get((algo.name, False)), type(sol)) else float('nan')
                    )
                    delta = float(sol.total_cost) - (prev_cost if prev_cost == prev_cost else float('nan'))
                    # update accumulators
                    tot, cnt = time_accum.get(key_w, (0.0, 0))
                    avg = (tot + elapsed) / (cnt + 1)
                    time_accum[key_w] = (tot + elapsed, cnt + 1)
                    # delta stats (Welford)
                    mean_d, m2_d, k = delta_stats.get(key_w, (0.0, 0.0, 0))
                    x = float(delta) if delta == delta else 0.0
                    k1 = k + 1
                    d = x - mean_d
                    mean_new = mean_d + d / k1
                    m2_new = m2_d + d * (x - mean_new)
                    delta_stats[key_w] = (mean_new, m2_new, k1)
                    prev_solutions[key_w] = sol
                    print(
                        f"   warm  {algo.name:<24s} cost={sol.total_cost:.2f} time_ms={elapsed:.2f} dCost={delta if delta==delta else 0.0:+.2f} valid={ok} groups={len(sol.groups)}"
                    )
                    _append_csv_row(
                        out_path,
                        [
                            run_id,
                            gname,
                            str(params),
                            lic,
                            algo.name,
                            True,
                            step,
                            Gs.number_of_nodes(),
                            Gs.number_of_edges(),
                            float(sol.total_cost),
                            float(elapsed),
                            float(delta if delta == delta else 0.0),
                            float(avg),
                            float(abs(delta) if delta == delta else 0.0),
                            float(((delta_stats[key_w][1] / max(1, delta_stats[key_w][2]-1)) ** 0.5) if delta_stats[key_w][2] > 1 else 0.0),
                            bool(ok),
                            int(len(issues)),
                            int(len(sol.groups)),
                            muts,
                        ],
                    )

                    # COLD RUN (for warm algos)
                    t0 = perf_counter()
                    sol_cold = algo.solve(Gs, lts, seed=RANDOM_SEED + step)
                    elapsed_c = (perf_counter() - t0) * 1000.0
                    ok_c, issues_c = validator.validate(sol_cold, Gs)
                    key_c = (algo.name, False)
                    prev_cost_c = float(prev_solutions.get(key_c, sol_cold).total_cost) if isinstance(prev_solutions.get(key_c), type(sol_cold)) else float('nan')
                    delta_c = float(sol_cold.total_cost) - (prev_cost_c if prev_cost_c == prev_cost_c else float('nan'))
                    tot_c, cnt_c = time_accum.get(key_c, (0.0, 0))
                    avg_c = (tot_c + elapsed_c) / (cnt_c + 1)
                    time_accum[key_c] = (tot_c + elapsed_c, cnt_c + 1)
                    mean_d, m2_d, k = delta_stats.get(key_c, (0.0, 0.0, 0))
                    x = float(delta_c) if delta_c == delta_c else 0.0
                    k1 = k + 1
                    d = x - mean_d
                    mean_new = mean_d + d / k1
                    m2_new = m2_d + d * (x - mean_new)
                    delta_stats[key_c] = (mean_new, m2_new, k1)
                    prev_solutions[key_c] = sol_cold
                    print(
                        f"   cold  {algo.name:<24s} cost={sol_cold.total_cost:.2f} time_ms={elapsed_c:.2f} dCost={delta_c if delta_c==delta_c else 0.0:+.2f} valid={ok_c} groups={len(sol_cold.groups)}"
                    )
                    _append_csv_row(
                        out_path,
                        [
                            run_id,
                            gname,
                            str(params),
                            lic,
                            algo.name,
                            False,
                            step,
                            Gs.number_of_nodes(),
                            Gs.number_of_edges(),
                            float(sol_cold.total_cost),
                            float(elapsed_c),
                            float(delta_c if delta_c == delta_c else 0.0),
                            float(avg_c),
                            float(abs(delta_c) if delta_c == delta_c else 0.0),
                            float(((delta_stats[key_c][1] / max(1, delta_stats[key_c][2]-1)) ** 0.5) if delta_stats[key_c][2] > 1 else 0.0),
                            bool(ok_c),
                            int(len(issues_c)),
                            int(len(sol_cold.groups)),
                            muts,
                        ],
                    )

                # Baseline algorithms (cold start each step)
                for algo in base_algos:
                    t0 = perf_counter()
                    sol = algo.solve(Gs, lts, seed=RANDOM_SEED + step)
                    elapsed = (perf_counter() - t0) * 1000.0
                    ok, issues = validator.validate(sol, Gs)
                    key_b = (algo.name, False)
                    prev_cost_b = float(prev_solutions.get(key_b, sol).total_cost) if isinstance(prev_solutions.get(key_b), type(sol)) else float('nan')
                    delta_b = float(sol.total_cost) - (prev_cost_b if prev_cost_b == prev_cost_b else float('nan'))
                    tot_b, cnt_b = time_accum.get(key_b, (0.0, 0))
                    avg_b = (tot_b + elapsed) / (cnt_b + 1)
                    time_accum[key_b] = (tot_b + elapsed, cnt_b + 1)
                    mean_d, m2_d, k = delta_stats.get(key_b, (0.0, 0.0, 0))
                    x = float(delta_b) if delta_b == delta_b else 0.0
                    k1 = k + 1
                    d = x - mean_d
                    mean_new = mean_d + d / k1
                    m2_new = m2_d + d * (x - mean_new)
                    delta_stats[key_b] = (mean_new, m2_new, k1)
                    prev_solutions[key_b] = sol
                    print(
                        f"   base  {algo.name:<24s} cost={sol.total_cost:.2f} time_ms={elapsed:.2f} dCost={delta_b if delta_b==delta_b else 0.0:+.2f} valid={ok} groups={len(sol.groups)}"
                    )
                    _append_csv_row(
                        out_path,
                        [
                            run_id,
                            gname,
                            str(params),
                            lic,
                            algo.name,
                            False,
                            step,
                            Gs.number_of_nodes(),
                            Gs.number_of_edges(),
                            float(sol.total_cost),
                            float(elapsed),
                            float(delta_b if delta_b == delta_b else 0.0),
                            float(avg_b),
                            float(abs(delta_b) if delta_b == delta_b else 0.0),
                            float(((delta_stats[key_b][1] / max(1, delta_stats[key_b][2]-1)) ** 0.5) if delta_stats[key_b][2] > 1 else 0.0),
                            bool(ok),
                            int(len(issues)),
                            int(len(sol.groups)),
                            muts,
                        ],
                    )

    print(f"csv: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

```

### src/glopt/cli/dynamic_real.py

```python
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import networkx as nx

from glopt.core import instantiate_algorithms
from glopt.core.solution_validator import SolutionValidator
from glopt.dynamic_simulator import DynamicNetworkSimulator, MutationParams
from glopt.io import build_paths, ensure_dir
from glopt.io.data_loader import RealWorldDataLoader
from glopt.license_config import LicenseConfigFactory
from glopt.algorithms.greedy import GreedyAlgorithm
from glopt.core.solution_builder import SolutionBuilder


# ==============================================
# Configuration (edit here)
# ==============================================

RUN_ID: str | None = None

LICENSE_CONFIGS: list[str] = [
    "spotify",
    "duolingo_super",
    "roman_domination",
    # sweep p for roman domination
    "roman_p_1_5",
    "roman_p_2_0",
    "roman_p_2_5",
    "roman_p_3_0",
]

NUM_STEPS: int = 8
RANDOM_SEED: int = 123

WARM_ALGOS: list[str] = [
    "GeneticAlgorithm",
    "SimulatedAnnealing",
    "TabuSearch",
    "AntColonyOptimization",
]
BASELINE_ALGOS: list[str] = [
    "GreedyAlgorithm",
    "ILPSolver",
]


def _repair_solution_for_graph(prev_solution, graph: nx.Graph, license_types) -> Any:
    ok_groups = []
    nodes = set(graph.nodes())
    for g in prev_solution.groups:
        if g.owner not in nodes:
            continue
        allowed = set(graph.neighbors(g.owner)) | {g.owner}
        allm = set(g.all_members) & nodes
        if not allm:
            continue
        if not allm.issubset(allowed):
            continue
        try:
            new_g = type(g)(g.license_type, g.owner, frozenset(allm - {g.owner}))
        except Exception:
            continue
        ok_groups.append(new_g)
    covered = set().union(*(g.all_members for g in ok_groups)) if ok_groups else set()
    uncovered = set(graph.nodes()) - covered
    if uncovered:
        H = graph.subgraph(uncovered).copy()
        greedy = GreedyAlgorithm().solve(H, list(license_types))
        ok_groups.extend(greedy.groups)
    return SolutionBuilder.create_solution_from_groups(ok_groups)


def _write_csv_header(path: Path, header: list[str]) -> None:
    import csv

    ensure_dir(str(path.parent))
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)


def _append_csv_row(path: Path, row: list[Any]) -> None:
    import csv

    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def main() -> int:
    run_id = (RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")) + "_dynamic_real"
    _, _, csv_dir = build_paths(run_id)
    out_path = Path(csv_dir) / f"{run_id}.csv"

    print("== glopt dynamic benchmark (real graphs) ==")
    print(f"run_id: {run_id}")
    print(f"licenses: {', '.join(LICENSE_CONFIGS)}")
    print(f"warm algos: {', '.join(WARM_ALGOS)}")
    print(f"baselines: {', '.join(BASELINE_ALGOS)}")
    print(f"steps: {NUM_STEPS} seed={RANDOM_SEED}")

    header = [
        "run_id",
        "graph",
        "graph_params",
        "license_config",
        "algorithm",
        "warm_start",
        "step",
        "ego_id",
        "n_nodes",
        "n_edges",
        "total_cost",
        "time_ms",
        "delta_cost",
        "avg_time_ms_per_step",
        "delta_cost_abs",
        "delta_cost_std_so_far",
        "valid",
        "issues",
        "groups",
        "mutations",
    ]
    _write_csv_header(out_path, header)

    validator = SolutionValidator(debug=False)
    loader = RealWorldDataLoader(data_dir="data")
    networks = loader.load_all_facebook_networks()

    for ego_id, graph in networks.items():
        print(f"ego_id={ego_id} nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")

        # Scale mutation params to graph size
        N = graph.number_of_nodes() or 1
        E0 = graph.number_of_edges() or N
        mut_params = MutationParams(
            add_nodes_prob=0.05,
            remove_nodes_prob=0.03,
            add_edges_prob=0.15,
            remove_edges_prob=0.10,
            max_nodes_add=max(1, int(0.02 * N)),
            max_nodes_remove=max(1, int(0.015 * N)),
            max_edges_add=max(1, int(0.03 * E0)),
            max_edges_remove=max(1, int(0.02 * E0)),
        )
        sim = DynamicNetworkSimulator(mutation_params=mut_params, seed=RANDOM_SEED)
        sim.next_node_id = max(graph.nodes()) + 1 if graph.nodes() else 0

        # Precompute mutation sequence
        graphs: list[nx.Graph] = [graph.copy()]
        mutations_per_step: list[list[str]] = [[]]
        g_curr = graph.copy()
        for _ in range(NUM_STEPS):
            g_curr, muts = sim._apply_mutations(g_curr)  # type: ignore[attr-defined]
            graphs.append(g_curr.copy())
            mutations_per_step.append(muts)

        for lic in LICENSE_CONFIGS:
            lts = LicenseConfigFactory.get_config(lic)
            warm_algos = instantiate_algorithms(WARM_ALGOS)
            base_algos = instantiate_algorithms(BASELINE_ALGOS)

            prev_solutions: dict[tuple[str, bool], Any] = {}
            time_accum: dict[tuple[str, bool], tuple[float, int]] = {}
            delta_stats: dict[tuple[str, bool], tuple[float, float, int]] = {}

            # Step 0 init
            G0 = graphs[0]
            for algo in warm_algos + base_algos:
                t0 = perf_counter()
                sol0 = algo.solve(G0, lts, seed=RANDOM_SEED)
                elapsed = (perf_counter() - t0) * 1000.0
                ok, issues = validator.validate(sol0, G0)
                key = (algo.name, False)
                prev_solutions[key] = sol0
                time_accum[key] = (elapsed, 1)
                delta_stats[key] = (0.0, 0.0, 0)
                print(
                    f"init -> ego={ego_id} / {lic} / {algo.name:<24s} cold   cost={sol0.total_cost:.2f} time_ms={elapsed:.2f} valid={ok} groups={len(sol0.groups)}"
                )
                _append_csv_row(
                    out_path,
                    [
                        run_id,
                        "facebook_ego",
                        str({"ego_id": ego_id}),
                        lic,
                        algo.name,
                        False,
                        0,
                        ego_id,
                        G0.number_of_nodes(),
                        G0.number_of_edges(),
                        float(sol0.total_cost),
                        float(elapsed),
                        0.0,
                        float(elapsed),
                        0.0,
                        0.0,
                        bool(ok),
                        int(len(issues)),
                        int(len(sol0.groups)),
                        "; ".join(mutations_per_step[0]),
                    ],
                )

            # Steps 1..NUM_STEPS
            for step in range(1, NUM_STEPS + 1):
                Gs = graphs[step]
                muts = "; ".join(mutations_per_step[step])
                print(
                    f"step {step:02d} -> ego={ego_id} lic={lic} nodes={Gs.number_of_nodes()} edges={Gs.number_of_edges()} mutations=[{muts}]"
                )

                # Warm-start algos: warm and cold variants
                for algo in warm_algos:
                    prev_warm = prev_solutions.get((algo.name, True)) or prev_solutions.get((algo.name, False))
                    warm = _repair_solution_for_graph(prev_warm, Gs, lts) if prev_warm is not None else None

                    # warm run
                    t0 = perf_counter()
                    sol = algo.solve(Gs, lts, seed=RANDOM_SEED + step, initial_solution=warm)
                    elapsed = (perf_counter() - t0) * 1000.0
                    ok, issues = validator.validate(sol, Gs)
                    key_w = (algo.name, True)
                    prev_cost = float(prev_solutions.get(key_w, sol).total_cost) if isinstance(prev_solutions.get(key_w), type(sol)) else (
                        float(prev_solutions.get((algo.name, False), sol).total_cost) if isinstance(prev_solutions.get((algo.name, False)), type(sol)) else float('nan')
                    )
                    delta = float(sol.total_cost) - (prev_cost if prev_cost == prev_cost else float('nan'))
                    tot, cnt = time_accum.get(key_w, (0.0, 0))
                    avg = (tot + elapsed) / (cnt + 1)
                    time_accum[key_w] = (tot + elapsed, cnt + 1)
                    mean_d, m2_d, k = delta_stats.get(key_w, (0.0, 0.0, 0))
                    x = float(delta) if delta == delta else 0.0
                    k1 = k + 1
                    dtmp = x - mean_d
                    mean_new = mean_d + dtmp / k1
                    m2_new = m2_d + dtmp * (x - mean_new)
                    delta_stats[key_w] = (mean_new, m2_new, k1)
                    prev_solutions[key_w] = sol
                    print(
                        f"   warm  {algo.name:<24s} cost={sol.total_cost:.2f} time_ms={elapsed:.2f} dCost={delta if delta==delta else 0.0:+.2f} valid={ok} groups={len(sol.groups)}"
                    )
                    _append_csv_row(
                        out_path,
                        [
                            run_id,
                            "facebook_ego",
                            str({"ego_id": ego_id}),
                            lic,
                            algo.name,
                            True,
                            step,
                            ego_id,
                            Gs.number_of_nodes(),
                            Gs.number_of_edges(),
                            float(sol.total_cost),
                            float(elapsed),
                            float(delta if delta == delta else 0.0),
                            float(avg),
                            float(abs(delta) if delta == delta else 0.0),
                            float(((delta_stats[key_w][1] / max(1, delta_stats[key_w][2]-1)) ** 0.5) if delta_stats[key_w][2] > 1 else 0.0),
                            bool(ok),
                            int(len(issues)),
                            int(len(sol.groups)),
                            muts,
                        ],
                    )

                    # cold run
                    t0 = perf_counter()
                    sol_cold = algo.solve(Gs, lts, seed=RANDOM_SEED + step)
                    elapsed_c = (perf_counter() - t0) * 1000.0
                    ok_c, issues_c = validator.validate(sol_cold, Gs)
                    key_c = (algo.name, False)
                    prev_cost_c = float(prev_solutions.get(key_c, sol_cold).total_cost) if isinstance(prev_solutions.get(key_c), type(sol_cold)) else float('nan')
                    delta_c = float(sol_cold.total_cost) - (prev_cost_c if prev_cost_c == prev_cost_c else float('nan'))
                    tot_c, cnt_c = time_accum.get(key_c, (0.0, 0))
                    avg_c = (tot_c + elapsed_c) / (cnt_c + 1)
                    time_accum[key_c] = (tot_c + elapsed_c, cnt_c + 1)
                    mean_d, m2_d, k = delta_stats.get(key_c, (0.0, 0.0, 0))
                    x = float(delta_c) if delta_c == delta_c else 0.0
                    k1 = k + 1
                    dtmp = x - mean_d
                    mean_new = mean_d + dtmp / k1
                    m2_new = m2_d + dtmp * (x - mean_new)
                    delta_stats[key_c] = (mean_new, m2_new, k1)
                    prev_solutions[key_c] = sol_cold
                    print(
                        f"   cold  {algo.name:<24s} cost={sol_cold.total_cost:.2f} time_ms={elapsed_c:.2f} dCost={delta_c if delta_c==delta_c else 0.0:+.2f} valid={ok_c} groups={len(sol_cold.groups)}"
                    )
                    _append_csv_row(
                        out_path,
                        [
                            run_id,
                            "facebook_ego",
                            str({"ego_id": ego_id}),
                            lic,
                            algo.name,
                            False,
                            step,
                            ego_id,
                            Gs.number_of_nodes(),
                            Gs.number_of_edges(),
                            float(sol_cold.total_cost),
                            float(elapsed_c),
                            float(delta_c if delta_c == delta_c else 0.0),
                            float(avg_c),
                            float(abs(delta_c) if delta_c == delta_c else 0.0),
                            float(((delta_stats[key_c][1] / max(1, delta_stats[key_c][2]-1)) ** 0.5) if delta_stats[key_c][2] > 1 else 0.0),
                            bool(ok_c),
                            int(len(issues_c)),
                            int(len(sol_cold.groups)),
                            muts,
                        ],
                    )

                # Baseline algos (cold only)
                for algo in base_algos:
                    t0 = perf_counter()
                    sol = algo.solve(Gs, lts, seed=RANDOM_SEED + step)
                    elapsed = (perf_counter() - t0) * 1000.0
                    ok, issues = validator.validate(sol, Gs)
                    key_b = (algo.name, False)
                    prev_cost_b = float(prev_solutions.get(key_b, sol).total_cost) if isinstance(prev_solutions.get(key_b), type(sol)) else float('nan')
                    delta_b = float(sol.total_cost) - (prev_cost_b if prev_cost_b == prev_cost_b else float('nan'))
                    tot_b, cnt_b = time_accum.get(key_b, (0.0, 0))
                    avg_b = (tot_b + elapsed) / (cnt_b + 1)
                    time_accum[key_b] = (tot_b + elapsed, cnt_b + 1)
                    mean_d, m2_d, k = delta_stats.get(key_b, (0.0, 0.0, 0))
                    x = float(delta_b) if delta_b == delta_b else 0.0
                    k1 = k + 1
                    dtmp = x - mean_d
                    mean_new = mean_d + dtmp / k1
                    m2_new = m2_d + dtmp * (x - mean_new)
                    delta_stats[key_b] = (mean_new, m2_new, k1)
                    prev_solutions[key_b] = sol
                    print(
                        f"   base  {algo.name:<24s} cost={sol.total_cost:.2f} time_ms={elapsed:.2f} dCost={delta_b if delta_b==delta_b else 0.0:+.2f} valid={ok} groups={len(sol.groups)}"
                    )
                    _append_csv_row(
                        out_path,
                        [
                            run_id,
                            "facebook_ego",
                            str({"ego_id": ego_id}),
                            lic,
                            algo.name,
                            False,
                            step,
                            ego_id,
                            Gs.number_of_nodes(),
                            Gs.number_of_edges(),
                            float(sol.total_cost),
                            float(elapsed),
                            float(delta_b if delta_b == delta_b else 0.0),
                            float(avg_b),
                            float(abs(delta_b) if delta_b == delta_b else 0.0),
                            float(((delta_stats[key_b][1] / max(1, delta_stats[key_b][2]-1)) ** 0.5) if delta_stats[key_b][2] > 1 else 0.0),
                            bool(ok),
                            int(len(issues)),
                            int(len(sol.groups)),
                            muts,
                        ],
                    )

    print(f"csv: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


```

### src/glopt/core/__init__.py

```python
from .models import Algorithm, LicenseGroup, LicenseType, Solution
from .mutations import MutationOperators
from .run import RunResult, generate_graph, instantiate_algorithms, run_once
from .solution_builder import SolutionBuilder
from .solution_validator import SolutionValidator

__all__ = [
    "Algorithm",
    "LicenseGroup",
    "LicenseType",
    "MutationOperators",
    "RunResult",
    "Solution",
    "SolutionBuilder",
    "SolutionValidator",
    "generate_graph",
    "instantiate_algorithms",
    "run_once",
]

```

### src/glopt/core/models.py

```python
from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import networkx as nx

N = TypeVar("N", bound=Hashable)


@dataclass(frozen=True, slots=True)
class LicenseType:
    name: str
    cost: float
    min_capacity: int
    max_capacity: int
    color: str = "#000000"

    def __post_init__(self) -> None:
        if self.cost < 0:
            msg = "cost must be >= 0"
            raise ValueError(msg)
        if self.min_capacity < 1:
            msg = "min_capacity must be >= 1"
            raise ValueError(msg)
        if self.max_capacity < self.min_capacity:
            msg = "max_capacity must be >= min_capacity"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class LicenseGroup[N: Hashable]:
    license_type: LicenseType
    owner: N
    additional_members: frozenset[N] = frozenset()

    @property
    def all_members(self) -> frozenset[N]:
        if self.owner in self.additional_members:
            return self.additional_members
        return frozenset({self.owner, *self.additional_members})

    @property
    def size(self) -> int:
        return len(self.all_members)

    def __post_init__(self) -> None:
        s = self.size
        if not (self.license_type.min_capacity <= s <= self.license_type.max_capacity):
            msg = f"group size {s} violates [{self.license_type.min_capacity}, {self.license_type.max_capacity}] for {self.license_type.name}"
            raise ValueError(msg)


@dataclass(slots=True)
class Solution[N: Hashable]:
    groups: tuple[LicenseGroup[N], ...] = ()

    @property
    def total_cost(self) -> float:
        return sum(g.license_type.cost for g in self.groups)


class Algorithm[N: Hashable](ABC):
    @abstractmethod
    def solve(self, graph: nx.Graph, license_types: Sequence[LicenseType], **kwargs: Any) -> Solution[N]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

```

### src/glopt/core/mutations.py

```python
from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .models import LicenseGroup, LicenseType, Solution
from .solution_builder import SolutionBuilder

if TYPE_CHECKING:
    from collections.abc import Sequence

    import networkx as nx


class MutationOperators:
    @staticmethod
    def generate_neighbors(
        base: Solution,
        graph: nx.Graph,
        license_types: Sequence[LicenseType],
        k: int = 10,
    ) -> list[Solution]:
        ops = (
            MutationOperators.change_license_type,
            MutationOperators.reassign_member,
            MutationOperators.merge_groups,
            MutationOperators.split_group,
        )
        weights = (0.30, 0.30, 0.20, 0.20)
        out: list[Solution] = []
        attempts = 0
        while len(out) < k and attempts < k * 10:
            attempts += 1
            op = random.choices(ops, weights=weights, k=1)[0]
            try:
                cand = op(base, graph, list(license_types))
            except Exception:
                cand = None
            if cand is not None:
                out.append(cand)
        return out

    @staticmethod
    def change_license_type(
        solution: Solution,
        graph: nx.Graph,
        license_types: list[LicenseType],
    ) -> Solution | None:
        if not solution.groups:
            return None
        group = random.choice(solution.groups)
        compatible = SolutionBuilder.get_compatible_license_types(group.size, license_types, exclude=group.license_type)
        if not compatible:
            return None
        new_lt = random.choice(compatible)

        new_groups = []
        for g in solution.groups:
            if g is group:
                new_groups.append(LicenseGroup(new_lt, g.owner, g.additional_members))
            else:
                new_groups.append(g)
        return SolutionBuilder.create_solution_from_groups(new_groups)

    @staticmethod
    def reassign_member(
        solution: Solution,
        graph: nx.Graph,
        license_types: list[LicenseType],
    ) -> Solution | None:
        if len(solution.groups) < 2:
            return None

        donors = [g for g in solution.groups if g.size > g.license_type.min_capacity and g.additional_members]
        receivers = [g for g in solution.groups if g.size < g.license_type.max_capacity]
        if not donors or not receivers:
            return None

        from_group = random.choice(donors)
        pot_receivers = [g for g in receivers if g is not from_group]
        if not pot_receivers:
            return None
        to_group = random.choice(pot_receivers)

        member = random.choice(list(from_group.additional_members))
        allowed = SolutionBuilder.get_owner_neighbors_with_self(graph, to_group.owner)
        if member not in allowed:
            return None

        new_groups = []
        for g in solution.groups:
            if g is from_group:
                new_groups.append(LicenseGroup(g.license_type, g.owner, g.additional_members - {member}))
            elif g is to_group:
                new_groups.append(LicenseGroup(g.license_type, g.owner, g.additional_members | {member}))
            else:
                new_groups.append(g)
        return SolutionBuilder.create_solution_from_groups(new_groups)

    @staticmethod
    def merge_groups(
        solution: Solution,
        graph: nx.Graph,
        license_types: list[LicenseType],
    ) -> Solution | None:
        if len(solution.groups) < 2:
            return None
        g1, g2 = random.sample(list(solution.groups), 2)
        merged = SolutionBuilder.merge_groups(g1, g2, graph, license_types)
        if merged is None:
            return None

        new_groups = [g for g in solution.groups if g not in (g1, g2)]
        new_groups.append(merged)
        return SolutionBuilder.create_solution_from_groups(new_groups)

    @staticmethod
    def split_group(
        solution: Solution,
        graph: nx.Graph,
        license_types: list[LicenseType],
    ) -> Solution | None:
        if not solution.groups:
            return None

        splittable = [g for g in solution.groups if g.size > 2]
        if not splittable:
            return None

        group = random.choice(splittable)
        members = list(group.all_members)

        for _ in range(4):
            random.shuffle(members)
            cut = random.randint(1, len(members) - 1)
            part1, part2 = members[:cut], members[cut:]

            compat1 = SolutionBuilder.get_compatible_license_types(len(part1), license_types)
            compat2 = SolutionBuilder.get_compatible_license_types(len(part2), license_types)
            if not compat1 or not compat2:
                continue

            owner1 = random.choice(part1)
            owner2 = random.choice(part2)

            neigh1 = SolutionBuilder.get_owner_neighbors_with_self(graph, owner1)
            neigh2 = SolutionBuilder.get_owner_neighbors_with_self(graph, owner2)
            if not set(part1).issubset(neigh1) or not set(part2).issubset(neigh2):
                continue

            lt1 = random.choice(compat1)
            lt2 = random.choice(compat2)

            g1 = LicenseGroup(lt1, owner1, frozenset(set(part1) - {owner1}))
            g2 = LicenseGroup(lt2, owner2, frozenset(set(part2) - {owner2}))

            new_groups = [g for g in solution.groups if g is not group]
            new_groups.extend([g1, g2])
            return SolutionBuilder.create_solution_from_groups(new_groups)

        return None

```

### src/glopt/core/run.py

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import networkx as nx

from glopt import algorithms
from glopt.core.solution_validator import SolutionValidator
from glopt.io.graph_generator import GraphGeneratorFactory
from glopt.io.graph_visualizer import GraphVisualizer

if TYPE_CHECKING:
    from glopt.core.models import Algorithm, LicenseType, Solution


@dataclass(frozen=True)
class RunResult:
    run_id: str
    algorithm: str
    graph: str
    n_nodes: int
    n_edges: int
    graph_params: str
    license_config: str
    total_cost: float
    time_ms: float
    valid: bool
    issues: int
    image_path: str
    notes: str = ""


def generate_graph(name: str, n_nodes: int, params: dict[str, Any]) -> nx.Graph:
    gen = GraphGeneratorFactory.get(name)
    g = gen(n_nodes=n_nodes, **params)
    if not all(isinstance(v, int) for v in g.nodes()):
        mapping = {v: i for i, v in enumerate(g.nodes())}
        g = nx.relabel_nodes(g, mapping, copy=True)
    return g


def instantiate_algorithms(names: list[str]) -> list[Algorithm]:
    loaded: list[Algorithm] = []
    missing: list[str] = []
    for name in names:
        cls = getattr(algorithms, name, None)
        if cls is None:
            missing.append(name)
        else:
            loaded.append(cls())
    if missing:
        avail = ", ".join(getattr(algorithms, "__all__", []))
        msg = f"unknown algorithms: {', '.join(missing)}; available: {avail}"
        raise ValueError(msg)
    if not loaded:
        msg = "no algorithms selected"
        raise ValueError(msg)
    return loaded


def run_once(
    algo: Algorithm,
    graph: nx.Graph,
    license_types: list[LicenseType],
    run_id: str,
    graphs_dir: str,
    print_issue_limit: int | None = 20,
) -> RunResult:
    validator = SolutionValidator(debug=False)
    visualizer = GraphVisualizer(figsize=(12, 8))

    t0 = perf_counter()
    solution: Solution = algo.solve(graph=graph, license_types=license_types)
    elapsed_ms = (perf_counter() - t0) * 1000.0

    ok, issues = validator.validate(solution, graph)
    # Keep validation but avoid noisy prints; leave issues count in the result.

    img_name = f"{algo.name}_{graph.number_of_nodes()}n_{graph.number_of_edges()}e.png"
    img_path = str(Path(graphs_dir) / img_name)
    try:
        visualizer.visualize_solution(
            graph=graph,
            solution=solution,
            solver_name=algo.name,
            timestamp_folder=run_id,
            save_path=img_path,
        )
    except Exception:  # defensive: visualization may fail in headless environments
        img_path = ""

    return RunResult(
        run_id=run_id,
        algorithm=algo.name,
        graph="?",
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        graph_params="{}",
        license_config="?",
        total_cost=float(solution.total_cost),
        time_ms=elapsed_ms,
        valid=ok,
        issues=len(issues),
        image_path=img_path,
        notes="",
    )

```

### src/glopt/core/solution_builder.py

```python
from collections.abc import Hashable, Sequence

import networkx as nx

from .models import LicenseGroup, LicenseType, Solution

N = Hashable


class SolutionBuilder:
    @staticmethod
    def create_solution_from_groups(groups: list[LicenseGroup]) -> Solution:
        return Solution(groups=tuple(groups))

    @staticmethod
    def get_compatible_license_types(
        group_size: int,
        license_types: Sequence[LicenseType],
        exclude: LicenseType | None = None,
    ) -> list[LicenseType]:
        out: list[LicenseType] = []
        for lt in license_types:
            if exclude and lt == exclude:
                continue
            if lt.min_capacity <= group_size <= lt.max_capacity:
                out.append(lt)
        return out

    @staticmethod
    def get_owner_neighbors_with_self(graph: nx.Graph, owner: N) -> set[N]:
        return set(graph.neighbors(owner)) | {owner}

    @staticmethod
    def merge_groups(
        group1: LicenseGroup,
        group2: LicenseGroup,
        graph: nx.Graph,
        license_types: Sequence[LicenseType],
    ) -> LicenseGroup | None:
        members = group1.all_members | group2.all_members
        size = len(members)

        for lt in license_types:
            if lt.min_capacity <= size <= lt.max_capacity:
                for owner in members:
                    neigh = SolutionBuilder.get_owner_neighbors_with_self(graph, owner)
                    if members.issubset(neigh):
                        return LicenseGroup(lt, owner, frozenset(members - {owner}))
        return None

    @staticmethod
    def find_cheapest_single_license(license_types: Sequence[LicenseType]) -> LicenseType:
        singles = [lt for lt in license_types if lt.min_capacity <= 1]
        return min(singles or list(license_types), key=lambda lt: lt.cost)

    @staticmethod
    def find_cheapest_license_for_size(size: int, license_types: Sequence[LicenseType]) -> LicenseType | None:
        compat = [lt for lt in license_types if lt.min_capacity <= size <= lt.max_capacity]
        return min(compat, key=lambda lt: lt.cost) if compat else None

```

### src/glopt/core/solution_validator.py

```python
from collections.abc import Hashable
from dataclasses import dataclass
from typing import TypeVar

import networkx as nx

from .models import LicenseGroup, Solution

N = TypeVar("N", bound=Hashable)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    msg: str


class SolutionValidator:
    def __init__(self, debug: bool = False) -> None:
        self.debug = debug

    def validate(
        self,
        solution: Solution[N],
        graph: nx.Graph,
        all_nodes: set[N] | None = None,
    ) -> tuple[bool, list[ValidationIssue]]:
        issues: list[ValidationIssue] = []
        nodes: set[N] = set(graph.nodes()) if all_nodes is None else set(all_nodes)
        groups: tuple[LicenseGroup[N], ...] = tuple(solution.groups)

        issues += self._check_group_members(groups, nodes)
        issues += self._check_group_capacity(groups)
        issues += self._check_neighbors(groups, graph, nodes)
        issues += self._check_no_overlap(groups)
        issues += self._check_coverage(groups, nodes)

        if self.debug and issues:
            for _i in issues:
                pass

        return (not issues, issues)

    def is_valid_solution(
        self,
        solution: Solution[N],
        graph: nx.Graph,
        all_nodes: set[N] | None = None,
    ) -> bool:
        ok, _ = self.validate(solution, graph, all_nodes)
        return ok

    def _check_group_members(self, groups: tuple[LicenseGroup[N], ...], nodes: set[N]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for idx, g in enumerate(groups):
            outside = g.all_members - nodes
            if outside:
                issues.append(
                    ValidationIssue(
                        "MEMBER_NOT_IN_GRAPH",
                        f"group#{idx} owner {g.owner!r} has members not in graph: {sorted(outside)!r}",
                    )
                )
            if g.owner not in g.all_members:
                issues.append(ValidationIssue("OWNER_NOT_IN_GROUP", f"group#{idx} owner {g.owner!r} not included in its members"))
        return issues

    def _check_group_capacity(self, groups: tuple[LicenseGroup[N], ...]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for idx, g in enumerate(groups):
            mn, mx, sz = g.license_type.min_capacity, g.license_type.max_capacity, g.size
            if not (mn <= sz <= mx):
                issues.append(
                    ValidationIssue(
                        "CAPACITY_VIOLATION",
                        f"group#{idx} owner {g.owner!r} size={sz} not in [{mn}, {mx}] for license '{g.license_type.name}'",
                    ),
                )
        return issues

    def _check_neighbors(self, groups: tuple[LicenseGroup[N], ...], graph: nx.Graph, nodes: set[N]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for idx, g in enumerate(groups):
            if g.owner not in nodes:
                issues.append(ValidationIssue("OWNER_NOT_IN_GRAPH", f"group#{idx} owner {g.owner!r} not in graph"))
                continue
            allowed = set(graph.neighbors(g.owner)) | {g.owner}
            not_neighbors = g.all_members - allowed
            if not_neighbors:
                issues.append(
                    ValidationIssue(
                        "DISCONNECTED_MEMBER",
                        f"group#{idx} owner {g.owner!r} has non-neighbor members: {sorted(not_neighbors)!r}",
                    )
                )
        return issues

    def _check_no_overlap(self, groups: tuple[LicenseGroup[N], ...]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        seen: set[N] = set()
        for idx, g in enumerate(groups):
            overlap = seen & g.all_members
            if overlap:
                issues.append(ValidationIssue("OVERLAP", f"group#{idx} owner {g.owner!r} overlaps members {sorted(overlap)!r}"))
            seen.update(g.all_members)
        return issues

    def _check_coverage(self, groups: tuple[LicenseGroup[N], ...], nodes: set[N]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        covered = set().union(*(g.all_members for g in groups)) if groups else set()
        missing = nodes - covered
        extra = covered - nodes
        if missing:
            issues.append(ValidationIssue("MISSING_COVERAGE", f"missing nodes: {sorted(missing)!r}"))
        if extra:
            issues.append(ValidationIssue("EXTRA_COVERAGE", f"extra nodes not in graph: {sorted(extra)!r}"))
        return issues

```

### src/glopt/dynamic_simulator.py

```python
import csv
import random
from dataclasses import dataclass
from typing import Any

import networkx as nx

from .algorithms import GreedyAlgorithm
from .core import Algorithm, LicenseGroup, LicenseType, Solution
from .core.solution_builder import SolutionBuilder


@dataclass
class MutationParams:
    add_nodes_prob: float = 0.1
    remove_nodes_prob: float = 0.05
    add_edges_prob: float = 0.15
    remove_edges_prob: float = 0.1
    max_nodes_add: int = 3
    max_nodes_remove: int = 2
    max_edges_add: int = 5
    max_edges_remove: int = 3


@dataclass
class DynamicStep:
    step_number: int
    graph: nx.Graph
    solution: Solution
    mutations_applied: list[str]
    rebalance_cost_change: float


class DynamicNetworkSimulator:
    def __init__(
        self,
        rebalance_algorithm: Algorithm | None = None,
        mutation_params: MutationParams | None = None,
        seed: int | None = None,
    ) -> None:
        self.rebalance_algorithm = rebalance_algorithm or GreedyAlgorithm()
        self.mutation_params = mutation_params or MutationParams()
        self.seed = seed
        self.history: list[DynamicStep] = []
        self.next_node_id = 0

        if seed is not None:
            random.seed(seed)

    def simulate(
        self,
        initial_graph: nx.Graph,
        license_types: list[LicenseType],
        num_steps: int = 10,
        initial_algorithm: Algorithm | None = None,
    ) -> list[DynamicStep]:
        self.history.clear()
        current_graph = initial_graph.copy()
        self.next_node_id = max(current_graph.nodes()) + 1 if current_graph.nodes() else 0

        if initial_algorithm is None:
            initial_algorithm = self.rebalance_algorithm

        current_solution = initial_algorithm.solve(current_graph, license_types)

        self.history.append(
            DynamicStep(
                step_number=0,
                graph=current_graph.copy(),
                solution=current_solution,
                mutations_applied=[],
                rebalance_cost_change=0.0,
            )
        )

        for step in range(1, num_steps + 1):
            mutations_applied = []

            current_graph, step_mutations = self._apply_mutations(current_graph)
            mutations_applied.extend(step_mutations)

            old_cost = current_solution.total_cost
            current_solution = self._rebalance_licenses(current_graph, license_types, current_solution)
            new_cost = current_solution.total_cost
            cost_change = new_cost - old_cost

            self.history.append(
                DynamicStep(
                    step_number=step,
                    graph=current_graph.copy(),
                    solution=current_solution,
                    mutations_applied=mutations_applied,
                    rebalance_cost_change=cost_change,
                ),
            )

        return self.history

    def _apply_mutations(self, graph: nx.Graph) -> tuple[nx.Graph, list[str]]:
        mutations = []

        if random.random() < self.mutation_params.add_nodes_prob:
            num_add = random.randint(1, self.mutation_params.max_nodes_add)
            new_nodes = self._add_nodes(graph, num_add)
            mutations.append(f"Added nodes: {new_nodes}")

        if random.random() < self.mutation_params.remove_nodes_prob and len(graph.nodes()) > 5:
            num_remove = random.randint(1, min(self.mutation_params.max_nodes_remove, len(graph.nodes()) - 5))
            removed_nodes = self._remove_nodes(graph, num_remove)
            mutations.append(f"Removed nodes: {removed_nodes}")

        if random.random() < self.mutation_params.add_edges_prob:
            num_add = random.randint(1, self.mutation_params.max_edges_add)
            added_edges = self._add_edges(graph, num_add)
            mutations.append(f"Added {len(added_edges)} edges")

        if random.random() < self.mutation_params.remove_edges_prob and len(graph.edges()) > 0:
            num_remove = random.randint(1, min(self.mutation_params.max_edges_remove, len(graph.edges())))
            removed_edges = self._remove_edges(graph, num_remove)
            mutations.append(f"Removed {len(removed_edges)} edges")

        return graph, mutations

    def _add_nodes(self, graph: nx.Graph, num_nodes: int) -> list[int]:
        new_nodes = []
        existing_nodes = list(graph.nodes())

        for _ in range(num_nodes):
            new_node = self.next_node_id
            self.next_node_id += 1
            graph.add_node(new_node)
            new_nodes.append(new_node)

            if existing_nodes:
                num_connections = random.randint(1, min(3, len(existing_nodes)))
                neighbors = random.sample(existing_nodes, num_connections)
                for neighbor in neighbors:
                    graph.add_edge(new_node, neighbor)

        return new_nodes

    def _remove_nodes(self, graph: nx.Graph, num_nodes: int) -> list[int]:
        nodes_to_remove = random.sample(list(graph.nodes()), num_nodes)

        for node in nodes_to_remove:
            graph.remove_node(node)

        return nodes_to_remove

    def _add_edges(self, graph: nx.Graph, num_edges: int) -> list[tuple[int, int]]:
        nodes = list(graph.nodes())
        added_edges = []

        if len(nodes) < 2:
            return added_edges

        attempts = 0
        while len(added_edges) < num_edges and attempts < num_edges * 10:
            node1, node2 = random.sample(nodes, 2)
            if not graph.has_edge(node1, node2):
                graph.add_edge(node1, node2)
                added_edges.append((node1, node2))
            attempts += 1

        return added_edges

    def _remove_edges(self, graph: nx.Graph, num_edges: int) -> list[tuple[int, int]]:
        edges_to_remove = random.sample(list(graph.edges()), num_edges)

        for edge in edges_to_remove:
            graph.remove_edge(*edge)

        return edges_to_remove

    def _rebalance_licenses(self, graph: nx.Graph, license_types: list[LicenseType], old_solution: Solution) -> Solution:
        existing_nodes = set(graph.nodes())

        valid_groups = []
        uncovered_nodes = set(existing_nodes)

        for group in old_solution.groups:
            group_nodes = group.all_members

            if not group_nodes.issubset(existing_nodes):
                continue

            if self._is_group_valid(graph, group):
                valid_groups.append(group)
                uncovered_nodes -= group_nodes

        if uncovered_nodes:
            subgraph = graph.subgraph(uncovered_nodes).copy()

            if len(subgraph.nodes()) > 0:
                new_solution = self.rebalance_algorithm.solve(subgraph, license_types)
                valid_groups.extend(new_solution.groups)

        return SolutionBuilder.create_solution_from_groups(valid_groups)

    def _is_group_valid(self, graph: nx.Graph, group: LicenseGroup) -> bool:
        owner = group.owner
        additional_members = group.additional_members

        if owner not in graph.nodes():
            return False

        for member in additional_members:
            if member not in graph.nodes():
                return False
            if not graph.has_edge(owner, member):
                return False

        group_size = group.size
        license_type = group.license_type

        return license_type.min_capacity <= group_size <= license_type.max_capacity

    def get_simulation_summary(self) -> dict[str, Any]:
        if not self.history:
            return {}

        total_cost_changes = sum(step.rebalance_cost_change for step in self.history[1:])
        initial_cost = self.history[0].solution.total_cost
        final_cost = self.history[-1].solution.total_cost

        node_changes = []
        edge_changes = []

        for step in self.history[1:]:
            for mutation in step.mutations_applied:
                if "nodes" in mutation:
                    node_changes.append(mutation)
                elif "edges" in mutation:
                    edge_changes.append(mutation)

        return {
            "initial_cost": initial_cost,
            "final_cost": final_cost,
            "total_cost_change": total_cost_changes,
            "cost_efficiency": final_cost / initial_cost if initial_cost > 0 else 1.0,
            "num_steps": len(self.history) - 1,
            "node_mutations": len(node_changes),
            "edge_mutations": len(edge_changes),
            "avg_cost_change_per_step": total_cost_changes / max(1, len(self.history) - 1),
        }

    def export_history_to_csv(self, filename: str) -> None:
        fieldnames = ["step", "nodes", "edges", "cost", "groups", "cost_change", "mutations"]
        from pathlib import Path

        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for step in self.history:
                writer.writerow(
                    {
                        "step": step.step_number,
                        "nodes": len(step.graph.nodes()),
                        "edges": len(step.graph.edges()),
                        "cost": step.solution.total_cost,
                        "groups": len(step.solution.groups),
                        "cost_change": step.rebalance_cost_change,
                        "mutations": "; ".join(step.mutations_applied),
                    }
                )

    # DynamicScenarioFactory was unused; removed

```

### src/glopt/io/__init__.py

```python
from .csv_writer import BenchmarkCSVWriter, write_csv
from .data_loader import RealWorldDataLoader
from .fs import build_paths, ensure_dir
from .graph_generator import GraphGeneratorFactory
from .graph_visualizer import GraphVisualizer

__all__ = [
    "BenchmarkCSVWriter",
    "GraphGeneratorFactory",
    "GraphVisualizer",
    "RealWorldDataLoader",
    "build_paths",
    "ensure_dir",
    "write_csv",
]

```

### src/glopt/io/csv_writer.py

```python
import csv
import pathlib
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


def write_csv(csv_dir: str, run_id: str, rows: Iterable[Any]) -> str:
    out_path = Path(csv_dir) / f"{run_id}.csv"
    first = True
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = None
        for r in rows:
            d = asdict(r)
            if first:
                writer = csv.DictWriter(f, fieldnames=list(d.keys()))
                writer.writeheader()
                first = False
            writer.writerow(d)
    return str(out_path)


class BenchmarkCSVWriter:
    def __init__(self, output_dir: str = "runs/stats") -> None:
        self.output_dir = output_dir
        pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = str(Path(output_dir) / f"{timestamp}.csv")
        self.fieldnames = [
            "algorithm",
            "graph_type",
            "nodes",
            "edges",
            "graph_k",
            "graph_p",
            "graph_m",
            "license_config",
            "cost",
            "execution_time",
            "groups_count",
            "avg_degree",
            "seed",
        ]
        with Path(self.csv_path).open("w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

    # write_result and get_csv_path were unused; removed

```

### src/glopt/io/data_loader.py

```python
import logging
from pathlib import Path
from typing import Any

import networkx as nx


class RealWorldDataLoader:
    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)

    def load_facebook_ego_network(self, ego_id: str) -> nx.Graph:
        facebook_dir = self.data_dir / "facebook"

        edges_file = facebook_dir / f"{ego_id}.edges"
        if not edges_file.exists():
            msg = f"Plik edges nie istnieje: {edges_file}"
            raise FileNotFoundError(msg)

        graph = nx.Graph()

        ego_node = int(ego_id)
        graph.add_node(ego_node, is_ego=True)

        with edges_file.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        node1, node2 = int(parts[0]), int(parts[1])
                        graph.add_edge(node1, node2)

                        graph.add_edge(ego_node, node1)
                        graph.add_edge(ego_node, node2)

        self._load_node_features(graph, facebook_dir, ego_id)

        self._load_circles(graph, facebook_dir, ego_id)

        self.logger.info(f"Załadowano Facebook ego network {ego_id}: {len(graph.nodes())} węzłów, {len(graph.edges())} krawędzi")

        return graph

    def load_all_facebook_networks(self) -> dict[str, nx.Graph]:
        facebook_dir = self.data_dir / "facebook"
        networks = {}

        if not facebook_dir.exists():
            msg = f"Katalog Facebook nie istnieje: {facebook_dir}"
            raise FileNotFoundError(msg)

        edge_files = list(facebook_dir.glob("*.edges"))

        for edge_file in edge_files:
            ego_id = edge_file.stem
            try:
                network = self.load_facebook_ego_network(ego_id)
                networks[ego_id] = network
            except Exception:
                self.logger.warning("Nie udało się załadować network %s", ego_id)

        self.logger.info("Załadowano %d Facebook ego networks", len(networks))
        return networks

    def get_facebook_network_stats(self) -> dict[str, dict[str, Any]]:
        networks = self.load_all_facebook_networks()
        stats = {}

        for ego_id, graph in networks.items():
            stats[ego_id] = {
                "nodes": len(graph.nodes()),
                "edges": len(graph.edges()),
                "density": nx.density(graph),
                "avg_clustering": nx.average_clustering(graph),
                "is_connected": nx.is_connected(graph),
                "components": nx.number_connected_components(graph),
                "avg_degree": sum(dict(graph.degree()).values()) / len(graph.nodes()) if len(graph.nodes()) > 0 else 0,
            }

            circles_info = self._get_circles_info(self.data_dir / "facebook", ego_id)
            if circles_info:
                stats[ego_id]["circles"] = circles_info

        return stats

    # create_combined_facebook_network was unused; removed to keep loader slim

    def _load_node_features(self, graph: nx.Graph, data_dir: Path, ego_id: str) -> None:
        feat_file = data_dir / f"{ego_id}.feat"
        egofeat_file = data_dir / f"{ego_id}.egofeat"
        featnames_file = data_dir / f"{ego_id}.featnames"

        feature_names = []
        if featnames_file.exists():
            with featnames_file.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(maxsplit=1)
                        if len(parts) >= 2:
                            feature_names.append(parts[1])

        if feat_file.exists():
            with feat_file.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            node_id = int(parts[0])
                            features = [int(x) for x in parts[1:]]

                            if node_id in graph.nodes():
                                graph.nodes[node_id]["features"] = features
                                graph.nodes[node_id]["feature_count"] = sum(features)

        ego_node = int(ego_id)
        if egofeat_file.exists() and ego_node in graph.nodes():
            with egofeat_file.open() as f:
                line = f.readline().strip()
                if line:
                    features = [int(x) for x in line.split()]
                    graph.nodes[ego_node]["features"] = features
                    graph.nodes[ego_node]["feature_count"] = sum(features)

    def _load_circles(self, graph: nx.Graph, data_dir: Path, ego_id: str) -> None:
        circles_file = data_dir / f"{ego_id}.circles"

        if not circles_file.exists():
            return

        circles = []
        with circles_file.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        circle_name = parts[0]
                        circle_members = [int(x) for x in parts[1:] if x.isdigit()]
                        circles.append({"name": circle_name, "members": circle_members, "size": len(circle_members)})

        for node_id in graph.nodes():
            node_circles = []
            for i, circle in enumerate(circles):
                if node_id in circle["members"]:
                    node_circles.append(i)
            graph.nodes[node_id]["circles"] = node_circles

        graph.graph["circles"] = circles

    def _get_circles_info(self, data_dir: Path, ego_id: str) -> dict[str, Any] | None:
        circles_file = data_dir / f"{ego_id}.circles"

        if not circles_file.exists():
            return None

        circles = []
        with circles_file.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        circle_name = parts[0]
                        circle_size = len(parts) - 1
                        circles.append({"name": circle_name, "size": circle_size})

        return {
            "total_circles": len(circles),
            "avg_circle_size": sum(c["size"] for c in circles) / len(circles) if circles else 0,
            "max_circle_size": max((c["size"] for c in circles), default=0),
            "min_circle_size": min((c["size"] for c in circles), default=0),
        }

    # get_suitable_networks_for_testing was unused; removed

```

### src/glopt/io/fs.py

```python
from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str) -> None:
    Path(path).mkdir(exist_ok=True, parents=True)


def build_paths(run_id: str) -> tuple[str, str, str]:
    base = Path("runs") / run_id
    graphs_dir = base / "graphs"
    csv_dir = base / "csv"
    ensure_dir(str(graphs_dir))
    ensure_dir(str(csv_dir))
    return str(base), str(graphs_dir), str(csv_dir)

```

### src/glopt/io/graph_generator.py

```python
from collections.abc import Callable
from typing import ClassVar

import networkx as nx

GeneratorFn = Callable[..., nx.Graph]


class GraphGeneratorFactory:
    _GENERATORS: ClassVar[dict[str, GeneratorFn]] = {
        "random": lambda *, n_nodes, **p: GraphGeneratorFactory._random(n_nodes, **p),
        "scale_free": lambda *, n_nodes, **p: GraphGeneratorFactory._scale_free(n_nodes, **p),
        "small_world": lambda *, n_nodes, **p: GraphGeneratorFactory._small_world(n_nodes, **p),
        "complete": lambda *, n_nodes, **p: GraphGeneratorFactory._complete(n_nodes, **p),
        "star": lambda *, n_nodes, **p: GraphGeneratorFactory._star(n_nodes, **p),
        "path": lambda *, n_nodes, **p: GraphGeneratorFactory._path(n_nodes, **p),
        "cycle": lambda *, n_nodes, **p: GraphGeneratorFactory._cycle(n_nodes, **p),
        "tree": lambda *, n_nodes, **p: GraphGeneratorFactory._tree(n_nodes, **p),
    }

    @classmethod
    def get(cls, name: str) -> GeneratorFn:
        try:
            return cls._GENERATORS[name]
        except KeyError:
            available = ", ".join(cls._GENERATORS.keys())
            msg = f"unknown graph generator '{name}'. available: {available}"
            raise ValueError(msg) from None

    @staticmethod
    def _random(n_nodes: int, *, p: float = 0.1, seed: int | None = None) -> nx.Graph:
        return nx.gnp_random_graph(n=n_nodes, p=p, seed=seed)

    @staticmethod
    def _scale_free(n_nodes: int, *, m: int = 2, seed: int | None = None) -> nx.Graph:
        return nx.barabasi_albert_graph(n=n_nodes, m=m, seed=seed)

    @staticmethod
    def _small_world(n_nodes: int, *, k: int = 4, p: float = 0.1, seed: int | None = None) -> nx.Graph:
        return nx.watts_strogatz_graph(n=n_nodes, k=k, p=p, seed=seed)

    @staticmethod
    def _complete(n_nodes: int) -> nx.Graph:
        return nx.complete_graph(n=n_nodes)

    @staticmethod
    def _star(n_nodes: int) -> nx.Graph:
        return nx.star_graph(n=n_nodes - 1)

    @staticmethod
    def _path(n_nodes: int) -> nx.Graph:
        return nx.path_graph(n=n_nodes)

    @staticmethod
    def _cycle(n_nodes: int) -> nx.Graph:
        return nx.cycle_graph(n=n_nodes)

    @staticmethod
    def _tree(n_nodes: int, *, seed: int | None = None) -> nx.Graph:
        if n_nodes == 1:
            graph = nx.Graph()
            graph.add_node(0)
            return graph

        base = nx.complete_graph(n_nodes)
        return nx.random_spanning_tree(base, weight=None, seed=seed)

```

### src/glopt/io/graph_visualizer.py

```python
from datetime import datetime
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import pathlib

import matplotlib.pyplot as plt
import networkx as nx

from glopt.core import Solution


class GraphVisualizer:
    def __init__(
        self,
        figsize: tuple[int, int] = (12, 8),
        layout_seed: int = 42,
        reuse_layout: bool = True,
    ) -> None:
        self.figsize = figsize
        self.layout_seed = layout_seed
        self.reuse_layout = reuse_layout

        self.default_edge_color = "#808080"
        self.owner_size = 500
        self.member_size = 300
        self.solo_size = 400

        self._pos: dict[Any, tuple[float, float]] | None = None

    def visualize_solution(
        self,
        graph: nx.Graph,
        solution: Solution,
        solver_name: str,
        timestamp_folder: str | None = None,
        save_path: str | None = None,
    ) -> str:
        if not self.reuse_layout or self._pos is None:
            self._pos = nx.spring_layout(graph, seed=self.layout_seed)
        else:
            self._update_positions_for_graph(graph)

        node_to_group = self._map_nodes_to_groups(solution)
        node_colors, node_sizes = self._get_node_properties(graph, node_to_group)
        edge_colors = self._get_edge_colors(graph, node_to_group)

        _, ax = plt.subplots(figsize=self.figsize)
        nx.draw_networkx_edges(graph, self._pos, edge_color=edge_colors, alpha=0.7, width=1.5, ax=ax)
        nx.draw_networkx_nodes(graph, self._pos, node_color=node_colors, node_size=node_sizes, ax=ax)

        self._add_legend(ax, solution)
        ax.axis("off")

        if save_path is None:
            if timestamp_folder is None:
                timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
            n_nodes, n_edges = graph.number_of_nodes(), graph.number_of_edges()
            save_path = f"runs/graphs/{timestamp_folder}/{solver_name}_{n_nodes}n_{n_edges}e.png"
        pathlib.Path(pathlib.Path(save_path).parent).mkdir(exist_ok=True, parents=True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path

    # reset_layout and set_layout were unused; removed

    def _update_positions_for_graph(self, graph: nx.Graph) -> None:
        assert self._pos is not None
        g_nodes = set(graph.nodes())
        pos_nodes = set(self._pos.keys())

        for n in pos_nodes - g_nodes:
            self._pos.pop(n, None)

        new_nodes = g_nodes - pos_nodes
        if not new_nodes:
            return

        for n in new_nodes:
            neigh = [v for v in graph.neighbors(n) if v in self._pos]
            if neigh:
                anchor = random_choice(neigh)
                ax, ay = self._pos[anchor]

                self._pos[n] = (ax + jitter(), ay + jitter())
            else:
                self._pos[n] = (jitter(scale=0.25), jitter(scale=0.25))

    def _map_nodes_to_groups(self, solution: Solution) -> dict[Any, Any]:
        mapping: dict[Any, Any] = {}
        for group in solution.groups:
            for member in group.all_members:
                mapping[member] = group
        return mapping

    def _get_node_properties(self, graph: nx.Graph, node_to_group: dict[Any, Any]) -> tuple[list[str], list[int]]:
        colors: list[str] = []
        sizes: list[int] = []
        owners = {g.owner for g in node_to_group.values()}

        for node in graph.nodes():
            group = node_to_group.get(node)
            if group is None:
                colors.append("#cccccc")
                sizes.append(self.solo_size)
            else:
                colors.append(group.license_type.color)
                sizes.append(self.owner_size if node in owners else self.member_size)
        return colors, sizes

    def _get_edge_colors(self, graph: nx.Graph, node_to_group: dict[Any, Any]) -> list[str]:
        colors: list[str] = []
        for u, v in graph.edges():
            g1 = node_to_group.get(u)
            g2 = node_to_group.get(v)
            if g1 is not None and g1 == g2:
                colors.append(g1.license_type.color)
            else:
                colors.append(self.default_edge_color)
        return colors

    def _add_legend(self, ax, solution: Solution) -> None:
        license_types = sorted({g.license_type for g in solution.groups}, key=lambda lt: lt.name)
        if not license_types:
            return
        elems = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=lt.color, markersize=10, label=lt.name) for lt in license_types]
        elems.append(plt.Line2D([0], [0], color=self.default_edge_color, linewidth=2, label="Other Edges"))
        ax.legend(handles=elems, loc="upper right", bbox_to_anchor=(0.98, 0.98))


def jitter(scale: float = 0.08) -> float:
    import random as _r

    return (_r.random() - 0.5) * 2.0 * scale


def random_choice(seq):
    import random as _r

    return _r.choice(list(seq))

```

### src/glopt/license_config.py

```python
from collections.abc import Callable
from typing import ClassVar

from .core import LicenseType


class LicenseConfigFactory:
    PURPLE = "#542f82"
    GOLD = "#cb8a35"
    GREEN = "#5d9f49"

    _CONFIGS: ClassVar[dict[str, Callable[[], list[LicenseType]]]] = {
        "duolingo_super": lambda: [
            LicenseType("Individual", 13.99, 1, 1, LicenseConfigFactory.PURPLE),
            LicenseType("Family", 29.17, 2, 6, LicenseConfigFactory.GOLD),
        ],
        "spotify": lambda: [
            LicenseType("Individual", 23.99, 1, 1, LicenseConfigFactory.PURPLE),
            LicenseType("Duo", 30.99, 2, 2, LicenseConfigFactory.GREEN),
            LicenseType("Family", 37.99, 2, 6, LicenseConfigFactory.GOLD),
        ],
        "roman_domination": lambda: [
            LicenseType("Solo", 1.0, 1, 1, LicenseConfigFactory.PURPLE),
            LicenseType("Group", 2.0, 2, 99999, LicenseConfigFactory.GOLD),
        ],
    }

    @classmethod
    def get_config(cls, name: str) -> list[LicenseType]:
        # Dynamic roman domination sweep: roman_p_1_5 or roman_p:2.5
        if name.startswith("roman_p_") or name.startswith("roman_p:"):
            p_str = name.split("_", 2)[2] if name.startswith("roman_p_") else name.split(":", 1)[1]
            p_str = p_str.replace("_", ".")
            try:
                p_val = float(p_str)
            except Exception:
                available = ", ".join(cls._CONFIGS.keys())
                raise ValueError(
                    f"Invalid roman price '{name}'. Available: {available} or roman_p_<x_y>/roman_p:<x.y>"
                )
            return [
                LicenseType("Solo", 1.0, 1, 1, cls.PURPLE),
                LicenseType("Group", p_val, 2, 99999, cls.GOLD),
            ]
        try:
            return cls._CONFIGS[name]()
        except KeyError:
            available = ", ".join(cls._CONFIGS.keys())
            msg = f"Unsupported license config: {name}. Available: {available} or roman_p_<x_y>/roman_p:<x.y>"
            raise ValueError(msg) from None

```

### tests/__init__.py

```python
# Make tests a package to satisfy ruff INP001.

```

### tests/test_algorithms.py

```python
import networkx as nx
import pytest

from glopt.algorithms import (
    AntColonyOptimization,
    DominatingSetAlgorithm,
    GeneticAlgorithm,
    GreedyAlgorithm,
    ILPSolver,
    NaiveAlgorithm,
    RandomizedAlgorithm,
    SimulatedAnnealing,
    TabuSearch,
)
from glopt.core.solution_validator import SolutionValidator
from glopt.io.graph_generator import GraphGeneratorFactory
from glopt.license_config import LicenseConfigFactory

GRAPH_SPECS = {
    "random": {"p": 0.1, "seed": 42},
    "small_world": {"k": 4, "p": 0.1, "seed": 42},
    "scale_free": {"m": 2, "seed": 42},
}

LICENSE_CFGS = ["roman_domination", "duolingo_super", "spotify"]

ALGOS = [
    ("ilp", lambda: ILPSolver(), {}, 25),
    ("greedy", lambda: GreedyAlgorithm(), {}, 4000),
    ("dominating_set", lambda: DominatingSetAlgorithm(), {}, 1500),
    ("randomized", lambda: RandomizedAlgorithm(seed=42), {}, 4000),
    ("genetic", lambda: GeneticAlgorithm(population_size=20, generations=20, seed=42), {}, 600),
    ("simulated_annealing", lambda: SimulatedAnnealing(max_iterations=200, max_stall=50), {}, 1000),
    ("tabu_search", lambda: TabuSearch(), {"max_iterations": 100, "neighbors_per_iter": 5, "tabu_tenure": 7}, 1500),
    ("ant_colony", lambda: AntColonyOptimization(num_ants=5, max_iterations=20), {}, 700),
    ("naive", lambda: NaiveAlgorithm(), {}, 10),
]


validator = SolutionValidator(debug=False)


def generate_graph(name: str, n: int) -> nx.Graph:
    params = GRAPH_SPECS[name]
    gen = GraphGeneratorFactory.get(name)
    return gen(n_nodes=n, **params)


@pytest.mark.parametrize("license_cfg", LICENSE_CFGS)
@pytest.mark.parametrize("graph_name", list(GRAPH_SPECS.keys()))
@pytest.mark.parametrize(("algo_id", "algo_factory", "algo_kwargs", "n_nodes"), ALGOS, ids=[a[0] for a in ALGOS])
def test_algorithms_validity(graph_name: str, license_cfg: str, algo_id, algo_factory, algo_kwargs, n_nodes: int) -> None:
    license_types = LicenseConfigFactory.get_config(license_cfg)
    graph = generate_graph(graph_name, n_nodes)
    algo = algo_factory()
    solution = algo.solve(graph=graph, license_types=license_types, **algo_kwargs)
    ok, issues = validator.validate(solution, graph)
    assert ok, f"{algo.name} invalid: {issues}"

```
