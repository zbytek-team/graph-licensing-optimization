# Thesis Guide -- Mapping Code to Chapters

This guide maps the thesis table of contents to code, data, and outputs produced by this repository.

## 1. Wprowadzenie
- 1.1-1.3 Narrative. Use README context and Results figures as references.

## 2. Model grafowy i analiza problemu
- 2.1 Reprezentacja grafowa: `src/glopt/io/graph_generator.py`, real: `data/facebook/*`.
- 2.2 Definicja problemu: `src/glopt/core/models.py` (LicenseType/Group/Solution), `src/glopt/core/solution_validator.py` (ograniczenia pokrycia/sąsiedztwa/pojemności).
- 2.3 Koszty i ograniczenia: `src/glopt/license_config.py` (Duolingo, roman, warianty `*_p_*`).

## 3. Związek z dominowaniem w grafach
- 3.1-3.2: Odwołania do konfiguracji `roman_domination` i `roman_p_*` (unbounded capacity group).
- 3.3 Złożoność: komentarze przy `src/glopt/algorithms/*` i ILP.

## 4. Dane testowe
- 4.1 Grafy syntetyczne: generowane przez CLI (`benchmark.py`, `dynamic.py`) i cache w `data/graphs_cache` (gpickle+json).
- 4.2 Grafy rzeczywiste: `data/facebook/*` + CLIs `benchmark_real.py`, `dynamic_real.py`.

## 5. Metody algorytmiczne
- 5.1 Dokładne: `src/glopt/algorithms/ilp.py`.
- 5.2 Zachłanne i heurystyczne: `src/glopt/algorithms/{greedy,randomized,dominating_set,genetic,simulated_annealing,tabu_search,ant_colony}.py`.

## 6. Eksperymenty i analiza wyników
- 6.1 Kryteria: koszt całkowity, czas (ms), profile wydajności; wykresy i tabele generuje `make analyze`.
- 6.2 Środowisko: Python 3.13 (wymuszone), `uv`; logi w `runs/<run_id>/glopt.log`.
- 6.3 Eksperymenty: CLIs w `src/glopt/cli/*` (konfiguracja w kodzie, spójny logging).
- Analizy i figury: `runs/<run_id>/analysis/**`
  - `*_cost_vs_n.png`, `*_time_vs_n.png`, `*_density_vs_time.png`, `*_pareto_cost_time.png`, `perf_profile_*`, `aggregates.csv`.

## 7. Analiza dynamicznej wersji
- 7.1 Definicja: `src/glopt/dynamic_simulator.py` + `src/glopt/cli/dynamic*.py`.
- 7.2 Adaptacja: warm-starty, projekcja rozwiązań (w CLI dynamicznych).
- 7.3 Eksperymenty: wyniki i figury w `analysis` (analogicznie jak w 6.x).

## 8. Rozszerzenia modelu
- 8.1 Polityki cenowe: `roman_p_*`, `duolingo_p_*`.
- 8.2 Dodatkowe typy licencji: `spotify`, można rozszerzać w `license_config.py`.

## 9. Zakończenie
- 9.1-9.2: Wnioski oparte o `aggregates.csv`, profile wydajności, pivoty z `summary_pandas.py`.

## Szybki przepływ pracy (Figury/Tabele)
1. Uruchom benchmark/dynamic (syntetyczne/real): `make benchmark`, `make dynamic`, `make benchmark_real`, `make dynamic_real`.
2. Analiza: `make analyze` (gdy brak runs/*/csv, zostanie użyte `filtered_results.zip`).
3. Auto-raport (markdown): `make report RUN=<run_id>` ⇒ `docs/reports/<run_id>/report.md` z linkami do figur i tabel.
