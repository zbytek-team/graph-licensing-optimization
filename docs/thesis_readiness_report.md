# Ocena gotowości repozytorium do pracy magisterskiej

Temat: „Modelowanie optymalnych sposobów zakupu licencji oprogramowania w sieciach społecznościowych za pomocą dominowania w grafach”  (ang. „Modeling optimal ways to purchase software licenses in social networks via graph domination”)

Autor: Marcin Połajdowicz  •  Promotor: dr inż. Joanna Raczek


## Streszczenie
Repozytorium implementuje kompletny aparat obliczeniowy do badania problemu: model formalny, zestaw algorytmów (dokładne i metaheurystyki), rozbudowany benchmark (statyczny) oraz mini‑benchmark dynamiczny z porównaniem warm‑start vs cold‑start. Obecne skrypty generują wyniki CSV oraz wspierają analizę (skrypty wizualizacji). Część teoretyczna (dowód równoważności z dominacją rzymską oraz opis wariantów) wymaga opracowania w pracy, ale kod umożliwia przeprowadzenie potrzebnych eksperymentów.

Najważniejsze braki do rozważenia: dołączenie grafów rzeczywistych (Facebook ego‑networks) do pętli benchmarków CLI, przegląd parametryczny cen (np. sweep stosunku p=c₂/c₁), oraz jasne osadzenie konfiguracji „roman_domination” jako instancji dominacji rzymskiej w tekście pracy.


## Mapowanie na wymagania pracy (zadania)

- Opisanie grafowego modelu sieci społecznościowej
  - Stan: Zaimplementowane. Rdzeń w `src/glopt/core` (modele, walidator), generatory grafów syntetycznych w `glopt.io.graph_generator` (Erdős-Rényi, Barabási-Albert, Watts-Strogatz, oraz proste grafy kontrolne).
  - Dodatkowo: walidator rozwiązań sprawdza pojemności, sąsiedztwo właścicieli i pełne pokrycie.

- Opisanie możliwości zakupu podstawowych licencji typu Duolingo Super
  - Stan: Zaimplementowane jako konfiguracje licencji w `glopt.license_config.LicenseConfigFactory` (np. `spotify`, `duolingo_super`) - każda licencja ma koszt, min/max pojemność i kolor do wizualizacji.
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
  - Cache grafów na dysku (`data/graphs_cache`) - rozgrzewany automatycznie, bieg pracuje na gotowych instancjach (szybko i deterministycznie).
  - CSV z obszernym zestawem metryk (czas, koszt, walidacja, zagęszczenie, średni stopień, klasteryzacja, liczba komponentów, rozkład typów licencji itp.).

- Benchmark dynamiczny (`glopt.cli.dynamic`)
  - Wspólna sekwencja mutacji dla porównywanych algorytmów, warm vs cold dla metaheurystyk i baseline’y (Greedy/ILP) - logowane krok po kroku, CSV z delta_cost i średnim czasem.

- Analiza wyników (`scripts/analyze.py`)
  - Rysunki kosztu vs n, czasy (log‑scale) z pasmami ufności, Pareto (czas vs koszt), zależność od gęstości, profile wydajności (Dolan-Moré), miks licencji, tabele agregatów (średnie/odchylenia/CI95).

- Dane rzeczywiste
  - Loader Facebook ego‑networks w `glopt.io.data_loader.RealWorldDataLoader` (folder `data/facebook`). Zwraca grafy i metadane (cechy, kręgi).
  - Rekomendacja: dołączyć te grafy do pętli benchmarków (np. tryb „real”) - prosty krok integracyjny.


## Co jeszcze warto dodać/przygotować (lista kontrolna)

1. Integracja grafów rzeczywistych do CLI
   - Dodać tryb benchmarku dla `RealWorldDataLoader` - pętla po ego‑sieciach z `data/facebook` i tych samych algorytmach/licencjach; wynik do osobnego CSV.

2. Sweep cen grupowych (p)
   - Wygenerować serię konfiguracji licencyjnych (np. `p ∈ {1.5, 2.0, 2.5, 3.0}`) i przebiec benchmark; w analizie pokazać wpływ p na koszt i strukturę grup.

3. Część teoretyczna (tekst pracy)
   - Formalny dowód równoważności z dominacją rzymską (opis odwzorowania rozwiązań grupowych na etykiety 0/1/2 i odwrotnie; analiza funkcji kosztu; rola L=∞).
   - Szkice redukcji do klasycznych problemów dominacji; przegląd złożoności obliczeniowej (NP‑trudność) i konsekwencji dla metod.

4. Metryki dodatkowe
   - Np. koszt per węzeł, udział pokrycia przez różne typy licencji, korelacja kosztu z gęstością/klasteryzacją/średnim stopniem; w dynamicznych - stabilność kosztu (wariancja delta_cost).

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

