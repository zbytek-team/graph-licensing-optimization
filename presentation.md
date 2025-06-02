# Modelowanie Optymalnego Zakupu Licencji Oprogramowania w Sieciach Społecznych za pomocą Dominacji w Grafach

---

## Slajd 1: Sformułowanie Problemu i Motywacja

- Problem dotyczy optymalnego wyboru między licencjami rodzinnymi a indywidualnymi w sieci społecznej
- Licencja rodzinna jest droższa, ale może obsłużyć wszystkich sąsiadów w sieci (maksymalnie 6 osób)
- Licencja indywidualna jest tańsza, ale obsługuje tylko jednego użytkownika
- Celem jest minimalizacja całkowitego kosztu przy zapewnieniu pokrycia wszystkich użytkowników

**Motywacja:**
- Inspiracja: struktura cenowa Duolingo Super (licencja rodzinna dla 6 osób za ~2.08x ceny indywidualnej)
- Teoria dominowania w grafach jako fundament matematyczny
- Praktyczne zastosowania w zarządzaniu licencjami zespołowymi

---

## Slajd 2: Model Matematyczny

**Struktura sieci społecznej:**
- Wierzchołki (V) reprezentują użytkowników
- Krawędzie (E) reprezentują relacje przyjaźni/współpracy
- Graf G = (V, E) modeluje sieć społeczną

**Rodzaje licencji:**
- Licencja solo: koszt `s` dla jednego użytkownika
- Licencja grupowa: koszt `g` dla właściciela grupy + maksymalnie `k-1` sąsiadów

**Funkcja celu:**
```
minimize: Σ(s × x_i) + Σ(g × y_i)
```

**Ograniczenia:**
- Każdy użytkownik musi mieć dostęp do licencji
- Członkowie grupy muszą być sąsiadami właściciela
- Maksymalny rozmiar grupy to `k` osób
- Członek może należeć tylko do jednej grupy

---

## Slajd 3: Związek z Dominowaniem Rzymskim

- Problem jest podobny do dominowania rzymskiego, ale z kluczowymi różnicami:
  - **Ograniczenie rozmiaru grupy:** maksymalnie `k` wierzchołków w grupie (w Duolingo k=6)
  - **Różne warianty cenowe:** gdy cena grupowa jest np. 2.08x większa, dla 2 osób solo jest bardziej opłacalne (2.00 < 2.08)
  - **Próg opłacalności:** grupowa licencja staje się korzystna dopiero od 3+ członków

**Złożoność problemu:**
- NP-trudny (podobnie jak dominowanie w grafach)
- Wymaga heurystyk i metaheurystyk dla większych instancji
- Dodatkowe ograniczenia czynią go bardziej skomplikowanym niż klasyczne dominowanie rzymskie

---

## Slajd 4: Architektura Implementacji

**Środowisko:**
- Python 3.13
- Biblioteki: NetworkX, PuLP, NumPy, SciPy, Matplotlib, Pandas

**Komponenty systemu:**
- **Algorytmy:** implementacje różnych podejść optymalizacyjnych
- **Generatory grafów:** tworzenie różnych topologii sieci
- **Benchmarking:** pomiar wydajności i jakości rozwiązań
- **Wizualizacja:** prezentacja wyników i porównań

**Struktura pakietu:**
```
src/graph_licensing/
├── algorithms/     # Algorytmy optymalizacyjne
├── generators/     # Generatory grafów testowych
├── models/         # Modele danych (LicenseConfig, LicenseSolution)
├── utils/          # Narzędzia (Benchmark, FileIO)
└── visualizers/    # Wizualizacja wyników
```

---

## Slajd 5: Klasyfikacja Algorytmów

**Algorytmy aproksymacyjne (approx/):**
- `GreedyAlgorithm`: zachłanny wybór najlepszych grup
- `DominatingSetAlgorithm`: oparty na heurystykach dominowania
- `RandomizedAlgorithm`: losowy wybór (baseline)

**Algorytmy dokładne (exact/):**
- `ILPAlgorithm`: programowanie liniowe całkowitoliczbowe (PuLP)
- `NaiveAlgorithm`: przegląd zupełny dla małych grafów

**Metaheurystyki (meta/):**
- `GeneticAlgorithm`: algorytm genetyczny z mutacją i krzyżowaniem
- `SimulatedAnnealingAlgorithm`: symulowane wyżarzanie
- `TabuSearchAlgorithm`: przeszukiwanie tabu

---

## Slajd 6: Algorytm ILP - Formalizacja Matematyczna

**Zmienne decyzyjne:**
- `x_i ∈ {0,1}`: czy wierzchołek i ma licencję solo
- `y_i ∈ {0,1}`: czy wierzchołek i jest właścicielem grupy
- `z_ij ∈ {0,1}`: czy wierzchołek j jest członkiem grupy właściciela i

**Funkcja celu:**
```
minimize: Σ(solo_price × x_i) + Σ(group_price × y_i)
```

**Ograniczenia:**
1. **Pokrycie:** `x_i + Σ_j z_ji = 1` ∀i (każdy musi mieć licencję)
2. **Rozmiar grupy:** `Σ_j z_ij ≤ group_size × y_i` ∀i
3. **Sąsiedztwo:** `z_ij = 0` jeśli (i,j) ∉ E i i≠j
4. **Spójność:** `z_ii = y_i` ∀i (właściciel należy do własnej grupy)

**Implementacja:** biblioteka PuLP z solverem CBC

---

## Slajd 7: Algorytm Greedy - Strategia Zachłanna

**Strategia:**
1. Sortowanie wierzchołków według stopnia (malejąco)
2. Dla każdego nieprzypisanego wierzchołka:
   - Sprawdzenie dostępnych sąsiadów
   - Ocena opłacalności utworzenia grupy
   - Wybór najlepszych sąsiadów (według stopnia)

**Kluczowa heurystyka:**
```python
def is_group_beneficial(self, group_members):
    return group_members * solo_price > group_price
```

**Zalety:**
- Szybkość: O(V log V + E)
- Dobra jakość rozwiązań w praktyce
- Prostota implementacji

**Wady:**
- Brak gwarancji optymalności
- Zachłanne decyzje mogą prowadzić do lokalnych optimów

---

## Slajd 8: Algorytm Tabu Search

**Strategia przeszukiwania:**
1. Start z rozwiązania zachłannego
2. Generowanie wszystkich sąsiednich rozwiązań
3. Wybór najlepszego niedozwolonego ruchu
4. Aktualizacja listy tabu

**Operacje sąsiedztwa:**
- Przypisanie licencji solo do wierzchołka
- Przeniesienie między grupami
- Tworzenie nowych grup z sąsiadami

**Parametry:**
- `max_iterations = 100`
- `max_no_improvement = 20`
- `tabu_size = nodes // 20`

**Problem:** bardzo długi czas działania dla większych grafów

---

## Slajd 9: Generowanie Grafów Testowych

**Typy grafów:**
- **Random (Erdős–Rényi):** losowe krawędzie z prawdopodobieństwem p
- **Scale-free (Barabási–Albert):** preferential attachment, power-law degree distribution
- **Small-world (Watts–Strogatz):** high clustering, short path lengths
- **Complete:** każdy z każdym połączony
- **Grid:** siatka 2D
- **Star:** jeden centralny węzeł
- **Path/Cycle:** łańcuch/cykl wierzchołków

**Dane rzeczywiste:**
- **ego-Facebook** ze SNAP Stanford
- Sieci społeczne Facebook (anonimizowane)
- 4,039 węzłów, 88,234 krawędzi
- 10 sieci ego o różnych rozmiarach

---

## Slajd 10: Interfejs Głównej Aplikacji (main.py)

**Trzy tryby działania:**

**1. Single - test pojedynczego algorytmu:**
```bash
python main.py single --algorithm greedy --graph-type random --graph-size 64
```

**2. Benchmark - pełne testy wydajnościowe:**
```bash
python main.py benchmark --algorithms greedy dominating_set --graph-types random scale_free
```

**3. Compare - porównanie algorytmów:**
```bash
python main.py compare --algorithms greedy simulated_annealing tabu_search --graph-type facebook
```

**Parametry konfiguracyjne:**
- `--solo-cost`: koszt licencji indywidualnej (default: 1.0)
- `--group-cost`: koszt licencji grupowej (default: 2.08)
- `--group-size`: maksymalny rozmiar grupy (default: 6)

---

## Slajd 11: Metryki Wydajności

**1. Koszt (Cost):**
- Całkowity koszt rozwiązania
- Liczba licencji solo vs grupowych
- Efektywność wykorzystania grup

**2. Czas wykonania (Runtime):**
- Czas rozwiązania w sekundach
- Skalowalność względem rozmiaru grafu
- Efektywność czasowa algorytmów

**3. Skalowalność:**
- Zachowanie dla grafów o rozmiarach 8-512 węzłów
- Stabilność wyników
- Wskaźnik sukcesu

**Metryki dodatkowe:**
- Validność rozwiązania
- Średni rozmiar grup
- Współczynnik krawędzie/węzły

---

## Slajd 12: Analiza Kosztów - Porównanie Algorytmów

![Wykres kosztów](analysis_chart_placeholder)

**Kluczowe obserwacje:**
- **Najlepsze koszty:** Greedy, DominatingSet, SimulatedAnnealing (identyczne dla małych grafów)
- **Słabe wyniki:** Genetic Algorithm (tylko licencje solo), Randomized (baseline)
- **TabuSearch:** najlepsze wyniki dla większych grafów, ale bardzo długi czas

**Statystyki z benchmark_results.csv:**
- Random 8 węzłów: Greedy/DominatingSet/SA = 5.08, Genetic/Randomized = 8.0
- Random 16 węzłów: TabuSearch = 7.24 (najlepszy), Genetic = 16.0 (najgorszy)
- Trend: wraz ze wzrostem rozmiaru różnice się zwiększają

---

## Slajd 13: Analiza Czasu Wykonania

![Wykres czasów](runtime_chart_placeholder)

**Ranking wydajności czasowej:**
1. **Najszybsze:** Greedy, DominatingSet (~10⁻⁵ s)
2. **Średnie:** Randomized (~10⁻⁵ s), SimulatedAnnealing (~10⁻⁴ s)
3. **Wolne:** TabuSearch (~10⁻² s), Genetic (~10⁻² s)

**Problemy skalowalności:**
- TabuSearch: dramatyczny wzrost czasu (0.002s → 0.04s dla 32 węzłów)
- Genetic Algorithm: niestabilne czasy, słaba konwergencja
- ILP: nie pokazany ze względu na limity czasowe dla większych instancji

---

## Slajd 14: Analiza Skalowalności

![Wykres skalowalności](scalability_chart_placeholder)

**Wzorce skalowalności:**
- **Greedy/DominatingSet:** liniowy wzrost czasu, stabilne koszty
- **SimulatedAnnealing:** dobra skalowalność czasu i jakości
- **TabuSearch:** kwadratowy wzrost czasu, ale najlepsze koszty
- **Genetic:** słaba konwergencja, problemy z większymi grafami

**Rekomendacje użycia:**
- **Małe grafy (≤32):** dowolny algorytm poza Genetic
- **Średnie grafy (32-128):** Greedy, SimulatedAnnealing
- **Duże grafy (>128):** tylko Greedy lub DominatingSet

---

## Slajd 15: Wydajność według Typu Grafu

**Random grafy:**
- Wszystkie algorytmy działają stabilnie
- Wysokie zagęszczenie krawędzi sprzyja grupom

**Scale-free grafy:**
- Algorytmy wykorzystują węzły o wysokim stopniu
- Greedy szczególnie efektywny (wykorzystuje centra)

**Small-world grafy:**
- Balans między lokalnym klastrowaniem a globalnymi połączeniami
- Średnia wydajność wszystkich algorytmów

**Facebook ego-networks:**
- Realistyczne struktury społeczne
- Najlepsze wyniki dla algorytmów wykorzystujących topologię

---

## Slajd 16: Wizualizacje - Random Graph

![Porównanie algorytmów - random](results/comparison/comparisonrandom.png)

**Obserwacje:**
- Różne strategie tworzenia grup
- Podobne koszty końcowe dla dobrych algorytmów
- Wizualne różnice w rozkładzie licencji

---

## Slajd 17: Wizualizacje - Scale-Free Graph

![Porównanie algorytmów - scale-free](results/comparison/comparisonscalefree.png)

**Charakterystyka:**
- Wykorzystanie węzłów o wysokim stopniu jako właścicieli grup
- Wyraźne centra z dużymi grupami
- Efektywne pokrycie przez hubs

---

## Slajd 18: Wizualizacje - Small-World Graph

![Porównanie algorytmów - small-world](results/comparison/comparisonsmallworld.png)

**Właściwości:**
- Lokalne klastry z globalnymi skrótami
- Mieszane strategie grupowania
- Balans między lokalnymi a globalnymi optimami

---

## Slajd 19: Wizualizacje - Facebook Network

![Porównanie algorytmów - facebook](results/comparison/comparisonfacebook.png)

**Realistyczne sieci społeczne:**
- Naturalne struktury społeczności
- Hierarchiczne grupy przyjaźni
- Najbliższe rzeczywistym przypadkom użycia

---

## Slajd 20: Grafy Specjalne - Gwiazda

**Graf gwiazdy (Star):**
- Jeden centralny węzeł połączony z wszystkimi innymi
- **Optymalne rozwiązanie:** 1 grupa (centrum + max k-1 liści) + reszta solo

**Wzór na optymalny koszt:**
```
cost = group_price + max(0, n-k) × solo_price
```

**Przykład dla n=10, k=6:**
- 1 grupa: centrum + 5 liści (koszt: 2.08)  
- 4 licencje solo (koszt: 4.0)
- **Całkowity koszt: 6.08**

**Implementacja łatwa - jeden przypadek specjalny**

---

## Slajd 21: Grafy Specjalne - Ścieżka i Cykl

**Ścieżka (Path) i Cykl (Cycle):**
- Każdy węzeł ma maksymalnie 2 sąsiadów
- Grupy maksymalnie 3-osobowe (w ścieżce) lub wszystkie k-osobowe
- **Kluczowa obserwacja:** ścieżka i cykl są równoważne dla grup

**Strategia optymalna:**
- Tworzenie grup o rozmiarze min(k, lokalny_segment)
- Minimalizacja nakładania się grup

**Wzór przybliżony:**
```
groups_needed = ⌈n / k⌉
cost ≈ groups_needed × group_price
```

**Prostota:** algorytmy greedy są praktycznie optymalne

---

## Slajd 22: Graf Pełny (Complete Graph)

**Graf pełny K_n:**
- Każdy węzeł połączony z każdym innym
- **Modeluje:** zespół gdzie wszyscy się znają
- **Optymalne rozwiązanie:** maksymalne wykorzystanie grup

**Strategia:**
- Tworzenie grup po k osób
- Minimalna liczba grup: ⌈n/k⌉

**Wzór na optymalny koszt:**
```
optimal_groups = ⌈n/k⌉
cost = optimal_groups × group_price
```

**Przykład dla n=20, k=6:**
- 4 grupy po 6 osób (24 > 20, więc ostatnia grupa 2-osobowa może być solo)
- Optymalizacja: 3 grupy + 2 solo = 3×2.08 + 2×1.0 = 8.24

---

## Slajd 23: Problemy i Ograniczenia Algorytmów

**Genetic Algorithm:**
- **Problem:** słaba konwergencja, często tylko licencje solo
- **Przyczyna:** nieodpowiednia funkcja fitness, słabe operatory krzyżowania
- **Wyniki:** najgorsze koszty ze wszystkich algorytmów

**Dominating Set Algorithm:**
- **Problem:** problemy z optymalnym wykorzystaniem ograniczeń grupowych
- **Obserwacja:** podobne wyniki do Greedy, ale teoretycznie powinien być lepszy

**Tabu Search:**
- **Problem:** bardzo długi czas wykonania (rząd magnitude wolniejszy)
- **Przyczyna:** kompleksowa generacja sąsiedztwa
- **Paradoks:** najlepsze wyniki, ale praktycznie nieużywalny dla większych grafów

**ILP:**
- **Ograniczenie:** skaluje się tylko do ~100 węzłów
- **Czas:** wykładniczy wzrost dla gęstych grafów

---

## Slajd 24: Wersja Dynamiczna (Planowana)

**Motywacja:**
- Sieci społeczne zmieniają się w czasie
- Nowi użytkownicy, zmiany relacji
- Potrzeba adaptacji strategii licencjonowania

**Planowane funkcjonalności:**
- **Dodawanie węzłów:** nowi użytkownicy dołączają do sieci
- **Usuwanie węzłów:** użytkownicy opuszczają system
- **Modyfikacja krawędzi:** zmiany w relacjach przyjaźni
- **Renegocjacja grup:** automatyczna reorganizacja licencji

**Wyzwania implementacyjne:**
- Minimalizacja zmian w istniejących rozwiązaniach
- Efektywne updaty bez pełnego przeliczania
- Stabilność kosztów w czasie

**Zastosowania:**
- Systemy subskrypcyjne z dynamiczną populacją
- Licencjonowanie w organizacjach o zmiennej strukturze

---

## Slajd 25: Wnioski i Rekomendacje

**Najlepsze algorytmy w praktyce:**
1. **Greedy Algorithm:** najlepszy balans koszt/czas/prostota
2. **Simulated Annealing:** dobra jakość, rozsądny czas
3. **Tabu Search:** najlepsze koszty, ale długi czas

**Rekomendacje zastosowań:**
- **Aplikacje real-time:** Greedy Algorithm
- **Optymalizacja offline:** Simulated Annealing lub Tabu Search
- **Małe sieci (<50 węzłów):** ILP dla optymalności
- **Duże sieci (>500 węzłów):** tylko Greedy

**Czynniki decyzyjne:**
- Stosunek cen (group_price/solo_price) wpływa na opłacalność grup
- Topologia sieci: scale-free sprzyja grupowaniu, random jest neutralny
- Ograniczenia czasowe: większość praktycznych przypadków wymaga <1s

**Znaczenie praktyczne:**
- Model może być zastosowany do Duolingo, Microsoft Teams, Adobe Creative Cloud
- Potencjalne oszczędności: 20-40% kosztów licencjonowania w organizacjach

---

## Slajd 26: Kierunki Dalszych Badań

**Rozszerzenia modelu:**
- **Hierarchiczne grupy:** grupy w grupach, różne poziomy dostępu
- **Czasowe licencjonowanie:** różne okresy ważności licencji
- **Częściowe pokrycie:** niektórzy użytkownicy nie wymagają pełnego dostępu

**Nowe algorytmy:**
- **Machine Learning:** uczenie się optymalnych strategii z danych historycznych
- **Approximation algorithms:** gwarancje teoretyczne jakości rozwiązań
- **Distributed algorithms:** optymalizacja w systemach rozproszonych

**Praktyczne implementacje:**
- **API dla systemów licencjonowania:** real-time recommendations
- **Integracja z platformami:** Slack, Microsoft 365, Google Workspace
- **Analiza kosztów:** narzędzia dla menedżerów IT

**Badania empiryczne:**
- **Rzeczywiste dane organizacyjne:** validacja na prawdziwych strukturach
- **A/B testing:** porównanie strategii w środowisku produkcyjnym
- **Analiza wrażliwości:** wpływ zmian parametrów na koszty

---

## Bibliografia i Zasoby

**Źródła danych:**
- SNAP Stanford Network Analysis Project (Facebook ego-networks)
- NetworkX library documentation
- PuLP optimization library

**Literatura teoretyczna:**
- Roman domination problems in graph theory
- Approximation algorithms for domination problems
- Social network analysis and community detection

**Implementacja:**
- GitHub repository: graph-licensing-optimization
- Python 3.13, NetworkX 3.4, PuLP 3.2
- Comprehensive benchmarking and visualization tools
