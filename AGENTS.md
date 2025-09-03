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

- Długość linii: max 100 znaków (wyjątki: importy, URL-e, `# noqa`, disable-komentarze).
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
