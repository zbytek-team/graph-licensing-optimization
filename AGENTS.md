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
