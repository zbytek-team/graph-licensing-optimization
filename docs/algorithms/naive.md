# Naive Algorithm

- Idea: pełne przeszukiwanie (podziały na grupy i właścicieli) - dokładny, ale wykładniczy.

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
- Pamięć: w zależności od sposobu generowania - co najmniej O(V).

## Wyniki z ostatniego custom.py
- Brak w obecnym biegu custom (wyłączony domyślnie ze względu na koszty obliczeń).

## Uwagi
- Przydatny tylko dla bardzo małych grafów (≤ 10 węzłów).

## Mapowanie pseudokodu na kod
- Plik: `src/glopt/algorithms/naive.py`
- `solve(...)` - generowanie podziałów i sprawdzanie wykonalności, wybór najlepszego po koszcie
- Typy pomocnicze: `Assignment` - lista krotek (licencja, właściciel, członkowie) odpowiadających grupom
