# Optymalizacja dystrybucji licencji

Projekt demonstruje algorytmy do optymalizacji rozdziału licencji w sieciach
komunikacyjnych. W zadaniu każdemu węzłowi w grafie należy przypisać typ
licencji tak, aby spełnić zadane ograniczenia (np. minimalna liczba licencji
premium) przy jednoczesnej minimalizacji kosztu. Repozytorium zawiera różne
podejścia algorytmiczne (dokładne i heurystyczne) oraz narzędzia do ich
porównywania na losowych i rzeczywistych danych.

## Wymagania

* Python 3.13 lub nowszy
* Zależności z `pyproject.toml`:
  * `matplotlib`
  * `networkx`
  * `pulp`
  * `pymupdf`

Możesz zainstalować wszystko poleceniem:

```bash
pip install -e .
# lub z pakietami deweloperskimi
pip install -e .[dev]
```

## Uruchamianie skryptów

W katalogu `scripts/` znajdują się pomocnicze narzędzia do uruchamiania
algorytmów:

* `benchmark_real_world.py` – porównanie algorytmów na rzeczywistych
  sieciach Facebooka:
  ```bash
  python scripts/benchmark_real_world.py
  ```
* `run_benchmark.py` – generowanie sztucznych grafów o różnych parametrach i
  zapis wyników do pliku CSV:
  ```bash
  python scripts/run_benchmark.py
  ```
* `run_comparison.py` – uruchomienie pełnego porównania wielu algorytmów na
  pojedynczym grafie:
  ```bash
  python scripts/run_comparison.py
  ```

Każdy skrypt korzysta z domyślnych ustawień zapisanych w kodzie; w razie
potrzeby można go modyfikować bezpośrednio w pliku.
