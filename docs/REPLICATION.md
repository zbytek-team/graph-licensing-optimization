# Replikacja eksperymentów

## Wymagania
- Python 3.11+
- `uv` do zarządzania środowiskiem (Makefile zakłada `uv`)

## Instalacja
```
make install
```

## Benchmark statyczny (syntetyczne grafy)
```
make benchmark
```
- Wyniki CSV: `runs/<run_id>_benchmark/csv/<run_id>_benchmark.csv`
- Wykresy i agregaty: `python3 scripts/analyze.py` (zapisuje do `runs/<run_id>/analysis`)

## Benchmark statyczny (grafy rzeczywiste – Facebook ego)
```
make benchmark_real
```
- Wyniki CSV: `runs/<run_id>_benchmark_real/csv/<run_id>_benchmark_real.csv`

## Benchmark dynamiczny (warm vs cold)
```
make dynamic
```
- Wyniki CSV: `runs/<run_id>_dynamic/csv/<run_id>_dynamic.csv`
- Zmienne konfiguracyjne (liczba kroków, intensywność mutacji) w `src/glopt/cli/dynamic.py`

## Analiza wyników
```
python3 scripts/analyze.py
```
- Generuje: koszt vs n, czas vs n (log), Pareto, profile wydajności, miks licencji, tabele agregatów (średnie, odchylenia, 95% CI)
- Wyjścia w `runs/<run_id>/analysis`

## Uwagi
- Benchmarki stosują twardy limit czasu 60 s na bieg algorytmu (kill procesu) i zatrzymanie skali n po pierwszym timeout dla danej pary (graph, algorithm).
- Cache grafów syntetycznych generowany jest automatycznie do `data/graphs_cache` przy starcie benchmarku.
