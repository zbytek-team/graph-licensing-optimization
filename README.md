# Graph Licensing Optimization

## Dane rzeczywiste

Repozytorium korzysta z rzeczywistych sieci Facebook w formacie [SNAP](https://snap.stanford.edu/data/). 
Pliki `.edges`, `.feat`, `.egofeat`, `.featnames` oraz `.circles` powinny znajdować się w katalogu `data/facebook/`.

### Pobieranie danych

Automatyczne pobranie i rozpakowanie archiwum z danymi:

```bash
python scripts/download_facebook_data.py
```

Można także pobrać archiwum `facebook.tar.gz` ręcznie ze strony
[SNAP](https://snap.stanford.edu/data/ego-Facebook.html) i wypakować je do `data/facebook/`.

### Uruchomienie benchmarku

Po przygotowaniu danych można uruchomić benchmark algorytmów na sieciach Facebook:

```bash
python scripts/benchmark_real_world.py
```
