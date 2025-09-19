# Dynamiczne mutacje – podsumowanie

Dane rzeczywiste – czasy dla algorytmów docelowych:
- Algorytm genetyczny (pref_triadic, ciepły start): mediana 0.41 s
- Algorytm genetyczny (pref_pref, ciepły start): mediana 0.49 s
- Algorytm genetyczny (rand_rewire, ciepły start): mediana 0.60 s
- Przeszukiwanie tabu (pref_triadic, ciepły start): mediana 1.19 s
- Przeszukiwanie tabu (pref_pref, ciepły start): mediana 1.28 s
- Algorytm mrówkowy (pref_triadic, ciepły start): mediana 1.32 s

Dane syntetyczne – koszt/węzeł dla algorytmów docelowych:
- Algorytm genetyczny (med, ciepły start): mediana 0.41
- Algorytm mrówkowy (low, ciepły start): mediana 0.41
- Algorytm genetyczny (low, ciepły start): mediana 0.41
- Przeszukiwanie tabu (low, ciepły start): mediana 0.42
- Algorytm genetyczny (high, ciepły start): mediana 0.42
- Przeszukiwanie tabu (med, ciepły start): mediana 0.42

Ciepły vs zimny start – koszt: ujemna delta oznacza przewagę ciepłego startu
- Brak pełnych par ciepły/zimny start dla wskazanych algorytmów w danych

Statystyki szczegółowe zapisano w katalogu tables/