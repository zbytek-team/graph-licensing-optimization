# Dynamiczne mutacje - podsumowanie

Dane rzeczywiste - czasy dla algorytmów docelowych:
- Algorytm genetyczny (pref_triadic, ciepły start): średnia 0.61 s
- Algorytm genetyczny (pref_pref, ciepły start): średnia 1.13 s
- Algorytm genetyczny (rand_rewire, ciepły start): średnia 1.23 s
- Przeszukiwanie tabu (pref_triadic, ciepły start): średnia 1.47 s
- Przeszukiwanie tabu (pref_pref, ciepły start): średnia 2.47 s
- Przeszukiwanie tabu (rand_rewire, ciepły start): średnia 2.58 s

Dane syntetyczne - koszt/węzeł dla algorytmów docelowych:
- Algorytm mrówkowy (med, ciepły start): średnia 0.42
- Algorytm mrówkowy (low, ciepły start): średnia 0.42
- Algorytm genetyczny (med, ciepły start): średnia 0.43
- Algorytm mrówkowy (high, ciepły start): średnia 0.43
- Algorytm genetyczny (high, ciepły start): średnia 0.43
- Algorytm genetyczny (low, ciepły start): średnia 0.43

Ciepły vs zimny start - koszt: ujemna delta oznacza przewagę ciepłego startu
- Brak pełnych par ciepły/zimny start dla wskazanych algorytmów w danych

Statystyki szczegółowe zapisano w katalogu tables/