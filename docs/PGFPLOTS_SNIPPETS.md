# PGFPlots Snippets -- ready to copy into your LaTeX

Poniżej znajdziesz minimalne, sprawdzone w praktyce fragmenty LaTeX (PGFPlots/pgfplotstable) do rysowania wykresów bezpośrednio z CSV generowanych w `results/**`.

W preambule dodaj (raz):

```tex
% Preamble (once)
\usepackage{pgfplots}
\usepackage{pgfplotstable}
\usepackage{booktabs}
\pgfplotsset{compat=1.18}
```

## 1) Cost vs n (z błędami 95% CI) -- pojedynczy algorytm / graf / licencja

Źródło danych: `results/benchmark_all/all/aggregates.csv` (analogicznie dla `benchmark_real_all`).
Kolumny: `algorithm, graph, n_nodes, license_config, cost_mean, cost_ci95, time_ms_mean, ...`.

```tex
% Wczytanie danych
\pgfplotstableread[col sep=comma]{results/benchmark_all/all/aggregates.csv}\agg

% Definicje filtrów (używamy \pdfstrcmp do dopasowania napisów)
\pgfplotsset{
  row filter/.code={%
    % Zmień wartości w nawiasach na swoje kryteria
    \def\ALG{GreedyAlgorithm}%
    \def\GRA{random}%
    \def\LIC{duolingo_super}%
    % warunek logiczny: (algorithm==ALG && graph==GRA && license_config==LIC)
    \pgfmathparse{(
      (\pdfstrcmp{\thisrow{algorithm}}{\ALG}==0) *
      (\pdfstrcmp{\thisrow{graph}}{\GRA}==0) *
      (\pdfstrcmp{\thisrow{license_config}}{\LIC}==0)
    )}
    \ifnum\pgfmathresult=1\relax\else\pgfplotstablerowfalse\fi
  }
}

\begin{tikzpicture}
  \begin{axis}[
    width=\linewidth, height=6cm,
    xlabel={$n$}, ylabel={cost},
    legend pos=north west,
    ymin=0,
  ]
    % Błąd: y explicit = cost_ci95
    \addplot+[only marks, mark=*, error bars/.cd, y dir=both, y explicit]
      table[
        x=n_nodes,
        y=cost_mean,
        y error=cost_ci95,
        row predicate/.code=\pgfplotsutilifundefined{row filter}{\relax}{\pgfkeysalso{/pgfplots/row filter}}%
      ] {\agg};
    \addlegendentry{Greedy -- random -- duolingo}
  \end{axis}
\end{tikzpicture}
```

Wskazówka: aby wykreślić kilka serii na jednym wykresie, powtórz `\addplot` z innymi wartościami `\ALG/\GRA/\LIC` i dopisz `\addlegendentry{...}`.

## 2) Time vs n (skala log) -- wiele algorytmów, to samo `graph` i `license`

```tex
\pgfplotstableread[col sep=comma]{results/benchmark_all/all/aggregates.csv}\agg
\def\GRA{random}
\def\LIC{duolingo_super}

\begin{tikzpicture}
  \begin{axis}[
    width=\linewidth, height=6cm,
    xlabel={$n$}, ylabel={time [ms]},
    ymode=log,
    legend pos=north west,
  ]
    % Greedy
    \pgfplotsset{row filter/.code={%
      \def\ALG{GreedyAlgorithm}%
      \pgfmathparse{(
        (\pdfstrcmp{\thisrow{algorithm}}{\ALG}==0) *
        (\pdfstrcmp{\thisrow{graph}}{\GRA}==0) *
        (\pdfstrcmp{\thisrow{license_config}}{\LIC}==0)
      )}
      \ifnum\pgfmathresult=1\relax\else\pgfplotstablerowfalse\fi
    }}
    \addplot+[mark=o]
      table[x=n_nodes, y=time_ms_mean, row predicate/.code=\pgfplotsutilifundefined{row filter}{\relax}{\pgfkeysalso{/pgfplots/row filter}}]{\agg};
    \addlegendentry{Greedy}

    % Randomized
    \pgfplotsset{row filter/.code={%
      \def\ALG{RandomizedAlgorithm}%
      \pgfmathparse{(
        (\pdfstrcmp{\thisrow{algorithm}}{\ALG}==0) *
        (\pdfstrcmp{\thisrow{graph}}{\GRA}==0) *
        (\pdfstrcmp{\thisrow{license_config}}{\LIC}==0)
      )}
      \ifnum\pgfmathresult=1\relax\else\pgfplotstablerowfalse\fi
    }}
    \addplot+[mark=triangle*]
      table[x=n_nodes, y=time_ms_mean, row predicate/.code=\pgfplotsutilifundefined{row filter}{\relax}{\pgfkeysalso{/pgfplots/row filter}}]{\agg};
    \addlegendentry{Randomized}
  \end{axis}
\end{tikzpicture}
```

## 3) Wykres słupkowy -- empiryczna złożoność czasu (nachylenie b z log-log)

Źródło: `results/**/all/time_scaling.csv` z kolumną `slope_b` (b w `time ≈ a·n^b`).

```tex
\pgfplotstableread[col sep=comma]{results/benchmark_all/all/time_scaling.csv}\ts

\begin{tikzpicture}
  \begin{axis}[
    width=\linewidth, height=7cm,
    ybar, ylabel={$b$ w $t\approx a n^b$},
    symbolic x coords={GreedyAlgorithm,RandomizedAlgorithm,TabuSearch,SimulatedAnnealing,GeneticAlgorithm,AntColonyOptimization,ILPSolver,DominatingSetAlgorithm},
    xtick=data,
    x tick label style={rotate=30, anchor=east},
    ymin=0,
  ]
    % Dla prostoty: weź medianę b per algorytm - można też filtrować po graph/licence
    % W PGFPlots łatwiej przygotować pre-zredukowane dane, ale można też dodać filtr podobny jak wyżej.
    % Poniższy przykład zakłada, że mamy jedną wartość b per algorytm (np. medianę).
    % Jeśli chcesz automatycznej mediany, zrób to w CSV lub pre-obróbce.
    \addplot+[
      ybar,
    ] table[x=algorithm, y=slope_b]{\ts};
  \end{axis}
\end{tikzpicture}
```

## 4) Tabela CSV → LaTeX (opcjonalnie, bez figury)

```tex
% Szybkie wstawienie tabeli z PGFPlotstable (np. fragment aggregates)
\pgfplotstabletypeset[
  col sep=comma,
  columns={algorithm,graph,n_nodes,cost_mean,cost_ci95,time_ms_mean,time_ms_ci95},
  columns/algorithm/.style={string type},
  columns/graph/.style={string type},
  every head row/.style={before row=\toprule, after row=\midrule},
  every last row/.style={after row=},
]{results/benchmark_all/all/aggregates.csv}
```

## Uwagi praktyczne
- Jeżeli filtrowanie w TeXu wydaje się zbyt rozbudowane: przefiltruj CSV wcześniej (pandas) lub wyprodukuj pomocnicze `.dat` z jedną serią -- wtedy `\addplot table[x=...,y=...]` będzie trywialne.
- Dla skali log pamiętaj o dodatnich wartościach (czas w ms nie może być 0). Nasze CSV generują wartości >0.
- Starsze wersje PGFPlots mogą wymagać `compat=1.17` lub innego -- dostosuj do swojego środowiska.

To wszystko -- skopiuj wybrany fragment, podmień `ALG/GRA/LIC` oraz ścieżki do CSV i buduj figury bezpośrednio z naszych danych.

