TC002 Move third-party import `networkx` into a type-checking block
 --> src/glopt/algorithms/ant_colony.py:6:20
  |
4 | from typing import Any
5 |
6 | import networkx as nx
  |                    ^^
7 |
8 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  |
help: Move into type-checking block

TID252 Prefer absolute imports over relative imports from parent modules
  --> src/glopt/algorithms/ant_colony.py:8:1
   |
 6 | import networkx as nx
 7 |
 8 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 9 | from ..core.solution_validator import SolutionValidator
10 | from .greedy import GreedyAlgorithm
   |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
  --> src/glopt/algorithms/ant_colony.py:8:1
   |
 6 | import networkx as nx
 7 |
 8 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 9 | from ..core.solution_validator import SolutionValidator
10 | from .greedy import GreedyAlgorithm
   |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
  --> src/glopt/algorithms/ant_colony.py:8:1
   |
 6 | import networkx as nx
 7 |
 8 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 9 | from ..core.solution_validator import SolutionValidator
10 | from .greedy import GreedyAlgorithm
   |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
  --> src/glopt/algorithms/ant_colony.py:8:1
   |
 6 | import networkx as nx
 7 |
 8 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 9 | from ..core.solution_validator import SolutionValidator
10 | from .greedy import GreedyAlgorithm
   |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
  --> src/glopt/algorithms/ant_colony.py:9:1
   |
 8 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
 9 | from ..core.solution_validator import SolutionValidator
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
10 | from .greedy import GreedyAlgorithm
   |
help: Replace relative imports from parent modules with absolute imports

PLR0913 Too many arguments in function definition (6 > 5)
  --> src/glopt/algorithms/ant_colony.py:20:9
   |
18 |         return "ant_colony_optimization"
19 |
20 |     def __init__(self, alpha=1.0, beta=2.0, evaporation=0.5, q0=0.9, num_ants=20, max_iterations=100) -> None:
   |         ^^^^^^^^
21 |         self.alpha = alpha
22 |         self.beta = beta
   |

D107 Missing docstring in `__init__`
  --> src/glopt/algorithms/ant_colony.py:20:9
   |
18 |         return "ant_colony_optimization"
19 |
20 |     def __init__(self, alpha=1.0, beta=2.0, evaporation=0.5, q0=0.9, num_ants=20, max_iterations=100) -> None:
   |         ^^^^^^^^
21 |         self.alpha = alpha
22 |         self.beta = beta
   |

ANN001 Missing type annotation for function argument `alpha`
  --> src/glopt/algorithms/ant_colony.py:20:24
   |
18 |         return "ant_colony_optimization"
19 |
20 |     def __init__(self, alpha=1.0, beta=2.0, evaporation=0.5, q0=0.9, num_ants=20, max_iterations=100) -> None:
   |                        ^^^^^
21 |         self.alpha = alpha
22 |         self.beta = beta
   |

ANN001 Missing type annotation for function argument `beta`
  --> src/glopt/algorithms/ant_colony.py:20:35
   |
18 |         return "ant_colony_optimization"
19 |
20 |     def __init__(self, alpha=1.0, beta=2.0, evaporation=0.5, q0=0.9, num_ants=20, max_iterations=100) -> None:
   |                                   ^^^^
21 |         self.alpha = alpha
22 |         self.beta = beta
   |

ANN001 Missing type annotation for function argument `evaporation`
  --> src/glopt/algorithms/ant_colony.py:20:45
   |
18 |         return "ant_colony_optimization"
19 |
20 |     def __init__(self, alpha=1.0, beta=2.0, evaporation=0.5, q0=0.9, num_ants=20, max_iterations=100) -> None:
   |                                             ^^^^^^^^^^^
21 |         self.alpha = alpha
22 |         self.beta = beta
   |

ANN001 Missing type annotation for function argument `q0`
  --> src/glopt/algorithms/ant_colony.py:20:62
   |
18 |         return "ant_colony_optimization"
19 |
20 |     def __init__(self, alpha=1.0, beta=2.0, evaporation=0.5, q0=0.9, num_ants=20, max_iterations=100) -> None:
   |                                                              ^^
21 |         self.alpha = alpha
22 |         self.beta = beta
   |

ANN001 Missing type annotation for function argument `num_ants`
  --> src/glopt/algorithms/ant_colony.py:20:70
   |
18 |         return "ant_colony_optimization"
19 |
20 |     def __init__(self, alpha=1.0, beta=2.0, evaporation=0.5, q0=0.9, num_ants=20, max_iterations=100) -> None:
   |                                                                      ^^^^^^^^
21 |         self.alpha = alpha
22 |         self.beta = beta
   |

ANN001 Missing type annotation for function argument `max_iterations`
  --> src/glopt/algorithms/ant_colony.py:20:83
   |
18 |         return "ant_colony_optimization"
19 |
20 |     def __init__(self, alpha=1.0, beta=2.0, evaporation=0.5, q0=0.9, num_ants=20, max_iterations=100) -> None:
   |                                                                                   ^^^^^^^^^^^^^^
21 |         self.alpha = alpha
22 |         self.beta = beta
   |

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `**_`
  --> src/glopt/algorithms/ant_colony.py:29:77
   |
27 |         self.validator = SolutionValidator(debug=False)
28 |
29 |     def solve(self, graph: nx.Graph, license_types: list[LicenseType], **_: Any) -> Solution:
   |                                                                             ^^^
30 |         pher = self._init_pher(graph, license_types)
31 |         heur = self._init_heur(graph, license_types)
   |

N803 Argument name `G` should be lowercase
  --> src/glopt/algorithms/ant_colony.py:56:15
   |
55 |     def _construct(
56 |         self, G: nx.Graph, lts: list[LicenseType], pher: dict[PKey, float], heur: dict[PKey, float]
   |               ^^^^^^^^^^^
57 |     ) -> Solution:
58 |         uncovered: set[Any] = set(G.nodes())
   |

N803 Argument name `G` should be lowercase
  --> src/glopt/algorithms/ant_colony.py:89:110
   |
88 |     def _select_owner(
89 |         self, uncovered: set[Any], lts: list[LicenseType], pher: dict[PKey, float], heur: dict[PKey, float], G: nx.Graph
   |                                                                                                              ^^^^^^^^^^^
90 |     ) -> Any | None:
91 |         if not uncovered:
   |

ARG002 Unused method argument: `G`
  --> src/glopt/algorithms/ant_colony.py:89:110
   |
88 |     def _select_owner(
89 |         self, uncovered: set[Any], lts: list[LicenseType], pher: dict[PKey, float], heur: dict[PKey, float], G: nx.Graph
   |                                                                                                              ^
90 |     ) -> Any | None:
91 |         if not uncovered:
   |

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `_select_owner`
  --> src/glopt/algorithms/ant_colony.py:90:10
   |
88 |     def _select_owner(
89 |         self, uncovered: set[Any], lts: list[LicenseType], pher: dict[PKey, float], heur: dict[PKey, float], G: nx.Graph
90 |     ) -> Any | None:
   |          ^^^^^^^^^^
91 |         if not uncovered:
92 |             return None
   |

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `owner`
   --> src/glopt/algorithms/ant_colony.py:104:22
    |
103 |     def _select_license(
104 |         self, owner: Any, lts: list[LicenseType], pher: dict[PKey, float], heur: dict[PKey, float]
    |                      ^^^
105 |     ) -> LicenseType | None:
106 |         if not lts:
    |

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `_roulette_or_best`
   --> src/glopt/algorithms/ant_colony.py:114:82
    |
112 |         return self._roulette_or_best(lts, scores)
113 |
114 |     def _roulette_or_best(self, choices: list[Any], scores: dict[Any, float]) -> Any:
    |                                                                                  ^^^
115 |         if not choices:
116 |             return None
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/algorithms/ant_colony.py:117:12
    |
115 |         if not choices:
116 |             return None
117 |         if random.random() < self.q0:
    |            ^^^^^^^^^^^^^^^
118 |             return max(choices, key=lambda c: scores.get(c, 0.0))
119 |         total = sum(max(0.0, scores.get(c, 0.0)) for c in choices)
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/algorithms/ant_colony.py:121:20
    |
119 |         total = sum(max(0.0, scores.get(c, 0.0)) for c in choices)
120 |         if total <= 0:
121 |             return random.choice(choices)
    |                    ^^^^^^^^^^^^^^^^^^^^^^
122 |         r = random.uniform(0, total)
123 |         acc = 0.0
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/algorithms/ant_colony.py:122:13
    |
120 |         if total <= 0:
121 |             return random.choice(choices)
122 |         r = random.uniform(0, total)
    |             ^^^^^^^^^^^^^^^^^^^^^^^^
123 |         acc = 0.0
124 |         for c in choices:
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/algorithms/ant_colony.py:128:16
    |
126 |             if acc >= r:
127 |                 return c
128 |         return random.choice(choices)
    |                ^^^^^^^^^^^^^^^^^^^^^^
129 |
130 |     def _init_pher(self, G: nx.Graph, lts: list[LicenseType]) -> dict[PKey, float]:
    |

N803 Argument name `G` should be lowercase
   --> src/glopt/algorithms/ant_colony.py:130:26
    |
128 |         return random.choice(choices)
129 |
130 |     def _init_pher(self, G: nx.Graph, lts: list[LicenseType]) -> dict[PKey, float]:
    |                          ^^^^^^^^^^^
131 |         return {(n, lt.name): 1.0 for n in G.nodes() for lt in lts}
    |

N803 Argument name `G` should be lowercase
   --> src/glopt/algorithms/ant_colony.py:133:26
    |
131 |         return {(n, lt.name): 1.0 for n in G.nodes() for lt in lts}
132 |
133 |     def _init_heur(self, G: nx.Graph, lts: list[LicenseType]) -> dict[PKey, float]:
    |                          ^^^^^^^^^^^
134 |         h: dict[PKey, float] = {}
135 |         for n in G.nodes():
    |

N803 Argument name `G` should be lowercase
   --> src/glopt/algorithms/ant_colony.py:157:36
    |
155 |                     pher[k] += q
156 |
157 |     def _fallback_singletons(self, G: nx.Graph, lts: list[LicenseType]) -> Solution:
    |                                    ^^^^^^^^^^^
158 |         lt1 = min([x for x in lts if x.min_capacity <= 1] or lts, key=lambda x: x.cost)
159 |         groups = [LicenseGroup(lt1, n, frozenset()) for n in G.nodes()]
    |

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/dominating_set.py:5:1
  |
3 | import networkx as nx
4 |
5 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
6 | from ..core.solution_builder import SolutionBuilder
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/dominating_set.py:5:1
  |
3 | import networkx as nx
4 |
5 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
6 | from ..core.solution_builder import SolutionBuilder
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/dominating_set.py:5:1
  |
3 | import networkx as nx
4 |
5 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
6 | from ..core.solution_builder import SolutionBuilder
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/dominating_set.py:5:1
  |
3 | import networkx as nx
4 |
5 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
6 | from ..core.solution_builder import SolutionBuilder
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/dominating_set.py:6:1
  |
5 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
6 | from ..core.solution_builder import SolutionBuilder
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
help: Replace relative imports from parent modules with absolute imports

ARG002 Unused method argument: `kwargs`
  --> src/glopt/algorithms/dominating_set.py:14:74
   |
12 |         return "dominating_set_algorithm"
13 |
14 |     def solve(self, graph: nx.Graph, license_types: list[LicenseType], **kwargs: Any) -> Solution:
   |                                                                          ^^^^^^
15 |         if len(graph.nodes()) == 0:
16 |             return Solution(groups=())
   |

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `**kwargs`
  --> src/glopt/algorithms/dominating_set.py:14:82
   |
12 |         return "dominating_set_algorithm"
13 |
14 |     def solve(self, graph: nx.Graph, license_types: list[LicenseType], **kwargs: Any) -> Solution:
   |                                                                                  ^^^
15 |         if len(graph.nodes()) == 0:
16 |             return Solution(groups=())
   |

SIM108 Use ternary operator `score = len(coverage) / min_cost_per_node if min_cost_per_node > 0 else len(coverage)` instead of `if`-`else`-block
  --> src/glopt/algorithms/dominating_set.py:87:17
   |
85 |                   min_cost_per_node = self._calculate_min_cost_per_node(len(coverage), license_types)
86 |
87 | /                 if min_cost_per_node > 0:
88 | |                     score = len(coverage) / min_cost_per_node
89 | |                 else:
90 | |                     score = len(coverage)
   | |_________________________________________^
91 |
92 |                   if score > best_score:
   |
help: Replace `if`-`else`-block with `score = len(coverage) / min_cost_per_node if min_cost_per_node > 0 else len(coverage)`

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `owner`
   --> src/glopt/algorithms/dominating_set.py:117:22
    |
116 |     def _find_best_cost_assignment(
117 |         self, owner: Any, available_nodes: set[Any], license_types: list[LicenseType]
    |                      ^^^
118 |     ) -> tuple[LicenseType, set[Any]]:
119 |         best_assignment = None
    |

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `owner`
   --> src/glopt/algorithms/dominating_set.py:140:49
    |
138 |         return best_assignment
139 |
140 |     def _select_best_group_members(self, owner: Any, available_nodes: set[Any], target_size: int) -> set[Any]:
    |                                                 ^^^
141 |         if target_size <= 0:
142 |             return set()
    |

ARG005 Unused lambda argument: `n`
   --> src/glopt/algorithms/dominating_set.py:152:36
    |
150 |         candidates = list(available_nodes - {owner})
151 |
152 |         candidates.sort(key=lambda n: len(available_nodes), reverse=True)
    |                                    ^
153 |
154 |         group_members.update(candidates[:remaining_slots])
    |

TC003 Move standard library import `collections.abc.Sequence` into a type-checking block
 --> src/glopt/algorithms/genetic.py:4:29
  |
3 | import random
4 | from collections.abc import Sequence
  |                             ^^^^^^^^
5 | from typing import Any
  |
help: Move into type-checking block

TC002 Move third-party import `networkx` into a type-checking block
 --> src/glopt/algorithms/genetic.py:7:20
  |
5 | from typing import Any
6 |
7 | import networkx as nx
  |                    ^^
8 |
9 | from ..core import Algorithm, LicenseType, Solution
  |
help: Move into type-checking block

TID252 Prefer absolute imports over relative imports from parent modules
  --> src/glopt/algorithms/genetic.py:9:1
   |
 7 | import networkx as nx
 8 |
 9 | from ..core import Algorithm, LicenseType, Solution
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
10 | from ..core.mutations import MutationOperators
11 | from ..core.solution_validator import SolutionValidator
   |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
  --> src/glopt/algorithms/genetic.py:9:1
   |
 7 | import networkx as nx
 8 |
 9 | from ..core import Algorithm, LicenseType, Solution
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
10 | from ..core.mutations import MutationOperators
11 | from ..core.solution_validator import SolutionValidator
   |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
  --> src/glopt/algorithms/genetic.py:9:1
   |
 7 | import networkx as nx
 8 |
 9 | from ..core import Algorithm, LicenseType, Solution
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
10 | from ..core.mutations import MutationOperators
11 | from ..core.solution_validator import SolutionValidator
   |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
  --> src/glopt/algorithms/genetic.py:10:1
   |
 9 | from ..core import Algorithm, LicenseType, Solution
10 | from ..core.mutations import MutationOperators
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
11 | from ..core.solution_validator import SolutionValidator
12 | from .randomized import RandomizedAlgorithm
   |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
  --> src/glopt/algorithms/genetic.py:11:1
   |
 9 | from ..core import Algorithm, LicenseType, Solution
10 | from ..core.mutations import MutationOperators
11 | from ..core.solution_validator import SolutionValidator
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
12 | from .randomized import RandomizedAlgorithm
   |
help: Replace relative imports from parent modules with absolute imports

D107 Missing docstring in `__init__`
  --> src/glopt/algorithms/genetic.py:16:9
   |
15 | class GeneticAlgorithm(Algorithm):
16 |     def __init__(
   |         ^^^^^^^^
17 |         self,
18 |         population_size: int = 30,
   |

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `**_`
  --> src/glopt/algorithms/genetic.py:37:14
   |
35 |         graph: nx.Graph,
36 |         license_types: Sequence[LicenseType],
37 |         **_: Any,
   |              ^^^
38 |     ) -> Solution:
39 |         if self.seed is not None:
   |

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/greedy.py:5:1
  |
3 | import networkx as nx
4 |
5 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
6 | from ..core.solution_builder import SolutionBuilder
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/greedy.py:5:1
  |
3 | import networkx as nx
4 |
5 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
6 | from ..core.solution_builder import SolutionBuilder
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/greedy.py:5:1
  |
3 | import networkx as nx
4 |
5 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
6 | from ..core.solution_builder import SolutionBuilder
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/greedy.py:5:1
  |
3 | import networkx as nx
4 |
5 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
6 | from ..core.solution_builder import SolutionBuilder
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/greedy.py:6:1
  |
5 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
6 | from ..core.solution_builder import SolutionBuilder
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
help: Replace relative imports from parent modules with absolute imports

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `**_`
  --> src/glopt/algorithms/greedy.py:18:14
   |
16 |         graph: nx.Graph,
17 |         license_types: list[LicenseType],
18 |         **_: Any,
   |              ^^^
19 |     ) -> Solution:
20 |         licenses = sorted(license_types, key=lambda lt: (-lt.max_capacity, lt.cost))
   |

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `owner`
  --> src/glopt/algorithms/greedy.py:62:16
   |
60 |     def _best_group_for_owner(
61 |         self,
62 |         owner: Any,
   |                ^^^
63 |         avail: set[Any],
64 |         graph: nx.Graph,
   |

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `owner`
  --> src/glopt/algorithms/greedy.py:92:16
   |
90 |     def _cheapest_feasible_group(
91 |         self,
92 |         owner: Any,
   |                ^^^
93 |         avail: set[Any],
94 |         graph: nx.Graph,
   |

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/ilp.py:6:1
  |
4 | import pulp
5 |
6 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/ilp.py:6:1
  |
4 | import pulp
5 |
6 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/ilp.py:6:1
  |
4 | import pulp
5 |
6 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/ilp.py:6:1
  |
4 | import pulp
5 |
6 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
help: Replace relative imports from parent modules with absolute imports

C901 `solve` is too complex (19 > 10)
  --> src/glopt/algorithms/ilp.py:14:9
   |
12 |         return "ilp"
13 |
14 |     def solve(
   |         ^^^^^
15 |         self,
16 |         graph: nx.Graph,
   |

PLR0912 Too many branches (18 > 12)
  --> src/glopt/algorithms/ilp.py:14:9
   |
12 |         return "ilp"
13 |
14 |     def solve(
   |         ^^^^^
15 |         self,
16 |         graph: nx.Graph,
   |

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `**kwargs`
  --> src/glopt/algorithms/ilp.py:18:19
   |
16 |         graph: nx.Graph,
17 |         license_types: list[LicenseType],
18 |         **kwargs: Any,
   |                   ^^^
19 |     ) -> Solution:
20 |         time_limit: int | None = kwargs.get("time_limit")
   |

TRY003 Avoid specifying long messages outside the exception class
  --> src/glopt/algorithms/ilp.py:65:19
   |
64 |         if model.status != pulp.LpStatusOptimal:
65 |             raise RuntimeError(f"ilp solver failed with status {pulp.LpStatus[model.status]}")
   |                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
66 |
67 |         groups: list[LicenseGroup] = []
   |

EM102 Exception must not use an f-string literal, assign to variable first
  --> src/glopt/algorithms/ilp.py:65:32
   |
64 |         if model.status != pulp.LpStatusOptimal:
65 |             raise RuntimeError(f"ilp solver failed with status {pulp.LpStatus[model.status]}")
   |                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
66 |
67 |         groups: list[LicenseGroup] = []
   |
help: Assign to variable; remove f-string literal

PLR2004 Magic value used in comparison, consider replacing `0.5` with a constant variable
  --> src/glopt/algorithms/ilp.py:70:88
   |
68 |         for i in nodes:
69 |             for t_idx, lt in enumerate(license_types):
70 |                 if active_vars[i, t_idx].varValue and active_vars[i, t_idx].varValue > 0.5:
   |                                                                                        ^^^
71 |                     members: set[Any] = set()
72 |                     for j in set(graph.neighbors(i)) | {i}:
   |

PLR2004 Magic value used in comparison, consider replacing `0.5` with a constant variable
  --> src/glopt/algorithms/ilp.py:74:68
   |
72 |                     for j in set(graph.neighbors(i)) | {i}:
73 |                         var = assign_vars.get((i, j, t_idx))
74 |                         if var and var.varValue and var.varValue > 0.5:
   |                                                                    ^^^
75 |                             members.add(j)
76 |                     if members:
   |

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/naive.py:7:1
  |
5 | import networkx as nx
6 |
7 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
8 | from ..core.solution_builder import SolutionBuilder
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/naive.py:7:1
  |
5 | import networkx as nx
6 |
7 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
8 | from ..core.solution_builder import SolutionBuilder
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/naive.py:7:1
  |
5 | import networkx as nx
6 |
7 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
8 | from ..core.solution_builder import SolutionBuilder
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/naive.py:7:1
  |
5 | import networkx as nx
6 |
7 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
8 | from ..core.solution_builder import SolutionBuilder
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
  --> src/glopt/algorithms/naive.py:8:1
   |
 7 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
 8 | from ..core.solution_builder import SolutionBuilder
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 9 |
10 | Assignment = list[tuple[LicenseType, Any, set[Any]]]
   |
help: Replace relative imports from parent modules with absolute imports

ARG002 Unused method argument: `kwargs`
  --> src/glopt/algorithms/naive.py:22:11
   |
20 |         graph: nx.Graph,
21 |         license_types: Sequence[LicenseType],
22 |         **kwargs: Any,
   |           ^^^^^^
23 |     ) -> Solution:
24 |         nodes: list[Any] = list(graph.nodes())
   |

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `**kwargs`
  --> src/glopt/algorithms/naive.py:22:19
   |
20 |         graph: nx.Graph,
21 |         license_types: Sequence[LicenseType],
22 |         **kwargs: Any,
   |                   ^^^
23 |     ) -> Solution:
24 |         nodes: list[Any] = list(graph.nodes())
   |

PLR2004 Magic value used in comparison, consider replacing `10` with a constant variable
  --> src/glopt/algorithms/naive.py:27:16
   |
25 |         n = len(nodes)
26 |
27 |         if n > 10:
   |                ^^
28 |             raise ValueError(f"graph too large for naive algorithm: {n} nodes > 10")
   |

TRY003 Avoid specifying long messages outside the exception class
  --> src/glopt/algorithms/naive.py:28:19
   |
27 |         if n > 10:
28 |             raise ValueError(f"graph too large for naive algorithm: {n} nodes > 10")
   |                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
29 |
30 |         if n == 0:
   |

EM102 Exception must not use an f-string literal, assign to variable first
  --> src/glopt/algorithms/naive.py:28:30
   |
27 |         if n > 10:
28 |             raise ValueError(f"graph too large for naive algorithm: {n} nodes > 10")
   |                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
29 |
30 |         if n == 0:
   |
help: Assign to variable; remove f-string literal

RUF005 Consider `[{first}, *smaller]` instead of concatenation
  --> src/glopt/algorithms/naive.py:75:19
   |
73 |         first, rest = nodes[0], nodes[1:]
74 |         for smaller in self._generate_partitions(rest):
75 |             yield [{first}] + smaller
   |                   ^^^^^^^^^^^^^^^^^^^
76 |
77 |             for i, block in enumerate(smaller):
   |
help: Replace with `[{first}, *smaller]`

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `owner`
   --> src/glopt/algorithms/naive.py:109:38
    |
107 |                 yield list(combo)
108 |
109 |     def _is_valid_group(self, owner: Any, members: set[Any], graph: nx.Graph) -> bool:
    |                                      ^^^
110 |         owner_neighbors = set(graph.neighbors(owner))
111 |         return all(m in owner_neighbors for m in members)
    |

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/randomized.py:6:1
  |
4 | import networkx as nx
5 |
6 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
7 | from ..core.solution_builder import SolutionBuilder
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/randomized.py:6:1
  |
4 | import networkx as nx
5 |
6 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
7 | from ..core.solution_builder import SolutionBuilder
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/randomized.py:6:1
  |
4 | import networkx as nx
5 |
6 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
7 | from ..core.solution_builder import SolutionBuilder
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/randomized.py:6:1
  |
4 | import networkx as nx
5 |
6 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
7 | from ..core.solution_builder import SolutionBuilder
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/randomized.py:7:1
  |
6 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
7 | from ..core.solution_builder import SolutionBuilder
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
help: Replace relative imports from parent modules with absolute imports

ANN204 Missing return type annotation for special method `__init__`
  --> src/glopt/algorithms/randomized.py:15:9
   |
13 |         return "randomized_algorithm"
14 |
15 |     def __init__(self, greedy_probability: float = 0.7, seed: int | None = None):
   |         ^^^^^^^^
16 |         self.greedy_probability = max(0.0, min(1.0, greedy_probability))
17 |         self.seed = seed
   |
help: Add return type annotation: `None`

D107 Missing docstring in `__init__`
  --> src/glopt/algorithms/randomized.py:15:9
   |
13 |         return "randomized_algorithm"
14 |
15 |     def __init__(self, greedy_probability: float = 0.7, seed: int | None = None):
   |         ^^^^^^^^
16 |         self.greedy_probability = max(0.0, min(1.0, greedy_probability))
17 |         self.seed = seed
   |

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `**kwargs`
  --> src/glopt/algorithms/randomized.py:21:82
   |
19 |             random.seed(seed)
20 |
21 |     def solve(self, graph: nx.Graph, license_types: list[LicenseType], **kwargs: Any) -> Solution:
   |                                                                                  ^^^
22 |         if len(graph.nodes()) == 0:
23 |             return Solution(groups=())
   |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
  --> src/glopt/algorithms/randomized.py:39:26
   |
37 |                 continue
38 |
39 |             use_greedy = random.random() < self.greedy_probability
   |                          ^^^^^^^^^^^^^^^
40 |
41 |             if use_greedy:
   |

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `node`
  --> src/glopt/algorithms/randomized.py:63:15
   |
61 |     def _greedy_assignment(
62 |         self,
63 |         node: Any,
   |               ^^^
64 |         uncovered_nodes: set[Any],
65 |         graph: nx.Graph,
   |

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `node`
  --> src/glopt/algorithms/randomized.py:94:15
   |
92 |     def _random_assignment(
93 |         self,
94 |         node: Any,
   |               ^^^
95 |         uncovered_nodes: set[Any],
96 |         graph: nx.Graph,
   |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/algorithms/randomized.py:115:26
    |
113 |                 continue
114 |
115 |             group_size = random.randint(license_type.min_capacity, max_possible_size)
    |                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
116 |
117 |             group_members = self._select_random_group_members(node, available_nodes, group_size)
    |

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `owner`
   --> src/glopt/algorithms/randomized.py:125:22
    |
124 |     def _select_greedy_group_members(
125 |         self, owner: Any, available_nodes: set[Any], target_size: int, graph: nx.Graph
    |                      ^^^
126 |     ) -> set[Any]:
127 |         if target_size <= 0:
    |

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `owner`
   --> src/glopt/algorithms/randomized.py:143:51
    |
141 |         return group_members
142 |
143 |     def _select_random_group_members(self, owner: Any, available_nodes: set[Any], target_size: int) -> set[Any]:
    |                                                   ^^^
144 |         if target_size <= 0:
145 |             return set()
    |

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/simulated_annealing.py:7:1
  |
5 | import networkx as nx
6 |
7 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
8 | from ..core.solution_builder import SolutionBuilder
9 | from ..core.solution_validator import SolutionValidator
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/simulated_annealing.py:7:1
  |
5 | import networkx as nx
6 |
7 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
8 | from ..core.solution_builder import SolutionBuilder
9 | from ..core.solution_validator import SolutionValidator
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/simulated_annealing.py:7:1
  |
5 | import networkx as nx
6 |
7 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
8 | from ..core.solution_builder import SolutionBuilder
9 | from ..core.solution_validator import SolutionValidator
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/simulated_annealing.py:7:1
  |
5 | import networkx as nx
6 |
7 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
8 | from ..core.solution_builder import SolutionBuilder
9 | from ..core.solution_validator import SolutionValidator
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/simulated_annealing.py:8:1
  |
7 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
8 | from ..core.solution_builder import SolutionBuilder
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
9 | from ..core.solution_validator import SolutionValidator
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/simulated_annealing.py:9:1
  |
7 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
8 | from ..core.solution_builder import SolutionBuilder
9 | from ..core.solution_validator import SolutionValidator
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
help: Replace relative imports from parent modules with absolute imports

D107 Missing docstring in `__init__`
  --> src/glopt/algorithms/simulated_annealing.py:17:9
   |
15 |         return "simulated_annealing"
16 |
17 |     def __init__(
   |         ^^^^^^^^
18 |         self,
19 |         initial_temperature: float = 100.0,
   |

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `**_`
  --> src/glopt/algorithms/simulated_annealing.py:32:77
   |
30 |         self.validator = SolutionValidator(debug=False)
31 |
32 |     def solve(self, graph: nx.Graph, license_types: list[LicenseType], **_: Any) -> Solution:
   |                                                                             ^^^
33 |         from .greedy import GreedyAlgorithm
   |

PLC0415 `import` should be at the top-level of a file
  --> src/glopt/algorithms/simulated_annealing.py:33:9
   |
32 |     def solve(self, graph: nx.Graph, license_types: list[LicenseType], **_: Any) -> Solution:
33 |         from .greedy import GreedyAlgorithm
   |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
34 |
35 |         current = GreedyAlgorithm().solve(graph, license_types)
   |

N806 Variable `T` in function should be lowercase
  --> src/glopt/algorithms/simulated_annealing.py:41:9
   |
40 |         best = current
41 |         T = self.initial_temperature
   |         ^
42 |         stall = 0
   |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
  --> src/glopt/algorithms/simulated_annealing.py:53:29
   |
51 |             else:
52 |                 d = neighbor.total_cost - current.total_cost
53 |                 if d < 0 or random.random() < math.exp(-d / max(T, 1e-12)):
   |                             ^^^^^^^^^^^^^^^
54 |                     current = neighbor
55 |                     if current.total_cost < best.total_cost:
   |

N806 Variable `T` in function should be lowercase
  --> src/glopt/algorithms/simulated_annealing.py:65:17
   |
63 |             if stall >= self.max_stall:
64 |                 stall = 0
65 |                 T = max(self.min_temperature, T * 0.5)
   |                 ^
66 |
67 |             T *= self.cooling_rate
   |

N806 Variable `T` in function should be lowercase
  --> src/glopt/algorithms/simulated_annealing.py:67:13
   |
65 |                 T = max(self.min_temperature, T * 0.5)
66 |
67 |             T *= self.cooling_rate
   |             ^
68 |
69 |         return best
   |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
  --> src/glopt/algorithms/simulated_annealing.py:85:18
   |
83 |         ]
84 |         for _ in range(12):
85 |             mv = random.choice(moves)
   |                  ^^^^^^^^^^^^^^^^^^^^
86 |             try:
87 |                 cand = mv(solution, graph, lts)
   |

BLE001 Do not catch blind exception: `Exception`
  --> src/glopt/algorithms/simulated_annealing.py:88:20
   |
86 |             try:
87 |                 cand = mv(solution, graph, lts)
88 |             except Exception:
   |                    ^^^^^^^^^
89 |                 cand = None
90 |             if cand:
   |

ARG002 Unused method argument: `graph`
  --> src/glopt/algorithms/simulated_annealing.py:96:54
   |
94 |         return None
95 |
96 |     def _mv_change_license(self, solution: Solution, graph: nx.Graph, lts: list[LicenseType]) -> Solution | None:
   |                                                      ^^^^^
97 |         if not solution.groups:
98 |             return None
   |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/algorithms/simulated_annealing.py:99:13
    |
 97 |         if not solution.groups:
 98 |             return None
 99 |         g = random.choice(solution.groups)
    |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
100 |         compat = SolutionBuilder.get_compatible_license_types(g.size, lts, exclude=g.license_type)
101 |         cheaper = [lt for lt in compat if lt.cost < g.license_type.cost]
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/algorithms/simulated_annealing.py:104:18
    |
102 |         if not cheaper:
103 |             return None
104 |         new_lt = random.choice(cheaper)
    |                  ^^^^^^^^^^^^^^^^^^^^^^
105 |
106 |         new_groups = [LicenseGroup(new_lt, g.owner, g.additional_members) if x is g else x for x in solution.groups]
    |

ARG002 Unused method argument: `lts`
   --> src/glopt/algorithms/simulated_annealing.py:109:68
    |
107 |         return Solution(groups=tuple(new_groups))
108 |
109 |     def _mv_move_member(self, solution: Solution, graph: nx.Graph, lts: list[LicenseType]) -> Solution | None:
    |                                                                    ^^^
110 |         donors = [g for g in solution.groups if g.additional_members and g.size > g.license_type.min_capacity]
111 |         if not donors:
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/algorithms/simulated_annealing.py:113:18
    |
111 |         if not donors:
112 |             return None
113 |         from_g = random.choice(donors)
    |                  ^^^^^^^^^^^^^^^^^^^^^
114 |         member = random.choice(list(from_g.additional_members))
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/algorithms/simulated_annealing.py:114:18
    |
112 |             return None
113 |         from_g = random.choice(donors)
114 |         member = random.choice(list(from_g.additional_members))
    |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
115 |
116 |         receivers = [g for g in solution.groups if g is not from_g and g.size < g.license_type.max_capacity]
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/algorithms/simulated_annealing.py:119:16
    |
117 |         if not receivers:
118 |             return None
119 |         to_g = random.choice(receivers)
    |                ^^^^^^^^^^^^^^^^^^^^^^^^
120 |
121 |         allowed = SolutionBuilder.get_owner_neighbors_with_self(graph, to_g.owner)
    |

ARG002 Unused method argument: `lts`
   --> src/glopt/algorithms/simulated_annealing.py:135:69
    |
133 |         return Solution(groups=tuple(new_groups))
134 |
135 |     def _mv_swap_members(self, solution: Solution, graph: nx.Graph, lts: list[LicenseType]) -> Solution | None:
    |                                                                     ^^^
136 |         if len(solution.groups) < 2:
137 |             return None
    |

PLR2004 Magic value used in comparison, consider replacing `2` with a constant variable
   --> src/glopt/algorithms/simulated_annealing.py:136:35
    |
135 |     def _mv_swap_members(self, solution: Solution, graph: nx.Graph, lts: list[LicenseType]) -> Solution | None:
136 |         if len(solution.groups) < 2:
    |                                   ^
137 |             return None
138 |         g1, g2 = random.sample(list(solution.groups), 2)
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/algorithms/simulated_annealing.py:144:14
    |
142 |         if not cand1 or not cand2:
143 |             return None
144 |         n1 = random.choice(cand1)
    |              ^^^^^^^^^^^^^^^^^^^^
145 |         n2 = random.choice(cand2)
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/algorithms/simulated_annealing.py:145:14
    |
143 |             return None
144 |         n1 = random.choice(cand1)
145 |         n2 = random.choice(cand2)
    |              ^^^^^^^^^^^^^^^^^^^^
146 |
147 |         if n1 not in SolutionBuilder.get_owner_neighbors_with_self(graph, g2.owner):
    |

PLR2004 Magic value used in comparison, consider replacing `2` with a constant variable
   --> src/glopt/algorithms/simulated_annealing.py:167:35
    |
166 |     def _mv_merge_groups(self, solution: Solution, graph: nx.Graph, lts: list[LicenseType]) -> Solution | None:
167 |         if len(solution.groups) < 2:
    |                                   ^
168 |             return None
169 |         g1, g2 = random.sample(list(solution.groups), 2)
    |

PLR2004 Magic value used in comparison, consider replacing `3` with a constant variable
   --> src/glopt/algorithms/simulated_annealing.py:178:63
    |
177 |     def _mv_split_group(self, solution: Solution, graph: nx.Graph, lts: list[LicenseType]) -> Solution | None:
178 |         splittable = [g for g in solution.groups if g.size >= 3]
    |                                                               ^
179 |         if not splittable:
180 |             return None
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/algorithms/simulated_annealing.py:181:13
    |
179 |         if not splittable:
180 |             return None
181 |         g = random.choice(splittable)
    |             ^^^^^^^^^^^^^^^^^^^^^^^^^
182 |         members = list(g.all_members)
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/algorithms/simulated_annealing.py:186:19
    |
184 |         for _ in range(4):
185 |             random.shuffle(members)
186 |             cut = random.randint(1, len(members) - 1)
    |                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
187 |             part1, part2 = members[:cut], members[cut:]
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/algorithms/simulated_annealing.py:194:22
    |
192 |                 continue
193 |
194 |             owner1 = random.choice(part1)
    |                      ^^^^^^^^^^^^^^^^^^^^
195 |             owner2 = random.choice(part2)
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/algorithms/simulated_annealing.py:195:22
    |
194 |             owner1 = random.choice(part1)
195 |             owner2 = random.choice(part2)
    |                      ^^^^^^^^^^^^^^^^^^^^
196 |
197 |             neigh1 = SolutionBuilder.get_owner_neighbors_with_self(graph, owner1)
    |

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/tabu_search.py:6:1
  |
4 | import networkx as nx
5 |
6 | from ..core import Algorithm, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
7 | from ..core.mutations import MutationOperators
8 | from ..core.solution_validator import SolutionValidator
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/tabu_search.py:6:1
  |
4 | import networkx as nx
5 |
6 | from ..core import Algorithm, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
7 | from ..core.mutations import MutationOperators
8 | from ..core.solution_validator import SolutionValidator
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/tabu_search.py:6:1
  |
4 | import networkx as nx
5 |
6 | from ..core import Algorithm, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
7 | from ..core.mutations import MutationOperators
8 | from ..core.solution_validator import SolutionValidator
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/tabu_search.py:7:1
  |
6 | from ..core import Algorithm, LicenseType, Solution
7 | from ..core.mutations import MutationOperators
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
8 | from ..core.solution_validator import SolutionValidator
9 | from .greedy import GreedyAlgorithm
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/tabu_search.py:8:1
  |
6 | from ..core import Algorithm, LicenseType, Solution
7 | from ..core.mutations import MutationOperators
8 | from ..core.solution_validator import SolutionValidator
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
9 | from .greedy import GreedyAlgorithm
  |
help: Replace relative imports from parent modules with absolute imports

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `**kwargs`
  --> src/glopt/algorithms/tabu_search.py:21:19
   |
19 |         graph: nx.Graph,
20 |         license_types: list[LicenseType],
21 |         **kwargs: Any,
   |                   ^^^
22 |     ) -> Solution:
23 |         max_iterations: int = kwargs.get("max_iterations", 1000)
   |

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/tree_dp.py:5:1
  |
3 | import networkx as nx
4 |
5 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
6 | from ..core.solution_builder import SolutionBuilder
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/tree_dp.py:5:1
  |
3 | import networkx as nx
4 |
5 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
6 | from ..core.solution_builder import SolutionBuilder
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/tree_dp.py:5:1
  |
3 | import networkx as nx
4 |
5 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
6 | from ..core.solution_builder import SolutionBuilder
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/tree_dp.py:5:1
  |
3 | import networkx as nx
4 |
5 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
6 | from ..core.solution_builder import SolutionBuilder
  |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
 --> src/glopt/algorithms/tree_dp.py:6:1
  |
5 | from ..core import Algorithm, LicenseGroup, LicenseType, Solution
6 | from ..core.solution_builder import SolutionBuilder
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |
help: Replace relative imports from parent modules with absolute imports

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `**_`
  --> src/glopt/algorithms/tree_dp.py:18:14
   |
16 |         graph: nx.Graph,
17 |         license_types: list[LicenseType],
18 |         **_: Any,
   |              ^^^
19 |     ) -> Solution:
20 |         if not nx.is_tree(graph):
   |

TRY003 Avoid specifying long messages outside the exception class
  --> src/glopt/algorithms/tree_dp.py:21:19
   |
19 |     ) -> Solution:
20 |         if not nx.is_tree(graph):
21 |             raise ValueError("TreeDynamicProgramming requires a tree graph")
   |                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
22 |
23 |         if len(graph.nodes()) == 0:
   |

EM101 Exception must not use a string literal, assign to variable first
  --> src/glopt/algorithms/tree_dp.py:21:30
   |
19 |     ) -> Solution:
20 |         if not nx.is_tree(graph):
21 |             raise ValueError("TreeDynamicProgramming requires a tree graph")
   |                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
22 |
23 |         if len(graph.nodes()) == 0:
   |
help: Assign to variable; remove string literal

RUF015 Prefer `next(iter(graph.nodes()))` over single element slice
  --> src/glopt/algorithms/tree_dp.py:27:20
   |
26 |         if len(graph.nodes()) == 1:
27 |             node = list(graph.nodes())[0]
   |                    ^^^^^^^^^^^^^^^^^^^^^^
28 |             cheapest = min(
29 |                 license_types, key=lambda lt: lt.cost if lt.min_capacity <= 1 <= lt.max_capacity else float("inf")
   |
help: Replace with `next(iter(graph.nodes()))`

RUF015 Prefer `next(iter(graph.nodes()))` over single element slice
  --> src/glopt/algorithms/tree_dp.py:34:16
   |
32 |             return SolutionBuilder.create_solution_from_groups([group])
33 |
34 |         root = list(graph.nodes())[0]
   |                ^^^^^^^^^^^^^^^^^^^^^^
35 |         memo = {}
36 |         cost, groups = self._solve_subtree(graph, root, None, license_types, memo)
   |
help: Replace with `next(iter(graph.nodes()))`

C901 `_solve_subtree` is too complex (13 > 10)
  --> src/glopt/algorithms/tree_dp.py:39:9
   |
37 |         return SolutionBuilder.create_solution_from_groups(groups)
38 |
39 |     def _solve_subtree(
   |         ^^^^^^^^^^^^^^
40 |         self, graph: nx.Graph, node: Any, parent: Any, license_types: list[LicenseType], memo: dict
41 |     ) -> tuple[float, list[LicenseGroup]]:
   |

PLR0912 Too many branches (13 > 12)
  --> src/glopt/algorithms/tree_dp.py:39:9
   |
37 |         return SolutionBuilder.create_solution_from_groups(groups)
38 |
39 |     def _solve_subtree(
   |         ^^^^^^^^^^^^^^
40 |         self, graph: nx.Graph, node: Any, parent: Any, license_types: list[LicenseType], memo: dict
41 |     ) -> tuple[float, list[LicenseGroup]]:
   |

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `node`
  --> src/glopt/algorithms/tree_dp.py:40:38
   |
39 |     def _solve_subtree(
40 |         self, graph: nx.Graph, node: Any, parent: Any, license_types: list[LicenseType], memo: dict
   |                                      ^^^
41 |     ) -> tuple[float, list[LicenseGroup]]:
42 |         children = [child for child in graph.neighbors(node) if child != parent]
   |

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `parent`
  --> src/glopt/algorithms/tree_dp.py:40:51
   |
39 |     def _solve_subtree(
40 |         self, graph: nx.Graph, node: Any, parent: Any, license_types: list[LicenseType], memo: dict
   |                                                   ^^^
41 |     ) -> tuple[float, list[LicenseGroup]]:
42 |         children = [child for child in graph.neighbors(node) if child != parent]
   |

PLC0415 `import` should be at the top-level of a file
  --> src/glopt/algorithms/tree_dp.py:76:17
   |
74 |                     continue
75 |
76 |                 from itertools import combinations
   |                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
77 |
78 |                 if num_children == 0:
   |

SIM108 Use ternary operator `child_combinations = [()] if num_children == 0 else combinations(children, num_children)` instead of `if`-`else`-block
  --> src/glopt/algorithms/tree_dp.py:78:17
   |
76 |                   from itertools import combinations
77 |
78 | /                 if num_children == 0:
79 | |                     child_combinations = [()]
80 | |                 else:
81 | |                     child_combinations = combinations(children, num_children)
   | |_____________________________________________________________________________^
82 |
83 |                   for child_combination in child_combinations:
   |
help: Replace `if`-`else`-block with `child_combinations = [()] if num_children == 0 else combinations(children, num_children)`

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `child`
   --> src/glopt/algorithms/tree_dp.py:109:39
    |
108 |     def _solve_child_subtree(
109 |         self, graph: nx.Graph, child: Any, parent: Any, license_types: list[LicenseType], memo: dict
    |                                       ^^^
110 |     ) -> tuple[float, list[LicenseGroup]]:
111 |         grandchildren = [gc for gc in graph.neighbors(child) if gc != parent]
    |

ANN401 Dynamically typed expressions (typing.Any) are disallowed in `parent`
   --> src/glopt/algorithms/tree_dp.py:109:52
    |
108 |     def _solve_child_subtree(
109 |         self, graph: nx.Graph, child: Any, parent: Any, license_types: list[LicenseType], memo: dict
    |                                                    ^^^
110 |     ) -> tuple[float, list[LicenseGroup]]:
111 |         grandchildren = [gc for gc in graph.neighbors(child) if gc != parent]
    |

ANN201 Missing return type annotation for public function `suppress_trace_output`
  --> src/glopt/cli/all.py:30:5
   |
29 | @contextmanager
30 | def suppress_trace_output():
   |     ^^^^^^^^^^^^^^^^^^^^^
31 |     orig_print_exc = traceback.print_exc
32 |     orig_stderr = sys.stderr
   |
help: Add return type annotation

ARG005 Unused lambda argument: `a`
  --> src/glopt/cli/all.py:34:39
   |
32 |     orig_stderr = sys.stderr
33 |     try:
34 |         traceback.print_exc = lambda *a, **k: None
   |                                       ^
35 |         sys.stderr = io.StringIO()
36 |         yield
   |

ARG005 Unused lambda argument: `k`
  --> src/glopt/cli/all.py:34:44
   |
32 |     orig_stderr = sys.stderr
33 |     try:
34 |         traceback.print_exc = lambda *a, **k: None
   |                                            ^
35 |         sys.stderr = io.StringIO()
36 |         yield
   |

T201 `print` found
  --> src/glopt/cli/all.py:44:5
   |
42 | def _err(msg: str, e: Exception) -> None:
43 |     brief = "".join(traceback.format_exception_only(type(e), e)).strip()
44 |     print(f"[ERROR] {msg}: {brief}", file=sys.stderr)
   |     ^^^^^
   |
help: Remove `print`

FBT001 Boolean-typed positional argument in function definition
  --> src/glopt/cli/all.py:47:17
   |
47 | def _fmt_status(valid: bool) -> str:
   |                 ^^^^^
48 |     return "ok" if valid else "invalid"
   |

T201 `print` found
  --> src/glopt/cli/all.py:53:9
   |
51 | def _print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
52 |     if not rows:
53 |         print(f"\n[LICENSE] {title} (no runs)")
   |         ^^^^^
54 |         return
   |
help: Remove `print`

T201 `print` found
  --> src/glopt/cli/all.py:69:5
   |
67 |         return "| " + " | ".join(padded) + " |"
68 |
69 |     print(f"\n[LICENSE] {title}")
   |     ^^^^^
70 |     print(line())
71 |     print(fmt_row(headers))
   |
help: Remove `print`

T201 `print` found
  --> src/glopt/cli/all.py:70:5
   |
69 |     print(f"\n[LICENSE] {title}")
70 |     print(line())
   |     ^^^^^
71 |     print(fmt_row(headers))
72 |     print(line(sep_mid="+"))
   |
help: Remove `print`

T201 `print` found
  --> src/glopt/cli/all.py:71:5
   |
69 |     print(f"\n[LICENSE] {title}")
70 |     print(line())
71 |     print(fmt_row(headers))
   |     ^^^^^
72 |     print(line(sep_mid="+"))
73 |     for r in rows:
   |
help: Remove `print`

T201 `print` found
  --> src/glopt/cli/all.py:72:5
   |
70 |     print(line())
71 |     print(fmt_row(headers))
72 |     print(line(sep_mid="+"))
   |     ^^^^^
73 |     for r in rows:
74 |         print(fmt_row(r))
   |
help: Remove `print`

T201 `print` found
  --> src/glopt/cli/all.py:74:9
   |
72 |     print(line(sep_mid="+"))
73 |     for r in rows:
74 |         print(fmt_row(r))
   |         ^^^^^
75 |     print(line())
   |
help: Remove `print`

T201 `print` found
  --> src/glopt/cli/all.py:75:5
   |
73 |     for r in rows:
74 |         print(fmt_row(r))
75 |     print(line())
   |     ^^^^^
   |
help: Remove `print`

C901 `main` is too complex (16 > 10)
  --> src/glopt/cli/all.py:92:5
   |
92 | def main() -> int:
   |     ^^^^
93 |     run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
94 |     _, graphs_dir_root, csv_dir = build_paths(run_id)
   |

PLR0912 Too many branches (13 > 12)
  --> src/glopt/cli/all.py:92:5
   |
92 | def main() -> int:
   |     ^^^^
93 |     run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
94 |     _, graphs_dir_root, csv_dir = build_paths(run_id)
   |

PLR0915 Too many statements (69 > 50)
  --> src/glopt/cli/all.py:92:5
   |
92 | def main() -> int:
   |     ^^^^
93 |     run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
94 |     _, graphs_dir_root, csv_dir = build_paths(run_id)
   |

DTZ005 `datetime.datetime.now()` called without a `tz` argument
  --> src/glopt/cli/all.py:93:14
   |
92 | def main() -> int:
93 |     run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
   |              ^^^^^^^^^^^^^^
94 |     _, graphs_dir_root, csv_dir = build_paths(run_id)
   |
help: Pass a `datetime.timezone` object to the `tz` parameter

SLF001 Private member accessed: `_GENERATORS`
  --> src/glopt/cli/all.py:97:28
   |
96 |     try:
97 |         graph_names = list(GraphGeneratorFactory._GENERATORS.keys())
   |                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
98 |     except Exception as e:
99 |         _err("loading graph generators", e)
   |

BLE001 Do not catch blind exception: `Exception`
   --> src/glopt/cli/all.py:98:12
    |
 96 |     try:
 97 |         graph_names = list(GraphGeneratorFactory._GENERATORS.keys())
 98 |     except Exception as e:
    |            ^^^^^^^^^
 99 |         _err("loading graph generators", e)
100 |         return 2
    |

SLF001 Private member accessed: `_CONFIGS`
   --> src/glopt/cli/all.py:103:32
    |
102 |     try:
103 |         license_configs = list(LicenseConfigFactory._CONFIGS.keys())
    |                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
104 |     except Exception as e:
105 |         _err("loading license configs", e)
    |

BLE001 Do not catch blind exception: `Exception`
   --> src/glopt/cli/all.py:104:12
    |
102 |     try:
103 |         license_configs = list(LicenseConfigFactory._CONFIGS.keys())
104 |     except Exception as e:
    |            ^^^^^^^^^
105 |         _err("loading license configs", e)
106 |         return 2
    |

BLE001 Do not catch blind exception: `Exception`
   --> src/glopt/cli/all.py:110:12
    |
108 |     try:
109 |         algorithm_names = list(algorithms.__all__)
110 |     except Exception as e:
    |            ^^^^^^^^^
111 |         _err("loading algorithms list", e)
112 |         return 2
    |

PLR1714 Consider merging multiple comparisons: `graph_name in {"complete", "star"}`.
   --> src/glopt/cli/all.py:118:12
    |
117 |     for graph_name in graph_names:
118 |         if graph_name == "complete" or graph_name == "star":
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
119 |             continue
    |
help: Merge multiple comparisons

T201 `print` found
   --> src/glopt/cli/all.py:122:9
    |
121 |         params = DEFAULT_GRAPH_PARAMS.get(graph_name, {})
122 |         print(f"\n[GRAPH] {graph_name} params={params}")
    |         ^^^^^
123 |
124 |         try:
    |
help: Remove `print`

BLE001 Do not catch blind exception: `Exception`
   --> src/glopt/cli/all.py:126:16
    |
124 |         try:
125 |             graph = generate_graph(graph_name, N_NODES, params)
126 |         except Exception as e:
    |                ^^^^^^^^^
127 |             _err(f"graph generation failed: {graph_name}", e)
128 |             had_errors = True
    |

BLE001 Do not catch blind exception: `Exception`
   --> src/glopt/cli/all.py:134:20
    |
132 |             try:
133 |                 license_types = LicenseConfigFactory.get_config(lic_name)
134 |             except Exception as e:
    |                    ^^^^^^^^^
135 |                 _err(f"license config failed: {lic_name}", e)
136 |                 had_errors = True
    |

PTH118 `os.path.join()` should be replaced by `Path` with `/` operator
   --> src/glopt/cli/all.py:139:21
    |
137 |                 continue
138 |
139 |             g_dir = os.path.join(graphs_dir_root, graph_name, lic_name)
    |                     ^^^^^^^^^^^^
140 |             try:
141 |                 ensure_dir(g_dir)
    |

BLE001 Do not catch blind exception: `Exception`
   --> src/glopt/cli/all.py:142:20
    |
140 |             try:
141 |                 ensure_dir(g_dir)
142 |             except Exception as e:
    |                    ^^^^^^^^^
143 |                 _err(f"ensure_dir failed: {g_dir}", e)
144 |                 had_errors = True
    |

BLE001 Do not catch blind exception: `Exception`
   --> src/glopt/cli/all.py:152:24
    |
150 |                 try:
151 |                     algo = instantiate_algorithms([algo_name])[0]
152 |                 except Exception as e:
    |                        ^^^^^^^^^
153 |                     _err(f"algorithm setup failed: {algo_name}", e)
154 |                     had_errors = True
    |

BLE001 Do not catch blind exception: `Exception`
   --> src/glopt/cli/all.py:166:24
    |
164 |                             graphs_dir=g_dir,
165 |                         )
166 |                 except Exception as e:
    |                        ^^^^^^^^^
167 |                     _err(f"run failed: algo={algo_name} graph={graph_name} lic={lic_name}", e)
168 |                     had_errors = True
    |

TRY300 Consider moving this statement to an `else` block
   --> src/glopt/cli/all.py:188:21
    |
186 |                 try:
187 |                     v = float(row[1])
188 |                     return (0, v)
    |                     ^^^^^^^^^^^^^
189 |                 except ValueError:
190 |                     return (1, float("inf"))
    |

BLE001 Do not catch blind exception: `Exception`
   --> src/glopt/cli/all.py:201:12
    |
199 |     try:
200 |         write_csv(csv_dir, run_id, results)
201 |     except Exception as e:
    |            ^^^^^^^^^
202 |         _err("writing CSV failed", e)
203 |         had_errors = True
    |

PLR2004 Magic value used in comparison, consider replacing `2` with a constant variable
  --> src/glopt/cli/benchmark.py:83:22
   |
81 |     elif name == "small_world":
82 |         k = int(p.get("k", 6))
83 |         if n_nodes > 2:
   |                      ^
84 |             k = max(2, min(k, n_nodes - 1))
85 |             if k % 2 == 1:
   |

PLC0415 `import` should be at the top-level of a file
   --> src/glopt/cli/benchmark.py:107:5
    |
105 |     conn: Connection,
106 | ) -> None:
107 |     from time import perf_counter
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
108 |
109 |     try:
    |

BLE001 Do not catch blind exception: `Exception`
   --> src/glopt/cli/benchmark.py:116:12
    |
114 |         elapsed_ms = (perf_counter() - t0) * 1000.0
115 |         conn.send((True, (algo.name, elapsed_ms, solution)))
116 |     except Exception as e:
    |            ^^^^^^^^^
117 |         conn.send((False, repr(e)))
118 |     finally:
    |

TRY003 Avoid specifying long messages outside the exception class
   --> src/glopt/cli/benchmark.py:138:19
    |
136 |                 algo_name, elapsed_ms, solution = payload
137 |                 return algo_name, float(elapsed_ms), solution
138 |             raise RuntimeError(f"solver error: {payload}")
    |                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
139 |         p.terminate()
140 |         p.join(1)
    |

EM102 Exception must not use an f-string literal, assign to variable first
   --> src/glopt/cli/benchmark.py:138:32
    |
136 |                 algo_name, elapsed_ms, solution = payload
137 |                 return algo_name, float(elapsed_ms), solution
138 |             raise RuntimeError(f"solver error: {payload}")
    |                                ^^^^^^^^^^^^^^^^^^^^^^^^^^
139 |         p.terminate()
140 |         p.join(1)
    |
help: Assign to variable; remove f-string literal

SIM105 Use `contextlib.suppress(Exception)` instead of `try`-`except`-`pass`
   --> src/glopt/cli/benchmark.py:143:9
    |
141 |           return None
142 |       finally:
143 | /         try:
144 | |             parent_conn.close()
145 | |         except Exception:
146 | |             pass
    | |________________^
147 |           if p.is_alive():
148 |               p.terminate()
    |
help: Replace with `contextlib.suppress(Exception)`

S110 `try`-`except`-`pass` detected, consider logging the exception
   --> src/glopt/cli/benchmark.py:145:9
    |
143 |           try:
144 |               parent_conn.close()
145 | /         except Exception:
146 | |             pass
    | |________________^
147 |           if p.is_alive():
148 |               p.terminate()
    |

BLE001 Do not catch blind exception: `Exception`
   --> src/glopt/cli/benchmark.py:145:16
    |
143 |         try:
144 |             parent_conn.close()
145 |         except Exception:
    |                ^^^^^^^^^
146 |             pass
147 |         if p.is_alive():
    |

PTH118 `os.path.join()` should be replaced by `Path` with `/` operator
   --> src/glopt/cli/benchmark.py:154:16
    |
152 | def write_algo_csv(csv_dir: str, run_id: str, license_name: str, algo_name: str, rows: Iterable[RunResult]) -> str:
153 |     safe_lic = "".join(c if c.isalnum() or c in "-_" else "_" for c in license_name)
154 |     out_path = os.path.join(csv_dir, f"{run_id}_{safe_lic}_{algo_name}.csv")
    |                ^^^^^^^^^^^^
155 |     first = True
156 |     with open(out_path, "w", newline="", encoding="utf-8") as f:
    |

PTH123 `open()` should be replaced by `Path.open()`
   --> src/glopt/cli/benchmark.py:156:10
    |
154 |     out_path = os.path.join(csv_dir, f"{run_id}_{safe_lic}_{algo_name}.csv")
155 |     first = True
156 |     with open(out_path, "w", newline="", encoding="utf-8") as f:
    |          ^^^^
157 |         import csv
    |

PLC0415 `import` should be at the top-level of a file
   --> src/glopt/cli/benchmark.py:157:9
    |
155 |     first = True
156 |     with open(out_path, "w", newline="", encoding="utf-8") as f:
157 |         import csv
    |         ^^^^^^^^^^
158 |
159 |         writer = None
    |

C901 `main` is too complex (13 > 10)
   --> src/glopt/cli/benchmark.py:174:5
    |
174 | def main() -> None:
    |     ^^^^
175 |     mp.set_start_method("spawn", force=True)
    |

PLR0915 Too many statements (66 > 50)
   --> src/glopt/cli/benchmark.py:174:5
    |
174 | def main() -> None:
    |     ^^^^
175 |     mp.set_start_method("spawn", force=True)
    |

DTZ005 `datetime.datetime.now()` called without a `tz` argument
   --> src/glopt/cli/benchmark.py:177:24
    |
175 |     mp.set_start_method("spawn", force=True)
176 |
177 |     run_id = RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")
    |                        ^^^^^^^^^^^^^^
178 |     _, _graphs_dir, csv_dir = build_paths(run_id)
179 |     ensure_dir(csv_dir)
    |
help: Pass a `datetime.timezone` object to the `tz` parameter

T201 `print` found
   --> src/glopt/cli/benchmark.py:184:9
    |
182 |     if missing:
183 |         avail = ", ".join(getattr(algorithms, "__all__", []))
184 |         print(f"[ERROR] unknown algorithms: {', '.join(missing)}; available: {avail}", file=sys.stderr)
    |         ^^^^^
185 |         sys.exit(2)
    |
help: Remove `print`

PLR2004 Magic value used in comparison, consider replacing `2` with a constant variable
   --> src/glopt/cli/benchmark.py:187:49
    |
185 |         sys.exit(2)
186 |
187 |     step = SIZES[1] - SIZES[0] if len(SIZES) >= 2 else 0
    |                                                 ^
188 |     print(f"Benchmark run_id={run_id}")
189 |     print(f"Graphs={GRAPH_NAMES}")
    |

T201 `print` found
   --> src/glopt/cli/benchmark.py:188:5
    |
187 |     step = SIZES[1] - SIZES[0] if len(SIZES) >= 2 else 0
188 |     print(f"Benchmark run_id={run_id}")
    |     ^^^^^
189 |     print(f"Graphs={GRAPH_NAMES}")
190 |     print(f"Sizes={SIZES[0]}..{SIZES[-1]} step={step}")
    |
help: Remove `print`

T201 `print` found
   --> src/glopt/cli/benchmark.py:189:5
    |
187 |     step = SIZES[1] - SIZES[0] if len(SIZES) >= 2 else 0
188 |     print(f"Benchmark run_id={run_id}")
189 |     print(f"Graphs={GRAPH_NAMES}")
    |     ^^^^^
190 |     print(f"Sizes={SIZES[0]}..{SIZES[-1]} step={step}")
191 |     print(f"Param overrides={GRAPH_PARAMS_OVERRIDES or '{}'}")
    |
help: Remove `print`

T201 `print` found
   --> src/glopt/cli/benchmark.py:190:5
    |
188 |     print(f"Benchmark run_id={run_id}")
189 |     print(f"Graphs={GRAPH_NAMES}")
190 |     print(f"Sizes={SIZES[0]}..{SIZES[-1]} step={step}")
    |     ^^^^^
191 |     print(f"Param overrides={GRAPH_PARAMS_OVERRIDES or '{}'}")
192 |     print(f"Algorithms={ALGORITHMS}")
    |
help: Remove `print`

T201 `print` found
   --> src/glopt/cli/benchmark.py:191:5
    |
189 |     print(f"Graphs={GRAPH_NAMES}")
190 |     print(f"Sizes={SIZES[0]}..{SIZES[-1]} step={step}")
191 |     print(f"Param overrides={GRAPH_PARAMS_OVERRIDES or '{}'}")
    |     ^^^^^
192 |     print(f"Algorithms={ALGORITHMS}")
193 |     print(f"Timeout={TIMEOUT_SECONDS}s\n")
    |
help: Remove `print`

T201 `print` found
   --> src/glopt/cli/benchmark.py:192:5
    |
190 |     print(f"Sizes={SIZES[0]}..{SIZES[-1]} step={step}")
191 |     print(f"Param overrides={GRAPH_PARAMS_OVERRIDES or '{}'}")
192 |     print(f"Algorithms={ALGORITHMS}")
    |     ^^^^^
193 |     print(f"Timeout={TIMEOUT_SECONDS}s\n")
    |
help: Remove `print`

T201 `print` found
   --> src/glopt/cli/benchmark.py:193:5
    |
191 |     print(f"Param overrides={GRAPH_PARAMS_OVERRIDES or '{}'}")
192 |     print(f"Algorithms={ALGORITHMS}")
193 |     print(f"Timeout={TIMEOUT_SECONDS}s\n")
    |     ^^^^^
194 |
195 |     for license_name in LICENSE_CONFIG_NAMES:
    |
help: Remove `print`

BLE001 Do not catch blind exception: `Exception`
   --> src/glopt/cli/benchmark.py:198:16
    |
196 |         try:
197 |             license_types = LicenseConfigFactory.get_config(license_name)
198 |         except Exception as e:
    |                ^^^^^^^^^
199 |             print(f"[ERROR] license config failed: {license_name}: {e}", file=sys.stderr)
200 |             traceback.print_exc(limit=10, file=sys.stderr)
    |

T201 `print` found
   --> src/glopt/cli/benchmark.py:199:13
    |
197 |             license_types = LicenseConfigFactory.get_config(license_name)
198 |         except Exception as e:
199 |             print(f"[ERROR] license config failed: {license_name}: {e}", file=sys.stderr)
    |             ^^^^^
200 |             traceback.print_exc(limit=10, file=sys.stderr)
201 |             continue
    |
help: Remove `print`

T201 `print` found
   --> src/glopt/cli/benchmark.py:203:9
    |
201 |             continue
202 |
203 |         print(f"=== LICENSE={license_name} ===")
    |         ^^^^^
204 |
205 |         for algo_name in ALGORITHMS:
    |
help: Remove `print`

T201 `print` found
   --> src/glopt/cli/benchmark.py:206:13
    |
205 |         for algo_name in ALGORITHMS:
206 |             print(f"-- algo={algo_name} --")
    |             ^^^^^
207 |             rows_all_graphs: list[RunResult] = []
208 |             total_timeouts = total_failures = total_successes = 0
    |
help: Remove `print`

T201 `print` found
   --> src/glopt/cli/benchmark.py:211:17
    |
210 |             for graph_name in GRAPH_NAMES:
211 |                 print(f"[graph={graph_name}]")
    |                 ^^^^^
212 |                 timeouts = failures = successes = 0
    |
help: Remove `print`

N806 Variable `G` in function should be lowercase
   --> src/glopt/cli/benchmark.py:222:25
    |
221 |                     try:
222 |                         G = generate_graph(graph_name, n, params)
    |                         ^
223 |                     except Exception as e:
224 |                         failures += 1
    |

BLE001 Do not catch blind exception: `Exception`
   --> src/glopt/cli/benchmark.py:223:28
    |
221 |                     try:
222 |                         G = generate_graph(graph_name, n, params)
223 |                     except Exception as e:
    |                            ^^^^^^^^^
224 |                         failures += 1
225 |                         print(f"[{license_name}][{algo_name}][{graph_name}] n={n} GEN-ERROR: {e}", file=sys.stderr)
    |

T201 `print` found
   --> src/glopt/cli/benchmark.py:225:25
    |
223 |                     except Exception as e:
224 |                         failures += 1
225 |                         print(f"[{license_name}][{algo_name}][{graph_name}] n={n} GEN-ERROR: {e}", file=sys.stderr)
    |                         ^^^^^
226 |                         rows_all_graphs.append(
227 |                             RunResult(
    |
help: Remove `print`

BLE001 Do not catch blind exception: `Exception`
   --> src/glopt/cli/benchmark.py:247:28
    |
245 |                     try:
246 |                         solved = solve_with_timeout(algo_name, G, license_types, TIMEOUT_SECONDS)
247 |                     except Exception as e:
    |                            ^^^^^^^^^
248 |                         failures += 1
249 |                         print(f"[{license_name}][{algo_name}][{graph_name}] n={n} SOLVER-ERROR: {e}", file=sys.stderr)
    |

T201 `print` found
   --> src/glopt/cli/benchmark.py:249:25
    |
247 |                     except Exception as e:
248 |                         failures += 1
249 |                         print(f"[{license_name}][{algo_name}][{graph_name}] n={n} SOLVER-ERROR: {e}", file=sys.stderr)
    |                         ^^^^^
250 |                         rows_all_graphs.append(
251 |                             RunResult(
    |
help: Remove `print`

T201 `print` found
   --> src/glopt/cli/benchmark.py:271:25
    |
269 | …     if solved is None:
270 | …         timeouts += 1
271 | …         print(
    |           ^^^^^
272 | …             f"[{license_name}][{algo_name}][{graph_name}] n={n} TIMEOUT {TIMEOUT_SECONDS}s → stop sizes for this graph"
273 | …         )
    |
help: Remove `print`

T201 `print` found
   --> src/glopt/cli/benchmark.py:297:25
    |
296 |                     if not ok and PRINT_ISSUE_LIMIT is not None:
297 |                         print(
    |                         ^^^^^
298 |                             f"[{license_name}][{algo_name}][{graph_name}] n={n} VALIDATION {len(issues)} issue(s)",
299 |                             file=sys.stderr,
    |
help: Remove `print`

T201 `print` found
   --> src/glopt/cli/benchmark.py:302:29
    |
300 |                         )
301 |                         for i in issues[:PRINT_ISSUE_LIMIT]:
302 |                             print(f"  - {i.code}: {i.msg}", file=sys.stderr)
    |                             ^^^^^
303 |                         if len(issues) > PRINT_ISSUE_LIMIT:
304 |                             print(f"  ... {len(issues) - PRINT_ISSUE_LIMIT} more", file=sys.stderr)
    |
help: Remove `print`

T201 `print` found
   --> src/glopt/cli/benchmark.py:304:29
    |
302 |                             print(f"  - {i.code}: {i.msg}", file=sys.stderr)
303 |                         if len(issues) > PRINT_ISSUE_LIMIT:
304 |                             print(f"  ... {len(issues) - PRINT_ISSUE_LIMIT} more", file=sys.stderr)
    |                             ^^^^^
305 |
306 |                     rows_all_graphs.append(
    |
help: Remove `print`

T201 `print` found
   --> src/glopt/cli/benchmark.py:324:21
    |
322 |                     )
323 |                     successes += 1
324 |                     print(
    |                     ^^^^^
325 |                         f"[{license_name}][{algo_name}][{graph_name}] n={n:4d} | edges={G.number_of_edges():6d} | "
326 |                         f"time={_fmt_ms(elapsed_ms):>10} | "
    |
help: Remove `print`

T201 `print` found
   --> src/glopt/cli/benchmark.py:331:17
    |
329 |                     )
330 |
331 |                 print(
    |                 ^^^^^
332 |                     f"[SUMMARY graph] {license_name} {algo_name} on {graph_name}: ok={successes} timeout={timeouts} fail={failures}\n"
333 |                 )
    |
help: Remove `print`

T201 `print` found
   --> src/glopt/cli/benchmark.py:339:13
    |
338 |             out_csv = write_algo_csv(csv_dir, run_id, license_name, algo_name, rows_all_graphs)
339 |             print(f"[CSV] {out_csv}")
    |             ^^^^^
340 |             print(
341 |                 f"[SUMMARY algo] {license_name} {algo_name}: ok={total_successes} timeout={total_timeouts} fail={total_failures}\n"
    |
help: Remove `print`

T201 `print` found
   --> src/glopt/cli/benchmark.py:340:13
    |
338 |             out_csv = write_algo_csv(csv_dir, run_id, license_name, algo_name, rows_all_graphs)
339 |             print(f"[CSV] {out_csv}")
340 |             print(
    |             ^^^^^
341 |                 f"[SUMMARY algo] {license_name} {algo_name}: ok={total_successes} timeout={total_timeouts} fail={total_failures}\n"
342 |             )
    |
help: Remove `print`

T201 `print` found
   --> src/glopt/cli/benchmark.py:344:5
    |
342 |             )
343 |
344 |     print("Benchmark done.")
    |     ^^^^^
    |
help: Remove `print`

DTZ005 `datetime.datetime.now()` called without a `tz` argument
  --> src/glopt/cli/dynamic.py:21:14
   |
20 | def main() -> int:
21 |     run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
   |              ^^^^^^^^^^^^^^
22 |     _, _, csv_dir = build_paths(run_id)
   |
help: Pass a `datetime.timezone` object to the `tz` parameter

BLE001 Do not catch blind exception: `Exception`
  --> src/glopt/cli/dynamic.py:27:12
   |
25 |         gen = GraphGeneratorFactory.get(GRAPH_NAME)
26 |         graph = gen(n_nodes=N_NODES, **GRAPH_PARAMS)
27 |     except Exception as e:
   |            ^^^^^^^^^
28 |         print(f"[ERROR] graph generation failed: {GRAPH_NAME}: {e}", file=sys.stderr)
29 |         traceback.print_exc(limit=10, file=sys.stderr)
   |

T201 `print` found
  --> src/glopt/cli/dynamic.py:28:9
   |
26 |         graph = gen(n_nodes=N_NODES, **GRAPH_PARAMS)
27 |     except Exception as e:
28 |         print(f"[ERROR] graph generation failed: {GRAPH_NAME}: {e}", file=sys.stderr)
   |         ^^^^^
29 |         traceback.print_exc(limit=10, file=sys.stderr)
30 |         return 2
   |
help: Remove `print`

BLE001 Do not catch blind exception: `Exception`
  --> src/glopt/cli/dynamic.py:34:12
   |
32 |     try:
33 |         license_types = LicenseConfigFactory.get_config(LICENSE_CONFIG)
34 |     except Exception as e:
   |            ^^^^^^^^^
35 |         print(f"[ERROR] license config failed: {LICENSE_CONFIG}: {e}", file=sys.stderr)
36 |         traceback.print_exc(limit=10, file=sys.stderr)
   |

T201 `print` found
  --> src/glopt/cli/dynamic.py:35:9
   |
33 |         license_types = LicenseConfigFactory.get_config(LICENSE_CONFIG)
34 |     except Exception as e:
35 |         print(f"[ERROR] license config failed: {LICENSE_CONFIG}: {e}", file=sys.stderr)
   |         ^^^^^
36 |         traceback.print_exc(limit=10, file=sys.stderr)
37 |         return 2
   |
help: Remove `print`

PTH118 `os.path.join()` should be replaced by `Path` with `/` operator
  --> src/glopt/cli/dynamic.py:44:20
   |
42 |     try:
43 |         ensure_dir(csv_dir)
44 |         out_path = os.path.join(csv_dir, f"{run_id}_dynamic.csv")
   |                    ^^^^^^^^^^^^
45 |         simulator.export_history_to_csv(out_path)
46 |         print(f"[DYNAMIC] history saved to {out_path}")
   |

T201 `print` found
  --> src/glopt/cli/dynamic.py:46:9
   |
44 |         out_path = os.path.join(csv_dir, f"{run_id}_dynamic.csv")
45 |         simulator.export_history_to_csv(out_path)
46 |         print(f"[DYNAMIC] history saved to {out_path}")
   |         ^^^^^
47 |     except Exception as e:
48 |         print(f"[ERROR] exporting history failed: {e}", file=sys.stderr)
   |
help: Remove `print`

BLE001 Do not catch blind exception: `Exception`
  --> src/glopt/cli/dynamic.py:47:12
   |
45 |         simulator.export_history_to_csv(out_path)
46 |         print(f"[DYNAMIC] history saved to {out_path}")
47 |     except Exception as e:
   |            ^^^^^^^^^
48 |         print(f"[ERROR] exporting history failed: {e}", file=sys.stderr)
49 |         traceback.print_exc(limit=10, file=sys.stderr)
   |

T201 `print` found
  --> src/glopt/cli/dynamic.py:48:9
   |
46 |         print(f"[DYNAMIC] history saved to {out_path}")
47 |     except Exception as e:
48 |         print(f"[ERROR] exporting history failed: {e}", file=sys.stderr)
   |         ^^^^^
49 |         traceback.print_exc(limit=10, file=sys.stderr)
50 |         return 2
   |
help: Remove `print`

T201 `print` found
  --> src/glopt/cli/dynamic.py:53:5
   |
52 |     summary = simulator.get_simulation_summary()
53 |     print(f"[DYNAMIC] summary: {summary}")
   |     ^^^^^
54 |     return 0
   |
help: Remove `print`

DTZ005 `datetime.datetime.now()` called without a `tz` argument
  --> src/glopt/cli/test.py:39:24
   |
38 | def main() -> None:
39 |     run_id = RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")
   |                        ^^^^^^^^^^^^^^
40 |     _, graphs_dir, csv_dir = build_paths(run_id)
   |
help: Pass a `datetime.timezone` object to the `tz` parameter

BLE001 Do not catch blind exception: `Exception`
  --> src/glopt/cli/test.py:44:12
   |
42 |     try:
43 |         graph = generate_graph(GRAPH_NAME, N_NODES, GRAPH_PARAMS)
44 |     except Exception as e:
   |            ^^^^^^^^^
45 |         print(f"[ERROR] graph generation failed: {GRAPH_NAME}: {e}", file=sys.stderr)
46 |         traceback.print_exc(limit=10, file=sys.stderr)
   |

T201 `print` found
  --> src/glopt/cli/test.py:45:9
   |
43 |         graph = generate_graph(GRAPH_NAME, N_NODES, GRAPH_PARAMS)
44 |     except Exception as e:
45 |         print(f"[ERROR] graph generation failed: {GRAPH_NAME}: {e}", file=sys.stderr)
   |         ^^^^^
46 |         traceback.print_exc(limit=10, file=sys.stderr)
47 |         sys.exit(2)
   |
help: Remove `print`

BLE001 Do not catch blind exception: `Exception`
  --> src/glopt/cli/test.py:51:12
   |
49 |     try:
50 |         license_types = LicenseConfigFactory.get_config(LICENSE_CONFIG_NAME)
51 |     except Exception as e:
   |            ^^^^^^^^^
52 |         print(f"[ERROR] license config failed: {LICENSE_CONFIG_NAME}: {e}", file=sys.stderr)
53 |         traceback.print_exc(limit=10, file=sys.stderr)
   |

T201 `print` found
  --> src/glopt/cli/test.py:52:9
   |
50 |         license_types = LicenseConfigFactory.get_config(LICENSE_CONFIG_NAME)
51 |     except Exception as e:
52 |         print(f"[ERROR] license config failed: {LICENSE_CONFIG_NAME}: {e}", file=sys.stderr)
   |         ^^^^^
53 |         traceback.print_exc(limit=10, file=sys.stderr)
54 |         sys.exit(2)
   |
help: Remove `print`

BLE001 Do not catch blind exception: `Exception`
  --> src/glopt/cli/test.py:58:12
   |
56 |     try:
57 |         algos = instantiate_algorithms(ALGORITHMS)
58 |     except Exception as e:
   |            ^^^^^^^^^
59 |         print(f"[ERROR] algorithm setup failed: {e}", file=sys.stderr)
60 |         traceback.print_exc(limit=10, file=sys.stderr)
   |

T201 `print` found
  --> src/glopt/cli/test.py:59:9
   |
57 |         algos = instantiate_algorithms(ALGORITHMS)
58 |     except Exception as e:
59 |         print(f"[ERROR] algorithm setup failed: {e}", file=sys.stderr)
   |         ^^^^^
60 |         traceback.print_exc(limit=10, file=sys.stderr)
61 |         sys.exit(2)
   |
help: Remove `print`

TRY003 Avoid specifying long messages outside the exception class
  --> src/glopt/core/models.py:21:19
   |
19 |     def __post_init__(self) -> None:
20 |         if self.cost < 0:
21 |             raise ValueError("cost must be >= 0")
   |                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
22 |         if self.min_capacity < 1:
23 |             raise ValueError("min_capacity must be >= 1")
   |

EM101 Exception must not use a string literal, assign to variable first
  --> src/glopt/core/models.py:21:30
   |
19 |     def __post_init__(self) -> None:
20 |         if self.cost < 0:
21 |             raise ValueError("cost must be >= 0")
   |                              ^^^^^^^^^^^^^^^^^^^
22 |         if self.min_capacity < 1:
23 |             raise ValueError("min_capacity must be >= 1")
   |
help: Assign to variable; remove string literal

TRY003 Avoid specifying long messages outside the exception class
  --> src/glopt/core/models.py:23:19
   |
21 |             raise ValueError("cost must be >= 0")
22 |         if self.min_capacity < 1:
23 |             raise ValueError("min_capacity must be >= 1")
   |                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
24 |         if self.max_capacity < self.min_capacity:
25 |             raise ValueError("max_capacity must be >= min_capacity")
   |

EM101 Exception must not use a string literal, assign to variable first
  --> src/glopt/core/models.py:23:30
   |
21 |             raise ValueError("cost must be >= 0")
22 |         if self.min_capacity < 1:
23 |             raise ValueError("min_capacity must be >= 1")
   |                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
24 |         if self.max_capacity < self.min_capacity:
25 |             raise ValueError("max_capacity must be >= min_capacity")
   |
help: Assign to variable; remove string literal

TRY003 Avoid specifying long messages outside the exception class
  --> src/glopt/core/models.py:25:19
   |
23 |             raise ValueError("min_capacity must be >= 1")
24 |         if self.max_capacity < self.min_capacity:
25 |             raise ValueError("max_capacity must be >= min_capacity")
   |                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   |

EM101 Exception must not use a string literal, assign to variable first
  --> src/glopt/core/models.py:25:30
   |
23 |             raise ValueError("min_capacity must be >= 1")
24 |         if self.max_capacity < self.min_capacity:
25 |             raise ValueError("max_capacity must be >= min_capacity")
   |                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   |
help: Assign to variable; remove string literal

UP046 Generic class `LicenseGroup` uses `Generic` subclass instead of type parameters
  --> src/glopt/core/models.py:29:20
   |
28 | @dataclass(frozen=True, slots=True)
29 | class LicenseGroup(Generic[N]):
   |                    ^^^^^^^^^^
30 |     license_type: LicenseType
31 |     owner: N
   |
help: Use type parameters

TRY003 Avoid specifying long messages outside the exception class
  --> src/glopt/core/models.py:47:19
   |
45 |   …     s = self.size
46 |   …     if not (self.license_type.min_capacity <= s <= self.license_type.max_capacity):
47 |   …         raise ValueError(
   |  _________________^
48 | | …             f"group size {s} violates [{self.license_type.min_capacity}, {self.license_type.max_capacity}] for {self.license_type.na…
49 | | …         )
   | |___________^
   |

EM102 Exception must not use an f-string literal, assign to variable first
  --> src/glopt/core/models.py:48:17
   |
46 | …not (self.license_type.min_capacity <= s <= self.license_type.max_capacity):
47 | … raise ValueError(
48 | …     f"group size {s} violates [{self.license_type.min_capacity}, {self.license_type.max_capacity}] for {self.license_type.name}"
   |       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
49 | … )
   |
help: Assign to variable; remove f-string literal

UP046 Generic class `Solution` uses `Generic` subclass instead of type parameters
  --> src/glopt/core/models.py:53:16
   |
52 | @dataclass(slots=True)
53 | class Solution(Generic[N]):
   |                ^^^^^^^^^^
54 |     groups: tuple[LicenseGroup[N], ...] = ()
   |
help: Use type parameters

UP046 Generic class `Algorithm` uses `Generic` subclass instead of type parameters
  --> src/glopt/core/models.py:68:22
   |
68 | class Algorithm(ABC, Generic[N]):
   |                      ^^^^^^^^^^
69 |     @abstractmethod
70 |     def solve(self, graph: nx.Graph, license_types: Sequence[LicenseType], **kwargs) -> Solution[N]: ...
   |
help: Use type parameters

ANN003 Missing type annotation for `**kwargs`
  --> src/glopt/core/models.py:70:76
   |
68 | class Algorithm(ABC, Generic[N]):
69 |     @abstractmethod
70 |     def solve(self, graph: nx.Graph, license_types: Sequence[LicenseType], **kwargs) -> Solution[N]: ...
   |                                                                            ^^^^^^^^
71 |
72 |     @property
   |

TC003 Move standard library import `collections.abc.Sequence` into a type-checking block
 --> src/glopt/core/mutations.py:4:29
  |
3 | import random
4 | from collections.abc import Sequence
  |                             ^^^^^^^^
5 |
6 | import networkx as nx
  |
help: Move into type-checking block

TC002 Move third-party import `networkx` into a type-checking block
 --> src/glopt/core/mutations.py:6:20
  |
4 | from collections.abc import Sequence
5 |
6 | import networkx as nx
  |                    ^^
7 |
8 | from .models import LicenseGroup, LicenseType, Solution
  |
help: Move into type-checking block

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
  --> src/glopt/core/mutations.py:31:18
   |
29 |         while len(out) < k and attempts < k * 10:
30 |             attempts += 1
31 |             op = random.choices(ops, weights=weights, k=1)[0]
   |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
32 |             try:
33 |                 cand = op(base, graph, list(license_types))
   |

BLE001 Do not catch blind exception: `Exception`
  --> src/glopt/core/mutations.py:34:20
   |
32 |             try:
33 |                 cand = op(base, graph, list(license_types))
34 |             except Exception:
   |                    ^^^^^^^^^
35 |                 cand = None
36 |             if cand is not None:
   |

ARG004 Unused static method argument: `graph`
  --> src/glopt/core/mutations.py:43:9
   |
41 |     def change_license_type(
42 |         solution: Solution,
43 |         graph: nx.Graph,
   |         ^^^^^
44 |         license_types: list[LicenseType],
45 |     ) -> Solution | None:
   |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
  --> src/glopt/core/mutations.py:48:17
   |
46 |         if not solution.groups:
47 |             return None
48 |         group = random.choice(solution.groups)
   |                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
49 |         compatible = SolutionBuilder.get_compatible_license_types(group.size, license_types, exclude=group.license_type)
50 |         if not compatible:
   |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
  --> src/glopt/core/mutations.py:52:18
   |
50 |         if not compatible:
51 |             return None
52 |         new_lt = random.choice(compatible)
   |                  ^^^^^^^^^^^^^^^^^^^^^^^^^
53 |
54 |         new_groups = []
   |

ARG004 Unused static method argument: `license_types`
  --> src/glopt/core/mutations.py:66:9
   |
64 |         solution: Solution,
65 |         graph: nx.Graph,
66 |         license_types: list[LicenseType],
   |         ^^^^^^^^^^^^^
67 |     ) -> Solution | None:
68 |         if len(solution.groups) < 2:
   |

PLR2004 Magic value used in comparison, consider replacing `2` with a constant variable
  --> src/glopt/core/mutations.py:68:35
   |
66 |         license_types: list[LicenseType],
67 |     ) -> Solution | None:
68 |         if len(solution.groups) < 2:
   |                                   ^
69 |             return None
   |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
  --> src/glopt/core/mutations.py:76:22
   |
74 |             return None
75 |
76 |         from_group = random.choice(donors)
   |                      ^^^^^^^^^^^^^^^^^^^^^
77 |         pot_receivers = [g for g in receivers if g is not from_group]
78 |         if not pot_receivers:
   |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
  --> src/glopt/core/mutations.py:80:20
   |
78 |         if not pot_receivers:
79 |             return None
80 |         to_group = random.choice(pot_receivers)
   |                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
81 |
82 |         member = random.choice(list(from_group.additional_members))
   |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
  --> src/glopt/core/mutations.py:82:18
   |
80 |         to_group = random.choice(pot_receivers)
81 |
82 |         member = random.choice(list(from_group.additional_members))
   |                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
83 |         allowed = SolutionBuilder.get_owner_neighbors_with_self(graph, to_group.owner)
84 |         if member not in allowed:
   |

PLR2004 Magic value used in comparison, consider replacing `2` with a constant variable
   --> src/glopt/core/mutations.py:103:35
    |
101 |         license_types: list[LicenseType],
102 |     ) -> Solution | None:
103 |         if len(solution.groups) < 2:
    |                                   ^
104 |             return None
105 |         g1, g2 = random.sample(list(solution.groups), 2)
    |

PLR2004 Magic value used in comparison, consider replacing `2` with a constant variable
   --> src/glopt/core/mutations.py:123:62
    |
121 |             return None
122 |
123 |         splittable = [g for g in solution.groups if g.size > 2]
    |                                                              ^
124 |         if not splittable:
125 |             return None
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/core/mutations.py:127:17
    |
125 |             return None
126 |
127 |         group = random.choice(splittable)
    |                 ^^^^^^^^^^^^^^^^^^^^^^^^^
128 |         members = list(group.all_members)
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/core/mutations.py:132:19
    |
130 |         for _ in range(4):
131 |             random.shuffle(members)
132 |             cut = random.randint(1, len(members) - 1)
    |                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
133 |             part1, part2 = members[:cut], members[cut:]
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/core/mutations.py:140:22
    |
138 |                 continue
139 |
140 |             owner1 = random.choice(part1)
    |                      ^^^^^^^^^^^^^^^^^^^^
141 |             owner2 = random.choice(part2)
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/core/mutations.py:141:22
    |
140 |             owner1 = random.choice(part1)
141 |             owner2 = random.choice(part2)
    |                      ^^^^^^^^^^^^^^^^^^^^
142 |
143 |             neigh1 = SolutionBuilder.get_owner_neighbors_with_self(graph, owner1)
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/core/mutations.py:148:19
    |
146 |                 continue
147 |
148 |             lt1 = random.choice(compat1)
    |                   ^^^^^^^^^^^^^^^^^^^^^^
149 |             lt2 = random.choice(compat2)
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/core/mutations.py:149:19
    |
148 |             lt1 = random.choice(compat1)
149 |             lt2 = random.choice(compat2)
    |                   ^^^^^^^^^^^^^^^^^^^^^^
150 |
151 |             g1 = LicenseGroup(lt1, owner1, frozenset(set(part1) - {owner1}))
    |

TID252 Prefer absolute imports over relative imports from parent modules
  --> src/glopt/core/run.py:12:1
   |
10 | import networkx as nx
11 |
12 | from .. import algorithms
   | ^^^^^^^^^^^^^^^^^^^^^^^^^
13 | from ..io.graph_generator import GraphGeneratorFactory
14 | from ..io.graph_visualizer import GraphVisualizer
   |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
  --> src/glopt/core/run.py:13:1
   |
12 | from .. import algorithms
13 | from ..io.graph_generator import GraphGeneratorFactory
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
14 | from ..io.graph_visualizer import GraphVisualizer
15 | from .models import Algorithm, LicenseType, Solution
   |
help: Replace relative imports from parent modules with absolute imports

TID252 Prefer absolute imports over relative imports from parent modules
  --> src/glopt/core/run.py:14:1
   |
12 | from .. import algorithms
13 | from ..io.graph_generator import GraphGeneratorFactory
14 | from ..io.graph_visualizer import GraphVisualizer
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
15 | from .models import Algorithm, LicenseType, Solution
16 | from .solution_validator import SolutionValidator
   |
help: Replace relative imports from parent modules with absolute imports

TC001 Move application import `.models.Algorithm` into a type-checking block
  --> src/glopt/core/run.py:15:21
   |
13 | from ..io.graph_generator import GraphGeneratorFactory
14 | from ..io.graph_visualizer import GraphVisualizer
15 | from .models import Algorithm, LicenseType, Solution
   |                     ^^^^^^^^^
16 | from .solution_validator import SolutionValidator
   |
help: Move into type-checking block

TC001 Move application import `.models.LicenseType` into a type-checking block
  --> src/glopt/core/run.py:15:32
   |
13 | from ..io.graph_generator import GraphGeneratorFactory
14 | from ..io.graph_visualizer import GraphVisualizer
15 | from .models import Algorithm, LicenseType, Solution
   |                                ^^^^^^^^^^^
16 | from .solution_validator import SolutionValidator
   |
help: Move into type-checking block

TC001 Move application import `.models.Solution` into a type-checking block
  --> src/glopt/core/run.py:15:45
   |
13 | from ..io.graph_generator import GraphGeneratorFactory
14 | from ..io.graph_visualizer import GraphVisualizer
15 | from .models import Algorithm, LicenseType, Solution
   |                                             ^^^^^^^^
16 | from .solution_validator import SolutionValidator
   |
help: Move into type-checking block

N806 Variable `G` in function should be lowercase
  --> src/glopt/core/run.py:38:5
   |
36 | def generate_graph(name: str, n_nodes: int, params: dict[str, Any]) -> nx.Graph:
37 |     gen = GraphGeneratorFactory.get(name)
38 |     G = gen(n_nodes=n_nodes, **params)
   |     ^
39 |     if not all(isinstance(v, int) for v in G.nodes()):
40 |         mapping = {v: i for i, v in enumerate(G.nodes())}
   |

N806 Variable `G` in function should be lowercase
  --> src/glopt/core/run.py:41:9
   |
39 |     if not all(isinstance(v, int) for v in G.nodes()):
40 |         mapping = {v: i for i, v in enumerate(G.nodes())}
41 |         G = nx.relabel_nodes(G, mapping, copy=True)
   |         ^
42 |     return G
   |

TRY003 Avoid specifying long messages outside the exception class
  --> src/glopt/core/run.py:56:15
   |
54 |     if missing:
55 |         avail = ", ".join(getattr(algorithms, "__all__", []))
56 |         raise ValueError(f"unknown algorithms: {', '.join(missing)}; available: {avail}")
   |               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
57 |     if not loaded:
58 |         raise ValueError("no algorithms selected")
   |

EM102 Exception must not use an f-string literal, assign to variable first
  --> src/glopt/core/run.py:56:26
   |
54 |     if missing:
55 |         avail = ", ".join(getattr(algorithms, "__all__", []))
56 |         raise ValueError(f"unknown algorithms: {', '.join(missing)}; available: {avail}")
   |                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
57 |     if not loaded:
58 |         raise ValueError("no algorithms selected")
   |
help: Assign to variable; remove f-string literal

TRY003 Avoid specifying long messages outside the exception class
  --> src/glopt/core/run.py:58:15
   |
56 |         raise ValueError(f"unknown algorithms: {', '.join(missing)}; available: {avail}")
57 |     if not loaded:
58 |         raise ValueError("no algorithms selected")
   |               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
59 |     return loaded
   |

EM101 Exception must not use a string literal, assign to variable first
  --> src/glopt/core/run.py:58:26
   |
56 |         raise ValueError(f"unknown algorithms: {', '.join(missing)}; available: {avail}")
57 |     if not loaded:
58 |         raise ValueError("no algorithms selected")
   |                          ^^^^^^^^^^^^^^^^^^^^^^^^
59 |     return loaded
   |
help: Assign to variable; remove string literal

PLR0913 Too many arguments in function definition (6 > 5)
  --> src/glopt/core/run.py:62:5
   |
62 | def run_once(
   |     ^^^^^^^^
63 |     algo: Algorithm,
64 |     graph: nx.Graph,
   |

BLE001 Do not catch blind exception: `Exception`
  --> src/glopt/core/run.py:77:12
   |
75 |         solution: Solution = algo.solve(graph=graph, license_types=license_types)
76 |         elapsed_ms = (perf_counter() - t0) * 1000.0
77 |     except Exception as e:
   |            ^^^^^^^^^
78 |         algo_name = getattr(algo, "name", algo.__class__.__name__)
79 |         print(f"[ERROR] solver crashed: {algo_name}: {e}", file=sys.stderr)
   |

T201 `print` found
  --> src/glopt/core/run.py:79:9
   |
77 |     except Exception as e:
78 |         algo_name = getattr(algo, "name", algo.__class__.__name__)
79 |         print(f"[ERROR] solver crashed: {algo_name}: {e}", file=sys.stderr)
   |         ^^^^^
80 |         traceback.print_exc(limit=20, file=sys.stderr)
81 |         return RunResult(
   |
help: Remove `print`

T201 `print` found
   --> src/glopt/core/run.py:99:9
    |
 97 |     ok, issues = validator.validate(solution, graph)
 98 |     if not ok:
 99 |         print(f"[VALIDATION] {algo.name}: {len(issues)} issue(s):", file=sys.stderr)
    |         ^^^^^
100 |         to_show = issues if print_issue_limit is None else issues[:print_issue_limit]
101 |         for i in to_show:
    |
help: Remove `print`

T201 `print` found
   --> src/glopt/core/run.py:102:13
    |
100 |         to_show = issues if print_issue_limit is None else issues[:print_issue_limit]
101 |         for i in to_show:
102 |             print(f"  - {i.code}: {i.msg}", file=sys.stderr)
    |             ^^^^^
103 |         if print_issue_limit is not None and len(issues) > print_issue_limit:
104 |             print(f"  ... {len(issues) - print_issue_limit} more", file=sys.stderr)
    |
help: Remove `print`

T201 `print` found
   --> src/glopt/core/run.py:104:13
    |
102 |             print(f"  - {i.code}: {i.msg}", file=sys.stderr)
103 |         if print_issue_limit is not None and len(issues) > print_issue_limit:
104 |             print(f"  ... {len(issues) - print_issue_limit} more", file=sys.stderr)
    |             ^^^^^
105 |
106 |     img_name = f"{algo.name}_{graph.number_of_nodes()}n_{graph.number_of_edges()}e.png"
    |
help: Remove `print`

PTH118 `os.path.join()` should be replaced by `Path` with `/` operator
   --> src/glopt/core/run.py:107:16
    |
106 |     img_name = f"{algo.name}_{graph.number_of_nodes()}n_{graph.number_of_edges()}e.png"
107 |     img_path = os.path.join(graphs_dir, img_name)
    |                ^^^^^^^^^^^^
108 |     try:
109 |         visualizer.visualize_solution(
    |

BLE001 Do not catch blind exception: `Exception`
   --> src/glopt/core/run.py:116:12
    |
114 |             save_path=img_path,
115 |         )
116 |     except Exception as e:
    |            ^^^^^^^^^
117 |         print(f"[WARN] failed to save image for {algo.name}: {e}", file=sys.stderr)
118 |         traceback.print_exc(limit=10, file=sys.stderr)
    |

T201 `print` found
   --> src/glopt/core/run.py:117:9
    |
115 |         )
116 |     except Exception as e:
117 |         print(f"[WARN] failed to save image for {algo.name}: {e}", file=sys.stderr)
    |         ^^^^^
118 |         traceback.print_exc(limit=10, file=sys.stderr)
119 |         img_path = ""
    |
help: Remove `print`

ANN204 Missing return type annotation for special method `__init__`
  --> src/glopt/core/solution_validator.py:19:9
   |
18 | class SolutionValidator:
19 |     def __init__(self, debug: bool = False):
   |         ^^^^^^^^
20 |         self.debug = debug
   |
help: Add return type annotation: `None`

D107 Missing docstring in `__init__`
  --> src/glopt/core/solution_validator.py:19:9
   |
18 | class SolutionValidator:
19 |     def __init__(self, debug: bool = False):
   |         ^^^^^^^^
20 |         self.debug = debug
   |

FBT001 Boolean-typed positional argument in function definition
  --> src/glopt/core/solution_validator.py:19:24
   |
18 | class SolutionValidator:
19 |     def __init__(self, debug: bool = False):
   |                        ^^^^^
20 |         self.debug = debug
   |

FBT002 Boolean default positional argument in function definition
  --> src/glopt/core/solution_validator.py:19:24
   |
18 | class SolutionValidator:
19 |     def __init__(self, debug: bool = False):
   |                        ^^^^^
20 |         self.debug = debug
   |

T201 `print` found
  --> src/glopt/core/solution_validator.py:40:17
   |
38 |         if self.debug and issues:
39 |             for i in issues:
40 |                 print(f"    {i.code}: {i.msg}")
   |                 ^^^^^
41 |
42 |         return (not issues, issues)
   |
help: Remove `print`

ANN204 Missing return type annotation for special method `__init__`
  --> src/glopt/dynamic_simulator.py:34:9
   |
33 | class DynamicNetworkSimulator:
34 |     def __init__(
   |         ^^^^^^^^
35 |         self,
36 |         rebalance_algorithm: Algorithm | None = None,
   |
help: Add return type annotation: `None`

D107 Missing docstring in `__init__`
  --> src/glopt/dynamic_simulator.py:34:9
   |
33 | class DynamicNetworkSimulator:
34 |     def __init__(
   |         ^^^^^^^^
35 |         self,
36 |         rebalance_algorithm: Algorithm | None = None,
   |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/dynamic_simulator.py:101:12
    |
 99 |         mutations = []
100 |
101 |         if random.random() < self.mutation_params.add_nodes_prob:
    |            ^^^^^^^^^^^^^^^
102 |             num_add = random.randint(1, self.mutation_params.max_nodes_add)
103 |             new_nodes = self._add_nodes(graph, num_add)
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/dynamic_simulator.py:102:23
    |
101 |         if random.random() < self.mutation_params.add_nodes_prob:
102 |             num_add = random.randint(1, self.mutation_params.max_nodes_add)
    |                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
103 |             new_nodes = self._add_nodes(graph, num_add)
104 |             mutations.append(f"Added nodes: {new_nodes}")
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/dynamic_simulator.py:106:12
    |
104 |             mutations.append(f"Added nodes: {new_nodes}")
105 |
106 |         if random.random() < self.mutation_params.remove_nodes_prob and len(graph.nodes()) > 5:
    |            ^^^^^^^^^^^^^^^
107 |             num_remove = random.randint(1, min(self.mutation_params.max_nodes_remove, len(graph.nodes()) - 5))
108 |             removed_nodes = self._remove_nodes(graph, num_remove)
    |

PLR2004 Magic value used in comparison, consider replacing `5` with a constant variable
   --> src/glopt/dynamic_simulator.py:106:94
    |
104 |             mutations.append(f"Added nodes: {new_nodes}")
105 |
106 |         if random.random() < self.mutation_params.remove_nodes_prob and len(graph.nodes()) > 5:
    |                                                                                              ^
107 |             num_remove = random.randint(1, min(self.mutation_params.max_nodes_remove, len(graph.nodes()) - 5))
108 |             removed_nodes = self._remove_nodes(graph, num_remove)
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/dynamic_simulator.py:107:26
    |
106 |         if random.random() < self.mutation_params.remove_nodes_prob and len(graph.nodes()) > 5:
107 |             num_remove = random.randint(1, min(self.mutation_params.max_nodes_remove, len(graph.nodes()) - 5))
    |                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
108 |             removed_nodes = self._remove_nodes(graph, num_remove)
109 |             mutations.append(f"Removed nodes: {removed_nodes}")
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/dynamic_simulator.py:111:12
    |
109 |             mutations.append(f"Removed nodes: {removed_nodes}")
110 |
111 |         if random.random() < self.mutation_params.add_edges_prob:
    |            ^^^^^^^^^^^^^^^
112 |             num_add = random.randint(1, self.mutation_params.max_edges_add)
113 |             added_edges = self._add_edges(graph, num_add)
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/dynamic_simulator.py:112:23
    |
111 |         if random.random() < self.mutation_params.add_edges_prob:
112 |             num_add = random.randint(1, self.mutation_params.max_edges_add)
    |                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
113 |             added_edges = self._add_edges(graph, num_add)
114 |             mutations.append(f"Added {len(added_edges)} edges")
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/dynamic_simulator.py:116:12
    |
114 |             mutations.append(f"Added {len(added_edges)} edges")
115 |
116 |         if random.random() < self.mutation_params.remove_edges_prob and len(graph.edges()) > 0:
    |            ^^^^^^^^^^^^^^^
117 |             num_remove = random.randint(1, min(self.mutation_params.max_edges_remove, len(graph.edges())))
118 |             removed_edges = self._remove_edges(graph, num_remove)
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/dynamic_simulator.py:117:26
    |
116 |         if random.random() < self.mutation_params.remove_edges_prob and len(graph.edges()) > 0:
117 |             num_remove = random.randint(1, min(self.mutation_params.max_edges_remove, len(graph.edges())))
    |                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
118 |             removed_edges = self._remove_edges(graph, num_remove)
119 |             mutations.append(f"Removed {len(removed_edges)} edges")
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/dynamic_simulator.py:134:35
    |
133 |             if existing_nodes:
134 |                 num_connections = random.randint(1, min(3, len(existing_nodes)))
    |                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
135 |                 neighbors = random.sample(existing_nodes, num_connections)
136 |                 for neighbor in neighbors:
    |

PLR2004 Magic value used in comparison, consider replacing `2` with a constant variable
   --> src/glopt/dynamic_simulator.py:153:25
    |
151 |         added_edges = []
152 |
153 |         if len(nodes) < 2:
    |                         ^
154 |             return added_edges
    |

PLC0415 `import` should be at the top-level of a file
   --> src/glopt/dynamic_simulator.py:249:9
    |
248 |     def export_history_to_csv(self, filename: str) -> None:
249 |         import csv
    |         ^^^^^^^^^^
250 |
251 |         with open(filename, "w", newline="") as csvfile:
    |

PTH123 `open()` should be replaced by `Path.open()`
   --> src/glopt/dynamic_simulator.py:251:14
    |
249 |         import csv
250 |
251 |         with open(filename, "w", newline="") as csvfile:
    |              ^^^^
252 |             fieldnames = ["step", "nodes", "edges", "cost", "groups", "cost_change", "mutations"]
253 |             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    |

ARG004 Unused static method argument: `seed`
   --> src/glopt/dynamic_simulator.py:272:32
    |
270 | class DynamicScenarioFactory:
271 |     @staticmethod
272 |     def create_growth_scenario(seed: int | None = None) -> MutationParams:
    |                                ^^^^
273 |         return MutationParams(
274 |             add_nodes_prob=0.3,
    |

ARG004 Unused static method argument: `seed`
   --> src/glopt/dynamic_simulator.py:285:31
    |
284 |     @staticmethod
285 |     def create_churn_scenario(seed: int | None = None) -> MutationParams:
    |                               ^^^^
286 |         return MutationParams(
287 |             add_nodes_prob=0.2,
    |

ARG004 Unused static method argument: `seed`
   --> src/glopt/dynamic_simulator.py:298:32
    |
297 |     @staticmethod
298 |     def create_stable_scenario(seed: int | None = None) -> MutationParams:
    |                                ^^^^
299 |         return MutationParams(
300 |             add_nodes_prob=0.1,
    |

PTH118 `os.path.join()` should be replaced by `Path` with `/` operator
  --> src/glopt/io/csv_writer.py:11:16
   |
10 | def write_csv(csv_dir: str, run_id: str, rows: Iterable[Any]) -> str:
11 |     out_path = os.path.join(csv_dir, f"{run_id}.csv")
   |                ^^^^^^^^^^^^
12 |     first = True
13 |     with open(out_path, "w", newline="", encoding="utf-8") as f:
   |

PTH123 `open()` should be replaced by `Path.open()`
  --> src/glopt/io/csv_writer.py:13:10
   |
11 |     out_path = os.path.join(csv_dir, f"{run_id}.csv")
12 |     first = True
13 |     with open(out_path, "w", newline="", encoding="utf-8") as f:
   |          ^^^^
14 |         writer = None
15 |         for r in rows:
   |

ANN204 Missing return type annotation for special method `__init__`
  --> src/glopt/io/csv_writer.py:26:9
   |
25 | class BenchmarkCSVWriter:
26 |     def __init__(self, output_dir: str = "runs/stats"):
   |         ^^^^^^^^
27 |         self.output_dir = output_dir
28 |         pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)
   |
help: Add return type annotation: `None`

D107 Missing docstring in `__init__`
  --> src/glopt/io/csv_writer.py:26:9
   |
25 | class BenchmarkCSVWriter:
26 |     def __init__(self, output_dir: str = "runs/stats"):
   |         ^^^^^^^^
27 |         self.output_dir = output_dir
28 |         pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)
   |

DTZ005 `datetime.datetime.now()` called without a `tz` argument
  --> src/glopt/io/csv_writer.py:29:21
   |
27 |         self.output_dir = output_dir
28 |         pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)
29 |         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   |                     ^^^^^^^^^^^^^^
30 |         self.csv_path = os.path.join(output_dir, f"{timestamp}.csv")
31 |         self.fieldnames = [
   |
help: Pass a `datetime.timezone` object to the `tz` parameter

PTH118 `os.path.join()` should be replaced by `Path` with `/` operator
  --> src/glopt/io/csv_writer.py:30:25
   |
28 |         pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)
29 |         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
30 |         self.csv_path = os.path.join(output_dir, f"{timestamp}.csv")
   |                         ^^^^^^^^^^^^
31 |         self.fieldnames = [
32 |             "algorithm",
   |

PTH123 `open()` should be replaced by `Path.open()`
  --> src/glopt/io/csv_writer.py:46:14
   |
44 |             "seed",
45 |         ]
46 |         with open(self.csv_path, "w", newline="") as csvfile:
   |              ^^^^
47 |             writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
48 |             writer.writeheader()
   |

ANN201 Missing return type annotation for public function `write_result`
  --> src/glopt/io/csv_writer.py:50:9
   |
48 |             writer.writeheader()
49 |
50 |     def write_result(self, result: dict[str, Any]):
   |         ^^^^^^^^^^^^
51 |         with open(self.csv_path, "a", newline="") as csvfile:
52 |             writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
   |
help: Add return type annotation: `None`

PTH123 `open()` should be replaced by `Path.open()`
  --> src/glopt/io/csv_writer.py:51:14
   |
50 |     def write_result(self, result: dict[str, Any]):
51 |         with open(self.csv_path, "a", newline="") as csvfile:
   |              ^^^^
52 |             writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
53 |             writer.writerow(result)
   |

ANN204 Missing return type annotation for special method `__init__`
  --> src/glopt/io/data_loader.py:9:9
   |
 8 | class RealWorldDataLoader:
 9 |     def __init__(self, data_dir: str = "data"):
   |         ^^^^^^^^
10 |         self.data_dir = Path(data_dir)
11 |         self.logger = logging.getLogger(__name__)
   |
help: Add return type annotation: `None`

D107 Missing docstring in `__init__`
  --> src/glopt/io/data_loader.py:9:9
   |
 8 | class RealWorldDataLoader:
 9 |     def __init__(self, data_dir: str = "data"):
   |         ^^^^^^^^
10 |         self.data_dir = Path(data_dir)
11 |         self.logger = logging.getLogger(__name__)
   |

TRY003 Avoid specifying long messages outside the exception class
  --> src/glopt/io/data_loader.py:18:19
   |
16 |         edges_file = facebook_dir / f"{ego_id}.edges"
17 |         if not edges_file.exists():
18 |             raise FileNotFoundError(f"Plik edges nie istnieje: {edges_file}")
   |                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
19 |
20 |         graph = nx.Graph()
   |

EM102 Exception must not use an f-string literal, assign to variable first
  --> src/glopt/io/data_loader.py:18:37
   |
16 |         edges_file = facebook_dir / f"{ego_id}.edges"
17 |         if not edges_file.exists():
18 |             raise FileNotFoundError(f"Plik edges nie istnieje: {edges_file}")
   |                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
19 |
20 |         graph = nx.Graph()
   |
help: Assign to variable; remove f-string literal

PTH123 `open()` should be replaced by `Path.open()`
  --> src/glopt/io/data_loader.py:25:14
   |
23 |         graph.add_node(ego_node, is_ego=True)
24 |
25 |         with open(edges_file) as f:
   |              ^^^^
26 |             for line in f:
27 |                 line = line.strip()
   |

PLW2901 `for` loop variable `line` overwritten by assignment target
  --> src/glopt/io/data_loader.py:27:17
   |
25 |         with open(edges_file) as f:
26 |             for line in f:
27 |                 line = line.strip()
   |                 ^^^^
28 |                 if line:
29 |                     parts = line.split()
   |

PLR2004 Magic value used in comparison, consider replacing `2` with a constant variable
  --> src/glopt/io/data_loader.py:30:38
   |
28 |                 if line:
29 |                     parts = line.split()
30 |                     if len(parts) >= 2:
   |                                      ^
31 |                         node1, node2 = int(parts[0]), int(parts[1])
32 |                         graph.add_edge(node1, node2)
   |

G004 Logging statement uses f-string
  --> src/glopt/io/data_loader.py:42:13
   |
41 |         self.logger.info(
42 |             f"Załadowano Facebook ego network {ego_id}: {len(graph.nodes())} węzłów, {len(graph.edges())} krawędzi"
   |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
43 |         )
   |
help: Convert to lazy `%` formatting

TRY003 Avoid specifying long messages outside the exception class
  --> src/glopt/io/data_loader.py:52:19
   |
51 |         if not facebook_dir.exists():
52 |             raise FileNotFoundError(f"Katalog Facebook nie istnieje: {facebook_dir}")
   |                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
53 |
54 |         edge_files = list(facebook_dir.glob("*.edges"))
   |

EM102 Exception must not use an f-string literal, assign to variable first
  --> src/glopt/io/data_loader.py:52:37
   |
51 |         if not facebook_dir.exists():
52 |             raise FileNotFoundError(f"Katalog Facebook nie istnieje: {facebook_dir}")
   |                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
53 |
54 |         edge_files = list(facebook_dir.glob("*.edges"))
   |
help: Assign to variable; remove f-string literal

BLE001 Do not catch blind exception: `Exception`
  --> src/glopt/io/data_loader.py:61:20
   |
59 |                 network = self.load_facebook_ego_network(ego_id)
60 |                 networks[ego_id] = network
61 |             except Exception as e:
   |                    ^^^^^^^^^
62 |                 self.logger.warning("Nie udało się załadować network %s: %s", ego_id, e)
   |

G004 Logging statement uses f-string
  --> src/glopt/io/data_loader.py:64:26
   |
62 |                 self.logger.warning("Nie udało się załadować network %s: %s", ego_id, e)
63 |
64 |         self.logger.info(f"Załadowano {len(networks)} Facebook ego networks")
   |                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
65 |         return networks
   |
help: Convert to lazy `%` formatting

B007 Loop control variable `ego_id` not used within loop body
   --> src/glopt/io/data_loader.py:99:13
    |
 97 |         node_offset = 0
 98 |
 99 |         for ego_id, graph in networks.items():
    |             ^^^^^^
100 |             mapping = {old_id: old_id + node_offset for old_id in graph.nodes()}
101 |             shifted_graph = nx.relabel_nodes(graph, mapping)
    |
help: Rename unused `ego_id` to `_ego_id`

PERF102 When using only the values of a dict use the `values()` method
   --> src/glopt/io/data_loader.py:99:30
    |
 97 |         node_offset = 0
 98 |
 99 |         for ego_id, graph in networks.items():
    |                              ^^^^^^^^^^^^^^
100 |             mapping = {old_id: old_id + node_offset for old_id in graph.nodes()}
101 |             shifted_graph = nx.relabel_nodes(graph, mapping)
    |
help: Replace `.items()` with `.values()`

G004 Logging statement uses f-string
   --> src/glopt/io/data_loader.py:108:13
    |
107 |         self.logger.info(
108 |             f"Utworzono połączony graf Facebook: {len(combined_graph.nodes())} węzłów, {len(combined_graph.edges())} krawędzi"
    |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
109 |         )
    |
help: Convert to lazy `%` formatting

C901 `_load_node_features` is too complex (12 > 10)
   --> src/glopt/io/data_loader.py:113:9
    |
111 |         return combined_graph
112 |
113 |     def _load_node_features(self, graph: nx.Graph, data_dir: Path, ego_id: str) -> None:
    |         ^^^^^^^^^^^^^^^^^^^
114 |         feat_file = data_dir / f"{ego_id}.feat"
115 |         egofeat_file = data_dir / f"{ego_id}.egofeat"
    |

PTH123 `open()` should be replaced by `Path.open()`
   --> src/glopt/io/data_loader.py:120:18
    |
118 |         feature_names = []
119 |         if featnames_file.exists():
120 |             with open(featnames_file) as f:
    |                  ^^^^
121 |                 for line in f:
122 |                     line = line.strip()
    |

PLW2901 `for` loop variable `line` overwritten by assignment target
   --> src/glopt/io/data_loader.py:122:21
    |
120 |             with open(featnames_file) as f:
121 |                 for line in f:
122 |                     line = line.strip()
    |                     ^^^^
123 |                     if line:
124 |                         parts = line.split(maxsplit=1)
    |

PLR2004 Magic value used in comparison, consider replacing `2` with a constant variable
   --> src/glopt/io/data_loader.py:125:42
    |
123 |                     if line:
124 |                         parts = line.split(maxsplit=1)
125 |                         if len(parts) >= 2:
    |                                          ^
126 |                             feature_names.append(parts[1])
    |

PTH123 `open()` should be replaced by `Path.open()`
   --> src/glopt/io/data_loader.py:129:18
    |
128 |         if feat_file.exists():
129 |             with open(feat_file) as f:
    |                  ^^^^
130 |                 for line in f:
131 |                     line = line.strip()
    |

PLW2901 `for` loop variable `line` overwritten by assignment target
   --> src/glopt/io/data_loader.py:131:21
    |
129 |             with open(feat_file) as f:
130 |                 for line in f:
131 |                     line = line.strip()
    |                     ^^^^
132 |                     if line:
133 |                         parts = line.split()
    |

PLR2004 Magic value used in comparison, consider replacing `2` with a constant variable
   --> src/glopt/io/data_loader.py:134:42
    |
132 |                     if line:
133 |                         parts = line.split()
134 |                         if len(parts) >= 2:
    |                                          ^
135 |                             node_id = int(parts[0])
136 |                             features = [int(x) for x in parts[1:]]
    |

PTH123 `open()` should be replaced by `Path.open()`
   --> src/glopt/io/data_loader.py:144:18
    |
142 |         ego_node = int(ego_id)
143 |         if egofeat_file.exists() and ego_node in graph.nodes():
144 |             with open(egofeat_file) as f:
    |                  ^^^^
145 |                 line = f.readline().strip()
146 |                 if line:
    |

PTH123 `open()` should be replaced by `Path.open()`
   --> src/glopt/io/data_loader.py:158:14
    |
157 |         circles = []
158 |         with open(circles_file) as f:
    |              ^^^^
159 |             for line in f:
160 |                 line = line.strip()
    |

PLW2901 `for` loop variable `line` overwritten by assignment target
   --> src/glopt/io/data_loader.py:160:17
    |
158 |         with open(circles_file) as f:
159 |             for line in f:
160 |                 line = line.strip()
    |                 ^^^^
161 |                 if line:
162 |                     parts = line.split()
    |

PLR2004 Magic value used in comparison, consider replacing `2` with a constant variable
   --> src/glopt/io/data_loader.py:163:38
    |
161 |                 if line:
162 |                     parts = line.split()
163 |                     if len(parts) >= 2:
    |                                      ^
164 |                         circle_name = parts[0]
165 |                         circle_members = [int(x) for x in parts[1:] if x.isdigit()]
    |

PTH123 `open()` should be replaced by `Path.open()`
   --> src/glopt/io/data_loader.py:184:14
    |
183 |         circles = []
184 |         with open(circles_file) as f:
    |              ^^^^
185 |             for line in f:
186 |                 line = line.strip()
    |

PLW2901 `for` loop variable `line` overwritten by assignment target
   --> src/glopt/io/data_loader.py:186:17
    |
184 |         with open(circles_file) as f:
185 |             for line in f:
186 |                 line = line.strip()
    |                 ^^^^
187 |                 if line:
188 |                     parts = line.split()
    |

PLR2004 Magic value used in comparison, consider replacing `2` with a constant variable
   --> src/glopt/io/data_loader.py:189:38
    |
187 |                 if line:
188 |                     parts = line.split()
189 |                     if len(parts) >= 2:
    |                                      ^
190 |                         circle_name = parts[0]
191 |                         circle_size = len(parts) - 1
    |

PTH118 `os.path.join()` should be replaced by `Path` with `/` operator
  --> src/glopt/io/fs.py:12:12
   |
11 | def build_paths(run_id: str) -> tuple[str, str, str]:
12 |     base = os.path.join("runs", run_id)
   |            ^^^^^^^^^^^^
13 |     graphs_dir = os.path.join(base, "graphs")
14 |     csv_dir = os.path.join(base, "csv")
   |

PTH118 `os.path.join()` should be replaced by `Path` with `/` operator
  --> src/glopt/io/fs.py:13:18
   |
11 | def build_paths(run_id: str) -> tuple[str, str, str]:
12 |     base = os.path.join("runs", run_id)
13 |     graphs_dir = os.path.join(base, "graphs")
   |                  ^^^^^^^^^^^^
14 |     csv_dir = os.path.join(base, "csv")
15 |     ensure_dir(graphs_dir)
   |

PTH118 `os.path.join()` should be replaced by `Path` with `/` operator
  --> src/glopt/io/fs.py:14:15
   |
12 |     base = os.path.join("runs", run_id)
13 |     graphs_dir = os.path.join(base, "graphs")
14 |     csv_dir = os.path.join(base, "csv")
   |               ^^^^^^^^^^^^
15 |     ensure_dir(graphs_dir)
16 |     ensure_dir(csv_dir)
   |

RUF012 Mutable class attributes should be annotated with `typing.ClassVar`
  --> src/glopt/io/graph_generator.py:9:43
   |
 8 |   class GraphGeneratorFactory:
 9 |       _GENERATORS: dict[str, GeneratorFn] = {
   |  ___________________________________________^
10 | |         "random": lambda *, n_nodes, **p: GraphGeneratorFactory._random(n_nodes, **p),
11 | |         "scale_free": lambda *, n_nodes, **p: GraphGeneratorFactory._scale_free(n_nodes, **p),
12 | |         "small_world": lambda *, n_nodes, **p: GraphGeneratorFactory._small_world(n_nodes, **p),
13 | |         "complete": lambda *, n_nodes, **p: GraphGeneratorFactory._complete(n_nodes, **p),
14 | |         "star": lambda *, n_nodes, **p: GraphGeneratorFactory._star(n_nodes, **p),
15 | |         "path": lambda *, n_nodes, **p: GraphGeneratorFactory._path(n_nodes, **p),
16 | |         "cycle": lambda *, n_nodes, **p: GraphGeneratorFactory._cycle(n_nodes, **p),
17 | |         "tree": lambda *, n_nodes, **p: GraphGeneratorFactory._tree(n_nodes, **p),
18 | |     }
   | |_____^
19 |
20 |       @classmethod
   |

B904 Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
  --> src/glopt/io/graph_generator.py:26:13
   |
24 |         except KeyError:
25 |             available = ", ".join(cls._GENERATORS.keys())
26 |             raise ValueError(f"unknown graph generator '{name}'. available: {available}")
   |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
27 |
28 |     @staticmethod
   |

TRY003 Avoid specifying long messages outside the exception class
  --> src/glopt/io/graph_generator.py:26:19
   |
24 |         except KeyError:
25 |             available = ", ".join(cls._GENERATORS.keys())
26 |             raise ValueError(f"unknown graph generator '{name}'. available: {available}")
   |                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
27 |
28 |     @staticmethod
   |

EM102 Exception must not use an f-string literal, assign to variable first
  --> src/glopt/io/graph_generator.py:26:30
   |
24 |         except KeyError:
25 |             available = ", ".join(cls._GENERATORS.keys())
26 |             raise ValueError(f"unknown graph generator '{name}'. available: {available}")
   |                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
27 |
28 |     @staticmethod
   |
help: Assign to variable; remove f-string literal

PLC0415 `import` should be at the top-level of a file
  --> src/glopt/io/graph_generator.py:58:9
   |
56 |     @staticmethod
57 |     def _tree(n_nodes: int, *, seed: int | None = None) -> nx.Graph:
58 |         import networkx as nx
   |         ^^^^^^^^^^^^^^^^^^^^^
59 |
60 |         if n_nodes == 1:
   |

N806 Variable `G` in function should be lowercase
  --> src/glopt/io/graph_generator.py:61:13
   |
60 |         if n_nodes == 1:
61 |             G = nx.Graph()
   |             ^
62 |             G.add_node(0)
63 |             return G
   |

ICN001 `matplotlib` should be imported as `mpl`
 --> src/glopt/io/graph_visualizer.py:4:8
  |
2 | from typing import Any
3 |
4 | import matplotlib
  |        ^^^^^^^^^^
5 |
6 | matplotlib.use("Agg")
  |
help: Alias `matplotlib` to `mpl`

TID252 Prefer absolute imports over relative imports from parent modules
  --> src/glopt/io/graph_visualizer.py:13:1
   |
11 | import networkx as nx
12 |
13 | from ..core import Solution
   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^
   |
help: Replace relative imports from parent modules with absolute imports

ANN204 Missing return type annotation for special method `__init__`
  --> src/glopt/io/graph_visualizer.py:17:9
   |
16 | class GraphVisualizer:
17 |     def __init__(
   |         ^^^^^^^^
18 |         self,
19 |         figsize: tuple[int, int] = (12, 8),
   |
help: Add return type annotation: `None`

D107 Missing docstring in `__init__`
  --> src/glopt/io/graph_visualizer.py:17:9
   |
16 | class GraphVisualizer:
17 |     def __init__(
   |         ^^^^^^^^
18 |         self,
19 |         figsize: tuple[int, int] = (12, 8),
   |

FBT001 Boolean-typed positional argument in function definition
  --> src/glopt/io/graph_visualizer.py:21:9
   |
19 |         figsize: tuple[int, int] = (12, 8),
20 |         layout_seed: int = 42,
21 |         reuse_layout: bool = True,
   |         ^^^^^^^^^^^^
22 |     ):
23 |         self.figsize = figsize
   |

FBT002 Boolean default positional argument in function definition
  --> src/glopt/io/graph_visualizer.py:21:9
   |
19 |         figsize: tuple[int, int] = (12, 8),
20 |         layout_seed: int = 42,
21 |         reuse_layout: bool = True,
   |         ^^^^^^^^^^^^
22 |     ):
23 |         self.figsize = figsize
   |

DTZ005 `datetime.datetime.now()` called without a `tz` argument
  --> src/glopt/io/graph_visualizer.py:60:36
   |
58 |         if save_path is None:
59 |             if timestamp_folder is None:
60 |                 timestamp_folder = datetime.now().strftime("%Y%m%d_%H%M%S")
   |                                    ^^^^^^^^^^^^^^
61 |             n_nodes, n_edges = graph.number_of_nodes(), graph.number_of_edges()
62 |             save_path = f"runs/graphs/{timestamp_folder}/{solver_name}_{n_nodes}n_{n_edges}e.png"
   |
help: Pass a `datetime.timezone` object to the `tz` parameter

S101 Use of `assert` detected
  --> src/glopt/io/graph_visualizer.py:77:9
   |
76 |     def _update_positions_for_graph(self, graph: nx.Graph) -> None:
77 |         assert self._pos is not None
   |         ^^^^^^
78 |         g_nodes = set(graph.nodes())
79 |         pos_nodes = set(self._pos.keys())
   |

ANN001 Missing type annotation for function argument `ax`
   --> src/glopt/io/graph_visualizer.py:131:27
    |
129 |         return colors
130 |
131 |     def _add_legend(self, ax, solution: Solution) -> None:
    |                           ^^
132 |         license_types = sorted({g.license_type for g in solution.groups}, key=lambda lt: lt.name)
133 |         if not license_types:
    |

PLC0415 `import` should be at the top-level of a file
   --> src/glopt/io/graph_visualizer.py:144:5
    |
143 | def jitter(scale: float = 0.08) -> float:
144 |     import random as _r
    |     ^^^^^^^^^^^^^^^^^^^
145 |
146 |     return (_r.random() - 0.5) * 2.0 * scale
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/io/graph_visualizer.py:146:13
    |
144 |     import random as _r
145 |
146 |     return (_r.random() - 0.5) * 2.0 * scale
    |             ^^^^^^^^^^^
    |

ANN201 Missing return type annotation for public function `random_choice`
   --> src/glopt/io/graph_visualizer.py:149:5
    |
149 | def random_choice(seq):
    |     ^^^^^^^^^^^^^
150 |     import random as _r
    |
help: Add return type annotation

ANN001 Missing type annotation for function argument `seq`
   --> src/glopt/io/graph_visualizer.py:149:19
    |
149 | def random_choice(seq):
    |                   ^^^
150 |     import random as _r
    |

PLC0415 `import` should be at the top-level of a file
   --> src/glopt/io/graph_visualizer.py:150:5
    |
149 | def random_choice(seq):
150 |     import random as _r
    |     ^^^^^^^^^^^^^^^^^^^
151 |
152 |     return _r.choice(list(seq))
    |

S311 Standard pseudo-random generators are not suitable for cryptographic purposes
   --> src/glopt/io/graph_visualizer.py:152:12
    |
150 |     import random as _r
151 |
152 |     return _r.choice(list(seq))
    |            ^^^^^^^^^^^^^^^^^^^^
    |

RUF012 Mutable class attributes should be annotated with `typing.ClassVar`
  --> src/glopt/license_config.py:11:60
   |
 9 |       GREEN = "#5d9f49"
10 |
11 |       _CONFIGS: dict[str, Callable[[], list[LicenseType]]] = {
   |  ____________________________________________________________^
12 | |         "duolingo_super": lambda: [
13 | |             LicenseType("Individual", 13.99, 1, 1, LicenseConfigFactory.PURPLE),
14 | |             LicenseType("Family", 29.17, 2, 6, LicenseConfigFactory.GOLD),
15 | |         ],
16 | |         "spotify": lambda: [
17 | |             LicenseType("Individual", 23.99, 1, 1, LicenseConfigFactory.PURPLE),
18 | |             LicenseType("Duo", 30.99, 2, 2, LicenseConfigFactory.GREEN),
19 | |             LicenseType("Family", 37.99, 2, 6, LicenseConfigFactory.GOLD),
20 | |         ],
21 | |         "roman_domination": lambda: [
22 | |             LicenseType("Solo", 1.0, 1, 1, LicenseConfigFactory.PURPLE),
23 | |             LicenseType("Group", 2.0, 2, 999, LicenseConfigFactory.GOLD),
24 | |         ],
25 | |     }
   | |_____^
26 |
27 |       @classmethod
   |

B904 Within an `except` clause, raise exceptions with `raise ... from err` or `raise ... from None` to distinguish them from errors in exception handling
  --> src/glopt/license_config.py:33:13
   |
31 |         except KeyError:
32 |             available = ", ".join(cls._CONFIGS.keys())
33 |             raise ValueError(f"Unsupported license config: {name}. Available: {available}")
   |             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   |

TRY003 Avoid specifying long messages outside the exception class
  --> src/glopt/license_config.py:33:19
   |
31 |         except KeyError:
32 |             available = ", ".join(cls._CONFIGS.keys())
33 |             raise ValueError(f"Unsupported license config: {name}. Available: {available}")
   |                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   |

EM102 Exception must not use an f-string literal, assign to variable first
  --> src/glopt/license_config.py:33:30
   |
31 |         except KeyError:
32 |             available = ", ".join(cls._CONFIGS.keys())
33 |             raise ValueError(f"Unsupported license config: {name}. Available: {available}")
   |                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   |
help: Assign to variable; remove f-string literal

INP001 File `tests/test_algorithms.py` is part of an implicit namespace package. Add an `__init__.py`.
--> tests/test_algorithms.py:1:1

PT006 Wrong type passed to first argument of `pytest.mark.parametrize`; expected `tuple`
  --> tests/test_algorithms.py:51:26
   |
49 | @pytest.mark.parametrize("license_cfg", LICENSE_CFGS)
50 | @pytest.mark.parametrize("graph_name", list(GRAPH_SPECS.keys()))
51 | @pytest.mark.parametrize("algo_id,algo_factory,algo_kwargs,n_nodes", ALGOS, ids=[a[0] for a in ALGOS])
   |                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
52 | def test_algorithms_validity(graph_name: str, license_cfg: str, algo_id, algo_factory, algo_kwargs, n_nodes: int):
53 |     license_types = LicenseConfigFactory.get_config(license_cfg)
   |
help: Use a `tuple` for the first argument

PLR0913 Too many arguments in function definition (6 > 5)
  --> tests/test_algorithms.py:52:5
   |
50 | @pytest.mark.parametrize("graph_name", list(GRAPH_SPECS.keys()))
51 | @pytest.mark.parametrize("algo_id,algo_factory,algo_kwargs,n_nodes", ALGOS, ids=[a[0] for a in ALGOS])
52 | def test_algorithms_validity(graph_name: str, license_cfg: str, algo_id, algo_factory, algo_kwargs, n_nodes: int):
   |     ^^^^^^^^^^^^^^^^^^^^^^^^
53 |     license_types = LicenseConfigFactory.get_config(license_cfg)
54 |     graph = generate_graph(graph_name, n_nodes)
   |

ANN201 Missing return type annotation for public function `test_algorithms_validity`
  --> tests/test_algorithms.py:52:5
   |
50 | @pytest.mark.parametrize("graph_name", list(GRAPH_SPECS.keys()))
51 | @pytest.mark.parametrize("algo_id,algo_factory,algo_kwargs,n_nodes", ALGOS, ids=[a[0] for a in ALGOS])
52 | def test_algorithms_validity(graph_name: str, license_cfg: str, algo_id, algo_factory, algo_kwargs, n_nodes: int):
   |     ^^^^^^^^^^^^^^^^^^^^^^^^
53 |     license_types = LicenseConfigFactory.get_config(license_cfg)
54 |     graph = generate_graph(graph_name, n_nodes)
   |
help: Add return type annotation: `None`

ANN001 Missing type annotation for function argument `algo_id`
  --> tests/test_algorithms.py:52:65
   |
50 | @pytest.mark.parametrize("graph_name", list(GRAPH_SPECS.keys()))
51 | @pytest.mark.parametrize("algo_id,algo_factory,algo_kwargs,n_nodes", ALGOS, ids=[a[0] for a in ALGOS])
52 | def test_algorithms_validity(graph_name: str, license_cfg: str, algo_id, algo_factory, algo_kwargs, n_nodes: int):
   |                                                                 ^^^^^^^
53 |     license_types = LicenseConfigFactory.get_config(license_cfg)
54 |     graph = generate_graph(graph_name, n_nodes)
   |

ARG001 Unused function argument: `algo_id`
  --> tests/test_algorithms.py:52:65
   |
50 | @pytest.mark.parametrize("graph_name", list(GRAPH_SPECS.keys()))
51 | @pytest.mark.parametrize("algo_id,algo_factory,algo_kwargs,n_nodes", ALGOS, ids=[a[0] for a in ALGOS])
52 | def test_algorithms_validity(graph_name: str, license_cfg: str, algo_id, algo_factory, algo_kwargs, n_nodes: int):
   |                                                                 ^^^^^^^
53 |     license_types = LicenseConfigFactory.get_config(license_cfg)
54 |     graph = generate_graph(graph_name, n_nodes)
   |

ANN001 Missing type annotation for function argument `algo_factory`
  --> tests/test_algorithms.py:52:74
   |
50 | @pytest.mark.parametrize("graph_name", list(GRAPH_SPECS.keys()))
51 | @pytest.mark.parametrize("algo_id,algo_factory,algo_kwargs,n_nodes", ALGOS, ids=[a[0] for a in ALGOS])
52 | def test_algorithms_validity(graph_name: str, license_cfg: str, algo_id, algo_factory, algo_kwargs, n_nodes: int):
   |                                                                          ^^^^^^^^^^^^
53 |     license_types = LicenseConfigFactory.get_config(license_cfg)
54 |     graph = generate_graph(graph_name, n_nodes)
   |

ANN001 Missing type annotation for function argument `algo_kwargs`
  --> tests/test_algorithms.py:52:88
   |
50 | @pytest.mark.parametrize("graph_name", list(GRAPH_SPECS.keys()))
51 | @pytest.mark.parametrize("algo_id,algo_factory,algo_kwargs,n_nodes", ALGOS, ids=[a[0] for a in ALGOS])
52 | def test_algorithms_validity(graph_name: str, license_cfg: str, algo_id, algo_factory, algo_kwargs, n_nodes: int):
   |                                                                                        ^^^^^^^^^^^
53 |     license_types = LicenseConfigFactory.get_config(license_cfg)
54 |     graph = generate_graph(graph_name, n_nodes)
   |

S101 Use of `assert` detected
  --> tests/test_algorithms.py:58:5
   |
56 |     solution = algo.solve(graph=graph, license_types=license_types, **algo_kwargs)
57 |     ok, issues = validator.validate(solution, graph)
58 |     assert ok, f"{algo.name} invalid: {issues}"
   |     ^^^^^^
   |

Found 389 errors.
No fixes available (143 hidden fixes can be enabled with the `--unsafe-fixes` option).
