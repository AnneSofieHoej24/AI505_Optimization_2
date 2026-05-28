# AI505 Optimization — Assignment 2

Two independent cases:

- **Case 1 — Interior-point centering.** Finds the analytic center of a polytope
  `A x <= b` via a log-barrier and Newton's method, then studies scaling
  invariance and compares gradient descent against Newton on a larger
  box-constrained barrier problem.
- **Case 2 — Team formation.** Partitions students into teams that maximize
  weighted attribute diversity while respecting team-size and pairwise
  disagreement constraints. Greedy + randomized construction, best-improvement
  local search, and a simulated-annealing metaheuristic, built on the
  ROAR-NET API.

## Layout

```
asg2/
├── data/                 # team-formation problem instances (.txt) + one .sol
└── src/
    ├── Case 1/           # interior-point centering
    └── Case 2/           # team formation
```

## Requirements

Python 3.13+ and:

```
numpy
scipy
matplotlib        # Case 1 plots
roar_net_api      # Case 2 construction / search primitives
```

## Running

### Case 1

```
cd "src/Case 1"
python main.py
```

Runs Task 3.1 → 3.2 → 3.3 → 4 in order, prints results to stdout, and writes
plots to `src/Case 1/figures/`.

### Case 2

```
cd "src/Case 2"
python SolutionChecker.py --vals <method> <runs> --path <instance.txt> [--seed N]
```

- `<method>`: `0` greedy, `1` randomized, `2` local search, `3` metaheuristic
- `<runs>`: number of repeated runs (stats are aggregated across them)
- `--path`: instance file from `data/`
- `--seed`: optional seed for the randomized methods

Each run writes its solution to `solution_<pid>.txt`, validates it with the
checker, and prints per-run objectives plus `best / median / mean / worst / std`.
The objective is **minimized**.

Example:

```
python SolutionChecker.py --vals 3 10 --path ../../data/tfp_200n_40q_5l_5u_10a_15d.txt --seed 42
```

`checker.py` and `team_formation_generator.py` are course-provided
(Apache-2.0); all other files are our own implementation.
