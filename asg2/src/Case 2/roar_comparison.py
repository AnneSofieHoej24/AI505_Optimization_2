"""Compare our own heuristics against the ROAR-NET API reference algorithms.

Our construction/solution/move classes already implement the ROAR-NET
construction operations, so `greedy_construction` runs on them directly. The
reference local-search and SA algorithms additionally need
`local_neighbourhood()` on the problem and `random_moves_without_replacement()`
on the neighbourhood; we supply those here as thin adapters so the original
modules stay untouched.

Run:
    python roar_comparison.py --path <instance.txt> [--runs N] [--budget S] [--seed K]
"""

import argparse
import contextlib
import io
import math
import random

from Construction import Problem as GreedyProblem, repair_min_size, run_construction
from ConstructionR import Problem as RandomProblem
from Local_search import LocalNeighbourhood, best_improvement as own_best_improvement
from Meta import simulated_annealing as own_sa, random_swap_move

from roar_net_api.algorithms import (
    greedy_construction,
    best_improvement as roar_best_improvement,
    sa as roar_sa,
)


class _SearchNeighbourhood(LocalNeighbourhood):
    """LocalNeighbourhood plus the random-order iteration the SA reference needs."""

    def random_moves_without_replacement(self, solution):
        moves = list(self.moves(solution))
        random.shuffle(moves)
        yield from moves


class _SearchProblem:
    """Adapter exposing local_neighbourhood() so roar's LS/SA can drive our model."""

    def __init__(self, problem):
        self.problem = problem

    def local_neighbourhood(self):
        return _SearchNeighbourhood(self.problem)


def _calibrate_init_temp(problem, solution):
    # Same calibration our own SA uses: ~97% acceptance of opening moves.
    sample = []
    for _ in range(200):
        m = random_swap_move(problem, solution)
        if m is not None:
            sample.append(abs(m.objective_value_increment(solution)))
    avg_delta = (sum(sample) / len(sample)) if sample else 1.0
    return avg_delta / abs(math.log(0.97))


def compare_greedy(path):
    # Deterministic construction; one run each is enough.
    with contextlib.redirect_stdout(io.StringIO()):
        own = run_construction(path, "/dev/null")

    p = GreedyProblem(path)
    s = greedy_construction(p)
    repair_min_size(s)
    roar = s.objective_value()
    return own, roar


def compare_randomized(path, runs):
    # ConstructionR is roar's greedy_construction driven by our randomized
    # neighbourhood; check whether that beats roar's plain greedy.
    objs = []
    for _ in range(runs):
        s = greedy_construction(RandomProblem(path))
        ov = s.objective_value()
        if ov is not None:
            objs.append(ov)
    best = min(objs)
    median = sorted(objs)[len(objs) // 2]
    return best, median, len(objs)


def compare_local_search(path):
    # Identical greedy start for both, so any difference is the search itself.
    p = RandomProblem(path)
    start = greedy_construction(p)
    own = own_best_improvement(start.copy_solution()).objective_value()
    roar = roar_best_improvement(_SearchProblem(p), start.copy_solution()).objective_value()
    return own, roar


def compare_sa(path, budget):
    p = GreedyProblem(path)
    start = greedy_construction(p)
    repair_min_size(start)
    init_temp = _calibrate_init_temp(p, start)

    with contextlib.redirect_stdout(io.StringIO()):
        own = own_sa(start.copy_solution(), time_limit=budget).objective_value()
    roar = roar_sa(_SearchProblem(p), start.copy_solution(), budget, init_temp).objective_value()
    return own, roar


def _summary(label, pairs):
    own_best = min(o for o, _ in pairs)
    roar_best = min(r for _, r in pairs)
    own_med = sorted(o for o, _ in pairs)[len(pairs) // 2]
    roar_med = sorted(r for _, r in pairs)[len(pairs) // 2]
    print(f"{label:<14} own: best={own_best} median={own_med}    "
          f"roar: best={roar_best} median={roar_med}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Own heuristics vs ROAR-NET reference algorithms.")
    parser.add_argument("--path", required=True, help="instance .txt file")
    parser.add_argument("--runs", type=int, default=5, help="repetitions for local search / SA")
    parser.add_argument("--rand-runs", type=int, default=10, help="repetitions for randomized construction")
    parser.add_argument("--budget", type=float, default=10.0, help="SA time budget per run, seconds")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    print(f"Instance: {args.path}  (runs={args.runs}, sa budget={args.budget}s)\n")

    own_g, roar_g = compare_greedy(args.path)
    print(f"{'greedy':<14} own: {own_g}    roar: {roar_g}  (deterministic)")

    rand_best, rand_med, n_ok = compare_randomized(args.path, args.rand_runs)
    note = "" if n_ok == args.rand_runs else f"  ({n_ok}/{args.rand_runs} complete)"
    print(f"{'randomized':<14} best={rand_best} median={rand_med} over {args.rand_runs} runs    "
          f"vs roar greedy baseline={roar_g}{note}")

    ls_pairs = [compare_local_search(args.path) for _ in range(args.runs)]
    _summary("local search", ls_pairs)

    sa_pairs = [compare_sa(args.path, args.budget) for _ in range(args.runs)]
    _summary("simulated ann", sa_pairs)
