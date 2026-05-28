import sys
import math
import random
import time

from roar_net_api.algorithms import greedy_construction

from Construction import Problem, repair_min_size
from Local_search import SwapMove, can_join


def random_swap_move(p, s, max_tries=200):
    """Sample a feasible random swap of two students on different teams, or None
    if no valid swap is found within max_tries attempts."""
    for _ in range(max_tries):
        a = random.randrange(p.n)
        b = random.randrange(p.n)
        if a == b or s.team[a] == s.team[b]:
            continue
        if can_join(a, s.team[b], b, s) and can_join(b, s.team[a], a, s):
            return SwapMove(a, s.team[a], b, s.team[b])
    return None


def simulated_annealing(solution, time_limit=60):
    """Simulated annealing over the swap neighbourhood. Calibrates the start
    temperature for ~97% initial acceptance, cools geometrically, and returns the
    best solution seen within the time limit."""
    p = solution.problem

    s        = solution.copy_solution()
    best     = s.copy_solution()

    sample = []
    for _ in range(200):
        m = random_swap_move(p, s)
        if m is not None:
            sample.append(abs(m.objective_value_increment(s)))
    avg_delta = (sum(sample) / len(sample)) if sample else 1.0
    T0 = avg_delta / abs(math.log(0.97))
    T  = T0

    alpha = 0.97

    steps_per_temp = max(100, 10 * p.n)

    print(f"  T0={T0:.2f}, steps/temp={steps_per_temp}")

    start      = time.time()
    iterations = 0
    accepted   = 0
    proposed   = 0

    while time.time() - start < time_limit:

        for _ in range(steps_per_temp):
            move = random_swap_move(p, s)
            if move is None:
                break

            delta    = move.objective_value_increment(s)
            proposed += 1

            if delta < 0 or random.random() < math.exp(-delta / T):
                move.apply_move(s)
                accepted += 1

                if s.objective_value() < best.objective_value():
                    best = s.copy_solution()

        T          *= alpha
        iterations += 1

        if proposed > 0 and accepted / proposed < 0.02:
            print(f"  Early stop: acceptance ratio {accepted/proposed:.2%}")
            break

    elapsed = time.time() - start
    ratio = accepted / proposed if proposed else 0
    print(f"  Iterations={iterations}, Accept={ratio:.2%}, Time={elapsed:.1f}s")
    return best


def RunMeta(instance_file, solution_file):
    """Build a greedy start, repair team sizes, run simulated annealing, write the
    result and return its objective."""
    p = Problem(instance_file)

    s  = greedy_construction(p)
    repair_min_size(s)
    s  = simulated_annealing(s, time_limit=60)
    ov = s.objective_value()

    with open(solution_file, 'w') as f:
        f.write(' '.join(map(str, s.team)) + '\n')
    print(f"Solution written to {solution_file}")

    return ov


if __name__ == '__main__':
    instance_file, solution_file = sys.argv[1], sys.argv[2]
    RunMeta(instance_file, solution_file)
