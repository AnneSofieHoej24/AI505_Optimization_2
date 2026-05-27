import sys
import math
import random
import time

from roar_net_api.algorithms import greedy_construction

from Construction import Problem, repair_min_size
from Local_search import SwapMove, can_join


def random_swap_move(p, s, max_tries=200):
    for _ in range(max_tries):
        a = random.randrange(p.n)
        b = random.randrange(p.n)
        if a == b or s.team[a] == s.team[b]:
            continue
        if can_join(a, s.team[b], b, s) and can_join(b, s.team[a], a, s):
            return SwapMove(a, s.team[a], b, s.team[b])
    return None


# ── Simulated Annealing ───────────────────────────────────────────────────────

def simulated_annealing(solution, time_limit=60):
    p = solution.problem

    s        = solution.copy_solution()
    best     = s.copy_solution()

    # Estimate T0 from average abs delta of sampled random moves
    # so that exp(-avg_delta / T0) ≈ 0.97 (97% accept at start)
    sample = []
    for _ in range(200):
        m = random_swap_move(p, s)
        if m is not None:
            sample.append(abs(m.objective_value_increment(s)))
    avg_delta = (sum(sample) / len(sample)) if sample else 1.0
    T0 = avg_delta / abs(math.log(0.97))
    T  = T0

    # Geometric cooling: T := alpha * T after each temperature level
    alpha = 0.97

    # Steps per temperature level: proportional to n
    steps_per_temp = max(100, 10 * p.n)

    print(f"  T0={T0:.2f}, steps/temp={steps_per_temp}")

    start      = time.time()
    iterations = 0
    accepted   = 0
    proposed   = 0

    while time.time() - start < time_limit:

        # Inner loop: keep temperature fixed for steps_per_temp steps
        for _ in range(steps_per_temp):
            # Sample a random valid swap directly (no full neighbourhood enumeration)
            move = random_swap_move(p, s)
            if move is None:
                break

            delta    = move.objective_value_increment(s)
            proposed += 1

            # Metropolis acceptance criterion:
            # always accept improvements, accept worsening with exp(-delta/T)
            if delta < 0 or random.random() < math.exp(-delta / T):
                move.apply_move(s)
                accepted += 1

                # Track best solution found so far
                if s.objective_value() < best.objective_value():
                    best = s.copy_solution()

        # Update temperature (geometric cooling)
        T          *= alpha
        iterations += 1

        # Termination: stop early if acceptance ratio drops below 2%
        if proposed > 0 and accepted / proposed < 0.02:
            print(f"  Early stop: acceptance ratio {accepted/proposed:.2%}")
            break

    elapsed = time.time() - start
    ratio = accepted / proposed if proposed else 0
    print(f"  Iterations={iterations}, Accept={ratio:.2%}, Time={elapsed:.1f}s")
    return best

def RunMeta(instance_file, solution_file):
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
    RunMeta(instance_file,solution_file)

    
