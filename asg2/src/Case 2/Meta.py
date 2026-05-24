import sys
import math
import random
import time

from Construction import Problem, greedy_construction
from Local_search import LocalNeighbourhood


# ── Simulated Annealing ───────────────────────────────────────────────────────

def simulated_annealing(solution, time_limit=60):
    p = solution.problem
    local_nb = LocalNeighbourhood(p)

    s        = solution.copy_solution()
    best     = s.copy_solution()

    # T0 chosen so that 97% of proposed steps are accepted at the start.
    # For a worsening move of size delta: exp(-delta/T0) = 0.97
    # => T0 = -delta / ln(0.97)
    # We estimate delta as ~2% of the current objective value
    T0 = s.objective_value() * 0.02 / abs(math.log(0.97))
    T  = T0

    # Geometric cooling: T := alpha * T after each temperature level
    alpha = 0.995

    # Steps per temperature level: n*(n-1) as per slides
    # (proportional to neighbourhood size)
    steps_per_temp = max(100, p.n * (p.n - 1))

    print(f"  T0={T0:.2f}, steps/temp={steps_per_temp}")

    start      = time.time()
    iterations = 0
    accepted   = 0
    proposed   = 0

    while time.time() - start < time_limit:

        # Inner loop: keep temperature fixed for steps_per_temp steps
        for _ in range(steps_per_temp):
            # Proposal mechanism: uniform random choice from neighbourhood
            moves = list(local_nb.moves(s))
            if not moves:
                break
            move = random.choice(moves)

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
    print(f"  Iterations={iterations}, Accept={accepted/proposed:.2%}, Time={elapsed:.1f}s")
    return best



if __name__ == '__main__':
    instance_file, solution_file = sys.argv[1], sys.argv[2]

    p = Problem(instance_file)

    # Run 10 times, each from a fresh greedy solution, keep the best
    best_solution = None
    results       = []
    for i in range(10):
        print(f"\nRun {i+1}")
        s  = greedy_construction(p)
        print(f"  Greedy: {s.objective_value()}")
        s  = simulated_annealing(s, time_limit=60)
        ov = s.objective_value()
        results.append(ov)
        print(f"  SA:     {ov}")
        if best_solution is None or ov < best_solution.objective_value():
            best_solution = s

    results.sort()
    median = results[len(results) // 2]
    print(f"\nBest:   {best_solution.objective_value()}")
    print(f"Median: {median}")

    with open(solution_file, 'w') as f:
        f.write(' '.join(map(str, best_solution.team)) + '\n')
    print(f"Solution written to {solution_file}")
