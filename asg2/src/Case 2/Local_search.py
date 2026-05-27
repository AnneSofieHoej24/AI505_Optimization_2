import sys
import math
import random

from ConstructionR import Problem, Solution, greedy_construction

from roar_net_api.algorithms import best_improvement as roar_best
from roar_net_api.algorithms import first_improvement as roar_first


class LocalNeighbourhood:

    def __init__(self, problem):
        self.problem = problem

    def moves(self, solution):
        p = self.problem
        # A move swaps student s (in team t_s) with student r (in team t_r).
        # Swapping keeps team sizes balanced so team_min/team_max stay satisfied.
        for s in range(p.n):
            t_s = solution.team[s]
            for r in range(s + 1, p.n):
                t_r = solution.team[r]
                if t_s == t_r:
                    continue  # same team, no point swapping

                # s joins t_r (r is leaving), r joins t_s (s is leaving)
                if not can_join(s, t_r, r, solution):
                    continue
                if not can_join(r, t_s, s, solution):
                    continue

                yield SwapMove(s, t_s, r, t_r)


def can_join(s, t, excluding, solution):
    # Check student s has no disagreement with anyone in team t,
    # ignoring 'excluding' who is simultaneously leaving the team
    for other in solution.members[t]:
        if other == excluding:
            continue
        if (min(s, other), max(s, other)) in solution.problem.disagreements:
            return False
    return True

class SwapMove:

    def __init__(self, s, t_s, r, t_r):
        self.s,   self.t_s = s,  t_s   # student s and their current team
        self.r,   self.t_r = r,  t_r   # student r and their current team
        self.ov_incr = None

    def __str__(self):
        return f"swap student {self.s} (team {self.t_s}) with student {self.r} (team {self.t_r})"

    def objective_value_increment(self, solution):
        # Only compute once and cache the result
        if self.ov_incr is None:
            p = solution.problem
            ms = solution.members[self.t_s]
            mr = solution.members[self.t_r]

            # Simulate the swap to compute new label diversity
            new_ms = (ms - {self.s}) | {self.r}
            new_mr = (mr - {self.r}) | {self.s}

            old, new = 0, 0
            for a in range(p.n_attrs):
                w = p.weights[a]
                old += w * len({p.labels[m][a] for m in ms})
                old += w * len({p.labels[m][a] for m in mr})
                new += w * len({p.labels[m][a] for m in new_ms})
                new += w * len({p.labels[m][a] for m in new_mr})

            self.ov_incr = new - old
        return self.ov_incr

    def apply_move(self, solution):
        solution.team[self.s] = self.t_r
        solution.team[self.r] = self.t_s
        solution.members[self.t_s].discard(self.s)
        solution.members[self.t_s].add(self.r)
        solution.members[self.t_r].discard(self.r)
        solution.members[self.t_r].add(self.s)
        return solution



def best_improvement(solution):
    p = solution.problem
    local_nb = LocalNeighbourhood(p)
    s = solution.copy_solution()

    while True:
        best_move, best_incr = None, math.inf
        for move in local_nb.moves(s):
            incr = move.objective_value_increment(s)
            if incr < best_incr:
                best_move, best_incr = move, incr
        # Stop when no improving move exists
        if best_move is None or best_incr >= 0:
            break
        best_move.apply_move(s)
    return s

# Hand-written 
#s = best_improvement(s)

# ROAR-NET best improvement
#s = roar_best(p, s)

# ROAR-NET first improvement
#s = roar_first(p, s)

def runLocalSearch(instance_file, solution_file):
    p = Problem(instance_file)

    s = greedy_construction(p)
    s = best_improvement(s)

    with open(solution_file, 'w') as f:
        f.write(' '.join(map(str, s.team)) + '\n')

    return s.objective_value()

if __name__ == '__main__':
    instance_file, solution_file = sys.argv[1], sys.argv[2]
    runLocalSearch(instance_file, solution_file)

