import sys
import math

from ConstructionR import Problem
from roar_net_api.algorithms import greedy_construction


class LocalNeighbourhood:
    """Local-search neighbourhood whose moves swap two students between teams,
    preserving team sizes so the size constraints stay satisfied."""

    def __init__(self, problem):
        """Store the problem this neighbourhood operates on."""
        self.problem = problem

    def moves(self, solution):
        """Yield every feasible size-preserving swap between students on different teams."""
        p = self.problem
        for s in range(p.n):
            t_s = solution.team[s]
            for r in range(s + 1, p.n):
                t_r = solution.team[r]
                if t_s == t_r:
                    continue

                if not can_join(s, t_r, r, solution):
                    continue
                if not can_join(r, t_s, s, solution):
                    continue

                yield SwapMove(s, t_s, r, t_r)


def can_join(s, t, excluding, solution):
    """Return True if student s can join team t without a disagreement, ignoring
    the member 'excluding' who is leaving t in the same swap."""
    for other in solution.members[t]:
        if other == excluding:
            continue
        if (min(s, other), max(s, other)) in solution.problem.disagreements:
            return False
    return True


class SwapMove:
    """A move that swaps student s (team t_s) with student r (team t_r)."""

    def __init__(self, s, t_s, r, t_r):
        """Record the two students and their current teams; cache starts empty."""
        self.s,   self.t_s = s,  t_s
        self.r,   self.t_r = r,  t_r
        self.ov_incr = None

    def __str__(self):
        """Describe the swap."""
        return f"swap student {self.s} (team {self.t_s}) with student {self.r} (team {self.t_r})"

    def objective_value_increment(self, solution):
        """Change in objective from applying the swap (cached after first call)."""
        if self.ov_incr is None:
            p = solution.problem
            ms = solution.members[self.t_s]
            mr = solution.members[self.t_r]

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
        """Apply the swap in place and return the mutated solution."""
        solution.team[self.s] = self.t_r
        solution.team[self.r] = self.t_s
        solution.members[self.t_s].discard(self.s)
        solution.members[self.t_s].add(self.r)
        solution.members[self.t_r].discard(self.r)
        solution.members[self.t_r].add(self.s)
        return solution


def best_improvement(solution):
    """Repeatedly apply the swap with the most negative objective increment until
    no improving swap remains, then return the resulting local optimum."""
    p = solution.problem
    local_nb = LocalNeighbourhood(p)
    s = solution.copy_solution()

    while True:
        best_move, best_incr = None, math.inf
        for move in local_nb.moves(s):
            incr = move.objective_value_increment(s)
            if incr < best_incr:
                best_move, best_incr = move, incr
        if best_move is None or best_incr >= 0:
            break
        best_move.apply_move(s)
    return s


def runLocalSearch(instance_file, solution_file):
    """Construct a start solution, run best-improvement local search, write the
    result and return its objective."""
    p = Problem(instance_file)

    s = greedy_construction(p)
    s = best_improvement(s)

    with open(solution_file, 'w') as f:
        f.write(' '.join(map(str, s.team)) + '\n')

    return s.objective_value()


if __name__ == '__main__':
    instance_file, solution_file = sys.argv[1], sys.argv[2]
    runLocalSearch(instance_file, solution_file)
