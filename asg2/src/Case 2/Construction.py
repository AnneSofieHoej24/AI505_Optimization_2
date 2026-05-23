import math
import random
import sys
from collections.abc import Iterable

from roar_net_api.algorithms import greedy_construction


class Problem:

    def __init__(self, path):
        # Read the instance file, skipping blank lines and comments (#)
        lines = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    lines.append(line)

        tokens = iter(lines)

        # First line: problem dimensions
        self.n, self.n_teams, self.n_attrs, n_disagree, self.team_min, self.team_max = (
            map(int, next(tokens).split())
        )

        # Second line: importance weight for each attribute
        self.weights = list(map(int, next(tokens).split()))

        # Next n lines: attribute labels for each student
        # labels[s][a] = label of student s for attribute a
        self.labels = [list(map(int, next(tokens).split())) for _ in range(self.n)]

        # Remaining lines: pairs of students who cannot share a team
        # Stored as (min, max) tuples so we can look them up consistently
        self.disagreements = set()
        for _ in range(n_disagree):
            a, b = map(int, next(tokens).split())
            self.disagreements.add((min(a, b), max(a, b)))

        # Pre-sort students by number of disagreements, most constrained first.
        # This helps the greedy avoid dead ends where a student has no valid team.
        conflict_count = [0] * self.n
        for a, b in self.disagreements:
            conflict_count[a] += 1
            conflict_count[b] += 1
        self.order = sorted(range(self.n), key=lambda s: -conflict_count[s])

    def empty_solution(self):
        # All students unassigned (-1), all teams empty
        return Solution(self, [-1] * self.n, [set() for _ in range(self.n_teams)])

    def construction_neighbourhood(self):
        return AddNeighbourhood(self)


class Solution:

    def __init__(self, problem, team, members):
        self.problem = problem
        self.team = team  # team[s]    = team index of student s (-1 if unassigned)
        self.members = members  # members[t] = set of students in team t

    def __str__(self):
        members = "\n".join([f"\t\t{t}: {str(m)}" for t, m in enumerate(self.members)])
        return f"\tteam:    {self.team}\n\tmembers:\n{members}"

    def copy_solution(self):
        return Solution(self.problem, self.team[:], [m.copy() for m in self.members])

    def objective_value(self):
        # Returns None if any student is still unassigned
        if -1 in self.team:
            return None
        p = self.problem
        total = 0
        for t in range(p.n_teams):
            for a in range(p.n_attrs):
                # Count how many distinct labels appear in team t for attribute a
                total += p.weights[a] * len({p.labels[s][a] for s in self.members[t]})
        return total

    def lower_bound(self):
        # Trivial lower bound: every team has at least 1 distinct label per attribute
        return sum(self.problem.weights) * self.problem.n_teams


class AddNeighbourhood:

    def __init__(self, problem):
        self.problem = problem

    def moves(self, solution):
        p = self.problem

        # Pick the next unassigned student in conflict-priority order
        s = next((s for s in p.order if solution.team[s] == -1), None)
        if s is None:
            return  # all students assigned, no moves possible

        # Yield one move per feasible team for student s
        for t in range(p.n_teams):
            # Skip full teams
            if len(solution.members[t]) >= p.team_max:
                continue
            # Skip teams where s has a disagreement with an existing member
            if any(
                (min(s, o), max(s, o)) in p.disagreements for o in solution.members[t]
            ):
                continue
            yield AddMove(s, t)


class AddMove:

    def __init__(self, s, t):
        self.s = s  # student to assign
        self.t = t  # team to assign them to

    def __str__(self):
        return f"assign student {self.s} to team {self.t}"

    def diversity_gain(self, solution):
        # Gain = sum of weights for attributes where student s introduces
        # a label not yet present in team t (i.e. objective increase)
        p = solution.problem
        return sum(
            p.weights[a]
            for a in range(p.n_attrs)
            if not any(
                p.labels[o][a] == p.labels[self.s][a] for o in solution.members[self.t]
            )
        )

    def apply_move(self, solution):
        # Mutates the solution in place
        solution.team[self.s] = self.t
        solution.members[self.t].add(self.s)
        # return solution


# Greedy self
def run_construction(instance_file, solution_file):
    p = Problem(instance_file)
    s = p.empty_solution()

    # Greedy best-improvement construction
    constr = p.construction_neighbourhood()
    while True:
        best_move, best_gain = None, -math.inf
        for move in constr.moves(s):
            gain = move.diversity_gain(s)
            if gain > best_gain:
                best_move, best_gain = move, gain
        if best_move is None:
            break
        best_move.apply_move(s)

    print(f"Objective value: {s.objective_value()}")

    with open(solution_file, "w") as f:
        f.write(" ".join(map(str, s.team)) + "\n")

    return s.objective_value()


if __name__ == "__main__":
    result = run_construction(sys.argv[1], sys.argv[2])


# ROAR API

# if __name__ == '__main__':
#     instance_file, solution_file = sys.argv[1], sys.argv[2]

#     p = Problem(instance_file)

#     # Use the ROAR-NET greedy construction algorithm
#     s = greedy_construction(p)

#     print(f"Objective value: {s.objective_value()}")

#     with open(solution_file, 'w') as f:
#         f.write(' '.join(map(str, s.team)) + '\n')
