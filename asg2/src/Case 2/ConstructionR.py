import math
import random
import sys

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

        # Sort students by number of disagreements, most constrained first
        conflict_count = [0] * self.n
        for a, b in self.disagreements:
            conflict_count[a] += 1
            conflict_count[b] += 1
        self.order = sorted(range(self.n), key=lambda s: -conflict_count[s])

    def empty_solution(self):
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
        if -1 in self.team:
            return None
        p = self.problem
        total = 0
        for t in range(p.n_teams):
            for a in range(p.n_attrs):
                total += p.weights[a] * len({p.labels[s][a] for s in self.members[t]})
        return total

    def lower_bound(self):
        return sum(self.problem.weights) * self.problem.n_teams


class AddNeighbourhood:

    def __init__(self, problem):
        self.problem = problem

    def moves(self, solution):
        p = self.problem

        # Get all unassigned students, still in conflict-priority order
        unassigned = [s for s in p.order if solution.team[s] == -1]
        if not unassigned:
            return

        # Pick randomly from the top-3 most constrained unassigned students.
        # This keeps hard-to-place students near the front while adding variation.
        candidates = unassigned[:3]
        s = random.choice(candidates)

        # Count remaining students and spots still needed to satisfy team_min
        n_remaining = len(unassigned)
        spots_needed = sum(
            max(0, p.team_min - len(solution.members[t])) for t in range(p.n_teams)
        )

        # Shuffle team order for additional variation
        team_order = list(range(p.n_teams))
        random.shuffle(team_order)

        for t in team_order:
            # Skip full teams (team_max constraint)
            if len(solution.members[t]) >= p.team_max:
                continue

            # Skip teams where s conflicts with an existing member
            if any(
                (min(s, o), max(s, o)) in p.disagreements for o in solution.members[t]
            ):
                continue

            # Enforce team_min: don't add to a team that already has enough
            # members if doing so would leave too few students for other teams
            t_needs_more = len(solution.members[t]) < p.team_min
            if not t_needs_more and n_remaining - 1 < spots_needed:
                continue

            yield AddMove(s, t)


class AddMove:

    def __init__(self, s, t):
        self.s = s  # student to assign
        self.t = t  # team to assign them to

    def __str__(self):
        return f"assign student {self.s} to team {self.t}"

    def lower_bound_increment(self, solution):
        # ROAR minimizes this. Objective is to MAXIMIZE diversity weight,
        # so negate: more new-label weight = more negative = preferred.
        p = solution.problem
        return -sum(
            p.weights[a]
            for a in range(p.n_attrs)
            if not any(
                p.labels[o][a] == p.labels[self.s][a] for o in solution.members[self.t]
            )
        )

    def apply_move(self, solution):
        # Assign student to team, mutate in place, and return solution
        # (ROAR-NET greedy_construction expects apply_move to return the solution)
        solution.team[self.s] = self.t
        solution.members[self.t].add(self.s)
        return solution


def run_randomized_construction(instance_file, solution_file):
    # Single randomized run. Outer driver (SolutionChecker) handles repetition + stats.
    p = Problem(instance_file)
    s = greedy_construction(p)
    ov = s.objective_value()

    with open(solution_file, "w") as f:
        f.write(" ".join(map(str, s.team)) + "\n")

    return ov


if __name__ == "__main__":
    instance_file = sys.argv[1]
    solution_file = sys.argv[2]
    obj = run_randomized_construction(instance_file, solution_file)
    print(f"Objective value: {obj}")
