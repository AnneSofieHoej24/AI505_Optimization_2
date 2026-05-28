import random
import sys

from roar_net_api.algorithms import greedy_construction


class Problem:
    """A team-formation instance whose construction neighbourhood adds
    randomness, so repeated greedy runs explore different solutions."""

    def __init__(self, path):
        """Parse an instance file and pre-sort students most-constrained first."""
        lines = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    lines.append(line)

        tokens = iter(lines)

        self.n, self.n_teams, self.n_attrs, n_disagree, self.team_min, self.team_max = (
            map(int, next(tokens).split())
        )

        self.weights = list(map(int, next(tokens).split()))

        self.labels = [list(map(int, next(tokens).split())) for _ in range(self.n)]

        self.disagreements = set()
        for _ in range(n_disagree):
            a, b = map(int, next(tokens).split())
            self.disagreements.add((min(a, b), max(a, b)))

        conflict_count = [0] * self.n
        for a, b in self.disagreements:
            conflict_count[a] += 1
            conflict_count[b] += 1
        self.order = sorted(range(self.n), key=lambda s: -conflict_count[s])

    def empty_solution(self):
        """Return a solution with every student unassigned and all teams empty."""
        return Solution(self, [-1] * self.n, [set() for _ in range(self.n_teams)])

    def construction_neighbourhood(self):
        """Return the randomized neighbourhood used to assign students."""
        return AddNeighbourhood(self)


class Solution:
    """An assignment of students to teams plus the per-team membership sets."""

    def __init__(self, problem, team, members):
        """Store the problem, the per-student team array and the per-team member sets."""
        self.problem = problem
        self.team = team
        self.members = members

    def __str__(self):
        """Human-readable dump of the team assignment and memberships."""
        members = "\n".join([f"\t\t{t}: {str(m)}" for t, m in enumerate(self.members)])
        return f"\tteam:    {self.team}\n\tmembers:\n{members}"

    def copy_solution(self):
        """Return a deep-enough copy that can be mutated independently."""
        return Solution(self.problem, self.team[:], [m.copy() for m in self.members])

    def objective_value(self):
        """Total weighted label diversity, or None if any student is unassigned."""
        if -1 in self.team:
            return None
        p = self.problem
        total = 0
        for t in range(p.n_teams):
            for a in range(p.n_attrs):
                total += p.weights[a] * len({p.labels[s][a] for s in self.members[t]})
        return total

    def lower_bound(self):
        """Trivial lower bound: at least one distinct label per attribute per team."""
        return sum(self.problem.weights) * self.problem.n_teams


class AddNeighbourhood:
    """Randomized construction neighbourhood: picks one of the most-constrained
    unassigned students and offers it shuffled feasible teams."""

    def __init__(self, problem):
        """Store the problem this neighbourhood operates on."""
        self.problem = problem

    def moves(self, solution):
        """Yield feasible AddMoves for a randomly chosen highly-constrained student."""
        p = self.problem

        unassigned = [s for s in p.order if solution.team[s] == -1]
        if not unassigned:
            return

        candidates = unassigned[:3]
        s = random.choice(candidates)

        n_remaining = len(unassigned)
        spots_needed = sum(
            max(0, p.team_min - len(solution.members[t])) for t in range(p.n_teams)
        )

        team_order = list(range(p.n_teams))
        random.shuffle(team_order)

        for t in team_order:
            if len(solution.members[t]) >= p.team_max:
                continue

            if any(
                (min(s, o), max(s, o)) in p.disagreements for o in solution.members[t]
            ):
                continue

            t_needs_more = len(solution.members[t]) < p.team_min
            if not t_needs_more and n_remaining - 1 < spots_needed:
                continue

            yield AddMove(s, t)


class AddMove:
    """A move that assigns one student to one team."""

    def __init__(self, s, t):
        """Record the student s and target team t."""
        self.s = s
        self.t = t

    def __str__(self):
        """Describe the assignment."""
        return f"assign student {self.s} to team {self.t}"

    def lower_bound_increment(self, solution):
        """Cost of the move: weights of attributes where s adds a new label to team t."""
        p = solution.problem
        return sum(
            p.weights[a]
            for a in range(p.n_attrs)
            if not any(
                p.labels[o][a] == p.labels[self.s][a] for o in solution.members[self.t]
            )
        )

    def apply_move(self, solution):
        """Apply the assignment in place and return the mutated solution."""
        solution.team[self.s] = self.t
        solution.members[self.t].add(self.s)
        return solution


def run_randomized_construction(instance_file, solution_file):
    """Run one randomized greedy construction, write the result and return its objective."""
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
