import math
import sys


class Problem:
    """A team-formation instance: students, attributes, weights, team-size
    bounds and the set of disagreeing student pairs."""

    def __init__(self, path):
        """Parse an instance file into dimensions, weights, labels and disagreements."""
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
        """Return the neighbourhood used to assign students one at a time."""
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
    """Construction neighbourhood that assigns the next student to a feasible team."""

    def __init__(self, problem):
        """Store the problem this neighbourhood operates on."""
        self.problem = problem

    def moves(self, solution):
        """Yield one AddMove per feasible team for the next unassigned student."""
        p = self.problem

        s = next((s for s in p.order if solution.team[s] == -1), None)
        if s is None:
            return

        for t in range(p.n_teams):
            if len(solution.members[t]) >= p.team_max:
                continue
            if any(
                (min(s, o), max(s, o)) in p.disagreements for o in solution.members[t]
            ):
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


def repair_min_size(s):
    """Move students from over-min teams into under-min teams until every team
    has at least team_min members, skipping moves that create a disagreement."""
    p = s.problem
    changed = True
    while changed:
        changed = False
        under = [t for t in range(p.n_teams) if len(s.members[t]) < p.team_min]
        if not under:
            return
        for u in under:
            moved = False
            for o in range(p.n_teams):
                if len(s.members[o]) <= p.team_min:
                    continue
                for stu in list(s.members[o]):
                    if any((min(stu, m), max(stu, m)) in p.disagreements for m in s.members[u]):
                        continue
                    s.members[o].discard(stu)
                    s.members[u].add(stu)
                    s.team[stu] = u
                    changed = True
                    moved = True
                    break
                if moved:
                    break
            if not moved:
                return


def run_construction(instance_file, solution_file):
    """Build a greedy solution (assign each student to the least-cost feasible
    team), repair team sizes, write it to solution_file and return its objective."""
    p = Problem(instance_file)
    s = p.empty_solution()

    constr = p.construction_neighbourhood()
    while True:
        best_move, best_incr = None, math.inf
        for move in constr.moves(s):
            incr = move.lower_bound_increment(s)
            if incr < best_incr:
                best_move, best_incr = move, incr
        if best_move is None:
            break
        best_move.apply_move(s)

    repair_min_size(s)

    print(f"Objective value: {s.objective_value()}")

    with open(solution_file, "w") as f:
        f.write(" ".join(map(str, s.team)) + "\n")

    return s.objective_value()


if __name__ == "__main__":
    result = run_construction(sys.argv[1], sys.argv[2])
