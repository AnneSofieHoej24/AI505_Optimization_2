# SPDX-FileCopyrightText: 2025 Andreia P. Guerreiro <andreia.guerreiro@tecnico.ulisboa.pt>
#
# SPDX-License-Identifier: Apache-2.0

import sys
import numpy as np


def next_uncommented_line(f, start_comment="#"):
    line = f.readline()
    while line.strip().startswith(start_comment):
        line = f.readline()
    return line


def read_instance(fname):
    with open(fname, "r") as f:
        line = next_uncommented_line(f)
        n, T, m, d, l, u = list(map(int, line.split()))
        line = next_uncommented_line(f)
        W = np.array(list(map(int, line.split())), dtype=int)
        GL = []
        D = []
        for g in range(n):
            line = next_uncommented_line(f)
            GL.append(list(map(int, line.split())))
        for di in range(d):
            line = next_uncommented_line(f)
            pair = list(map(int, line.split()))
            D.append([pair[0], pair[1]])
    return GL, T, m, W, D, l, u


def read_solution(fname, T):
    x = np.loadtxt(fname, dtype=int)    
    #    x = x - 1
    return x


def check(instance, sol):
    GL, T, m, W, D, l, u = instance
    n = len(GL)
    _, count = np.unique(sol, return_counts=True)
    assert len(sol) == n, f"There has to be {n} students"
    assert (sol >= 0).all() and (sol < T).all(), f"Invalid team index"
    assert (count >= l).all(), f"All teams must have at least {l} students"
    assert (count <= u).all(), f"All teams must have at most {u} students"

    for pair in D:
        assert sol[pair[0]] != sol[pair[1]], f"Students {pair[0]} and {pair[1]} cannot be in the same team"

    num_attrs = len(GL[0])
    labels_per_team = [[set() for _ in range(num_attrs)] for _ in range(T)]
    for g in range(n):
        t = sol[g]
        labels = GL[g]
        for a, label in enumerate(labels):
            labels_per_team[t][a].add(label)

    z = np.zeros((T, num_attrs), dtype=int)
    for t in range(T):
        for a in range(num_attrs):
            z[t, a] = len(labels_per_team[t][a])
    print(z)
    print(W)
    return (z * W[np.newaxis, :]).sum()


if __name__ == "__main__":
    instanceFile = sys.argv[1]
    solutionFile = sys.argv[2]

    instance = read_instance(instanceFile)
    solution = read_solution(solutionFile, instance[1])
    obj = check(instance, solution)
    print(obj)
