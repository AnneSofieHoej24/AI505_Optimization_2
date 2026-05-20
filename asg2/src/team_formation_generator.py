# SPDX-FileCopyrightText: 2025 Andreia P. Guerreiro <andreia.guerreiro@tecnico.ulisboa.pt>
# SPDX-FileCopyrightText: 2026 Marco Chiarandini <marco@imada.sdu.dk>
#
# SPDX-License-Identifier: Apache-2.0
# Modified by Marco Chiarandini in 2026 to match problem specifics for team formation.

import random
import sys, argparse
from time import time
import numpy as np
import math 

def random_labels(n, q, a):
    nlabels = np.random.choice([2, 4, 5, 8], size=a)    
    tmember_labels = []
    for g in range(n):
        labels = []
        for b in range(a):
            label = np.random.choice(np.arange(nlabels[b]), 1, replace=True)
            labels.extend(label)
        tmember_labels.append(labels)

    return tmember_labels

def random_attribute_weight(a, maxw):
    #return np.random.randint(1, maxw+1, a)
    return np.arange(10*a, 0, -10)

def random_disagreements(n, d):
    tmembers_ix = np.arange( n)
    D = np.array([np.random.choice(tmembers_ix, 2, replace=False) for di in range(d)])
    D = np.unique(D, axis=0)
    print(f"Generated {len(D)} disagreements")
    return list(D)


# def save_pb(fname, G, tmember_labels, T, L, D):
    # pass

def save_raw(fname, n, tmember_labels, q, L, W, D, l, u):
    f = open(fname, "w")
    f.write(f"# {n} tmembers, {q} teams, {L} attributes, {len(D)} disagreements, {l} to {u} tmembers per team\n")
    f.write(f"{n} {q} {L} {len(D)} {l} {u}\n")
    f.write(f"# Attribute weight\n")
    f.write(f"{' '.join(map(str, W))}\n")
    f.write("# Labels associated to each tmember\n")
    for gl in tmember_labels:
        f.write(f"{' '.join(map(str, gl))}\n")
    f.write("# Disagreements\n")
    for di in D:
        f.write(f"{' '.join(map(str, di))}\n")

    f.close()

#TODO
def save_pbo(fname, n, tmember_labels, q, L, W, D, l, u):
    pass


def readArguments():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
        description="Generate instances for the Team Formation Problem\n"
        )

    parser.add_argument("-o", type=str, default="./", help="output folder")
    parser.add_argument("-n", type=int, default=10, help="number of tmembers")
    parser.add_argument("-q", type=int, default=np.nan, help="number of teams")
    #parser.add_argument("-l", type=int, default=2, help="minimum number of tmembers per team")
    parser.add_argument("-u", type=int, default=np.nan, help="maximum number of tmembers per team")
    parser.add_argument("-a", type=int, default=10, help="number of attributes")
    parser.add_argument("-d", type=int, default=0, help="maximum number of disagreements")
    parser.add_argument("-W", type=int, default=10, help="difference in weight between attributes in [1,10]")
    parser.add_argument("-s", type=int, default=int(time()), help="random seed for reproducibility")
    args = parser.parse_args(sys.argv[1:])
    print ("args: %r\n" % args)

    if args.q is np.nan and args.u is np.nan:
        raise ValueError("Either q or u must be specified")
    if args.q is np.nan:
        args.q = math.ceil(args.n/args.u)
        args.l = math.floor(args.n/args.q)
    if args.u is np.nan:
        args.u = math.ceil(args.n/args.q)
        args.l = math.floor(args.n/args.q)

    random.seed(args.s)

    if args.n/args.u > args.q:
        raise ValueError("There are not enough teams")
    #if args.n/args.q < args.l:
    #    raise ValueError("There are too many teams")
    if args.l > args.u:
        raise ValueError("l cannot be greater than u")
    if args.W < 1 or args.W > 10:
        raise ValueError("W must be a number between 1 and 10")

    return args.o, args.n, args.q, args.l, args.u, args.a, args.d, args.W



if __name__ == "__main__":
    # n = 10
    # q = 3
    # u = 2,5
    # L = 10
    # d = 2
    # maxw = 10

    folder, n, q, l, u, a, d, W = readArguments()

    
    tmember_labels = random_labels(n, q, a)
    D = random_disagreements(n, d)
    W = random_attribute_weight(a, W)
    
    fname = folder+f"/tfp_{n}n_{q}q_{l}l_{u}u_{a}a_{len(D)}d.txt"
    save_raw(fname, n, tmember_labels, q, a, W, D, l, u)
