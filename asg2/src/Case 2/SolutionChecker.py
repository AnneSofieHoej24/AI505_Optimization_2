import argparse
import contextlib
import io
import os
import random

import numpy as np

from checker import check, read_instance, read_solution
from Construction import run_construction as runCons
from ConstructionR import run_randomized_construction as runRCons
from Local_search import runLocalSearch as runLS
from Meta import RunMeta as runMeta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vals",
        nargs=2,
        default=(0, 10),
        type=int,
        help="Two integers: first = construction (0=greedy, 1=randomized), second = iterations.",
    )

    parser.add_argument("--path", type=str, required=True, help="path to instance .txt file")
    parser.add_argument("--seed", type=int, default=None, help="seed for randomized construction")

    args = parser.parse_args()

    con = args.vals[0]
    n = args.vals[1]
    path = args.path

    if con not in (0, 1, 2, 3):
        print("first integer must be 0 (greedy), 1 (randomized), 2 (local search), or 3 (meta)!")
        exit()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    construction_map = {0: runCons, 1: runRCons, 2: runLS, 3: runMeta}
    selected_function = construction_map[con]

    inst = read_instance(path)

    sol_file = f"solution_{os.getpid()}.txt"
    results = []
    for i in range(n):
        obj = selected_function(path, sol_file)
        sol = read_solution(sol_file, inst[1])
        with contextlib.redirect_stdout(io.StringIO()):
            checked = check(inst, sol)
        assert checked == obj, f"checker={checked} disagrees with construction={obj}"
        print(f"Run {i+1}: obj={obj}")
        results.append(obj)

    arr = np.array(results)
    # Objective is MINIMIZED — best = min.
    print(
        f"best={arr.min()} median={np.median(arr)} mean={arr.mean():.1f} "
        f"worst={arr.max()} std={arr.std():.1f}"
    )
