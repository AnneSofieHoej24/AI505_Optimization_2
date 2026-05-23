import argparse
import contextlib
import io
import random

import numpy as np

from checker import check, read_instance, read_solution
from Construction import run_construction as runCons
from ConstructionR import run_randomized_construction as runRCons

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

    if con not in (0, 1):
        print("first integer must be 0 (greedy) or 1 (randomized)!")
        exit()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    construction_map = {0: runCons, 1: runRCons}
    selected_function = construction_map[con]

    inst = read_instance(path)

    results = []
    for _ in range(n):
        obj = selected_function(path, "solution.txt")
        sol = read_solution("solution.txt", inst[1])
        with contextlib.redirect_stdout(io.StringIO()):
            checked = check(inst, sol)
        assert checked == obj, f"checker={checked} disagrees with construction={obj}"
        results.append(obj)

    arr = np.array(results)
    print(
        f"best={arr.max()} median={np.median(arr)} mean={arr.mean():.1f} "
        f"min={arr.min()} std={arr.std():.1f}"
    )
