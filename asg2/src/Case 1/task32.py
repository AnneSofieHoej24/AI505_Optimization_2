"""Task 3.2 - scale the 5th constraint by a factor gamma. Both the feasible
set and the analytic center are unchanged in theory, because:
    -log(gamma * s_i) = -log(gamma) - log(s_i),
which is a constant in x. So the gradient and Hessian of the barrier are
identical.

This script confirms it numerically for several values of gamma.
"""

import numpy as np
import matplotlib.pyplot as plt

from common import A, b, solve_center, draw_feasible_region, style_axes


def run(x_star):
    """Re-solve the center after scaling the 5th constraint by several gammas
    and plot that every scaled center coincides with the original x_star."""
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    draw_feasible_region(ax2, alpha_fill=0.5)

    x_old = x_star
    original_label = "original ({0:.3f}, {1:.3f})".format(x_old[0], x_old[1])
    ax2.scatter(
        x_old[0], x_old[1],
        color="red", s=80, zorder=6,
        label=original_label,
    )

    gammas = [0.1, 1, 10, 100]
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]

    for i in range(len(gammas)):
        g = gammas[i]
        c = colors[i]

        A_g = A.copy()
        b_g = b.copy()
        A_g[4] = A_g[4] * g
        b_g[4] = b_g[4] * g

        result = solve_center(A_g, b_g)
        x_new = result.x

        diff = x_old - x_new
        dist = np.linalg.norm(diff)
        print("gamma={0:>5}: x_new = {1}, ||x_old - x_new|| = {2:.3e}".format(
            g, x_new, dist
        ))

        new_label = "$\\gamma={0}$: ({1:.3f}, {2:.3f})".format(
            g, x_new[0], x_new[1]
        )
        ax2.scatter(
            x_new[0], x_new[1],
            color=c, marker="x", s=80, zorder=5,
            label=new_label,
        )

    style_axes(ax2)
    ax2.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/task32_centers.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    from task31 import run as run_task31
    x_star = run_task31()
    run(x_star)
    plt.show()
