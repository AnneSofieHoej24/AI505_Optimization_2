"""Task 3.3 - normalize each constraint by dividing by b_i so that every
constraint becomes a_i^T x <= 1. This is the same kind of per-row scaling as
in Task 3.2, so the analytic center should be unchanged.
"""

import numpy as np
import matplotlib.pyplot as plt

from common import A, b, solve_center, draw_feasible_region, style_axes


def run(x_star):
    """Normalize every constraint to a_i^T x <= 1 (dividing row i by b_i),
    re-solve, and plot that the center matches the original x_star."""
    A_norm = A / b[:, None]
    b_norm = np.ones_like(b)

    result = solve_center(A_norm, b_norm)
    x_norm = result.x

    diff = x_star - x_norm
    dist = np.linalg.norm(diff)

    print("x_original =", x_star)
    print("x_normalized =", x_norm)
    print("||x_original - x_normalized|| = {0:.3e}".format(dist))

    fig3, ax3 = plt.subplots(figsize=(6, 6))
    draw_feasible_region(ax3, alpha_fill=0.5)

    orig_label = "original  ({0:.4f}, {1:.4f})".format(x_star[0], x_star[1])
    norm_label = "normalized ({0:.4f}, {1:.4f})".format(x_norm[0], x_norm[1])

    ax3.scatter(
        x_star[0], x_star[1],
        color="red", s=100, zorder=6,
        label=orig_label,
    )
    ax3.scatter(
        x_norm[0], x_norm[1],
        color="black", marker="x", s=100, zorder=7,
        label=norm_label,
    )

    style_axes(ax3)
    ax3.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/task33_normalized.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    from task31 import run as run_task31
    x_star = run_task31()
    run(x_star)
    plt.show()
