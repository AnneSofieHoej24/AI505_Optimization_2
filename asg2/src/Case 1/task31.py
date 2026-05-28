"""Task 3.1 - find the analytic center of the polytope A x <= b using Newton's
method, starting from the origin. Plot both the feasible region and the
Newton iterate path.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from common import A, b, solve_center, draw_feasible_region, style_axes


def run():
    """Run Newton from the origin to the analytic center, plot the feasible
    region and the Newton path, and return the center x*."""
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_feasible_region(ax, alpha_fill=0.75)

    ax.text(
        0, -0.5, "Feasible region",
        ha="center", va="center", fontsize=13, color="black",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )

    point = (0, 0)
    start_array = np.array(point, dtype=float)

    from common import barrier
    print("f(start) =", barrier(start_array, A, b))

    path_list = []
    path_list.append(np.array(point, dtype=float))

    def store_iterate(xk):
        """Callback that records each Newton iterate for plotting."""
        path_list.append(xk.copy())

    res = solve_center(A, b, x0=point, callback=store_iterate)
    x_star = res.x
    path = np.array(path_list)

    slack_at_star = b - A @ x_star
    min_slack = np.min(slack_at_star)

    print("x* =", x_star)
    print("f(x*) =", res.fun, "iters =", res.nit)
    print("min slack =", min_slack)

    style_axes(ax)
    plt.tight_layout()
    ax.scatter(point[0], point[1], color="k", zorder=5,
               label="Starting point = " + str(point))
    ax.legend(loc="upper left")
    plt.savefig("figures/simpelPlot.png", dpi=200, bbox_inches="tight")

    path_x = path[:, 0]
    path_y = path[:, 1]
    ax.plot(
        path_x, path_y,
        "o", color="#222", markersize=4, zorder=4,
        label="Newton path",
    )

    for i in range(len(path) - 1):
        p0 = path[i]
        p1 = path[i + 1]
        ax.annotate(
            "",
            xy=p1,
            xytext=p0,
            arrowprops=dict(arrowstyle="->", linestyle=":", color="#222"),
            zorder=4,
        )

    center_label = "Center $ \\approx$ ({0:.3f}, {1:.3f})".format(
        x_star[0], x_star[1]
    )
    ax.scatter(
        x_star[0], x_star[1],
        color="red", zorder=6,
        label=center_label,
    )
    ax.legend(loc="upper left")
    plt.savefig("figures/newton_path.png", dpi=200, bbox_inches="tight")

    return x_star


if __name__ == "__main__":
    run()
