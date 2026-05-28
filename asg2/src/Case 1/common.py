"""Shared setup for Case 1: constraints, feasible grid, barrier, and a
generic centering solver. Used by task31.py, task32.py and task33.py.
"""

import numpy as np
from scipy.optimize import minimize


A = np.array([
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1],
    [1, 1],
], dtype=float)
b = np.array([1, 1, 1, 1, 1.5], dtype=float)


x_values = np.linspace(-1.4, 1.4, 400)
y_values = np.linspace(-1.4, 1.4, 400)
xx, yy = np.meshgrid(x_values, y_values)


def compute_feasible_mask(A_input, b_input, grid_xx, grid_yy):
    """Return a boolean mask of which grid points satisfy A x <= b."""
    flat_x = grid_xx.ravel()
    flat_y = grid_yy.ravel()
    pts = np.stack([flat_x, flat_y], axis=1)

    constraint_values = pts @ A_input.T
    satisfied = constraint_values <= b_input
    feasible_flat = np.all(satisfied, axis=1)
    return feasible_flat.reshape(grid_xx.shape)


feasible = compute_feasible_mask(A, b, xx, yy)


def barrier(x, A_input, b_input):
    """Log-barrier f(x) = -sum_i log(b_i - a_i^T x); +inf outside the polytope."""
    s = b_input - A_input @ x
    if np.any(s <= 0):
        return np.inf
    total = 0.0
    for i in range(len(s)):
        total = total - np.log(s[i])
    return total


def barrier_grad(x, A_input, b_input):
    """Gradient of the log-barrier: A^T (1 / s) with s = b - A x."""
    s = b_input - A_input @ x
    inv_s = 1.0 / s
    return A_input.T @ inv_s


def barrier_hess(x, A_input, b_input):
    """Hessian of the log-barrier: A^T diag(1 / s^2) A with s = b - A x."""
    s = b_input - A_input @ x
    diag_values = 1.0 / (s ** 2)
    D = np.diag(diag_values)
    return A_input.T @ D @ A_input


def solve_center(A_input, b_input, x0=(0.0, 0.0), callback=None):
    """Find the analytic center of {x : A x <= b} using Newton's method."""

    def f_local(x):
        """Barrier value at x for the captured A, b."""
        return barrier(x, A_input, b_input)

    def g_local(x):
        """Barrier gradient at x for the captured A, b."""
        return barrier_grad(x, A_input, b_input)

    def h_local(x):
        """Barrier Hessian at x for the captured A, b."""
        return barrier_hess(x, A_input, b_input)

    result = minimize(
        f_local,
        x0,
        jac=g_local,
        hess=h_local,
        method="trust-exact",
        callback=callback,
    )
    return result


def draw_feasible_region(ax, alpha_fill=0.75):
    """Draw the feasible region fill and outline on the given axes."""
    ax.contourf(
        xx, yy, feasible,
        levels=[0.5, 1.5],
        colors=["#00D6FC"],
        alpha=alpha_fill,
    )
    ax.contour(
        xx, yy, feasible,
        levels=[0.5],
        colors=["#FC2600"],
        linewidths=2,
    )


def style_axes(ax):
    """Apply the common axis styling used across all plots."""
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5)
    ax.grid(True, alpha=0.3)
