"""Task 4 - gradient descent and Newton's method on the log-barrier centering
problem with box constraints.

The problem is:
    a_i^T x <= 1   for i = 1..m
    -1 <= x_j <= 1 for j = 1..n

The barrier function is:
    f(x) = -sum_i log(1 - a_i^T x)
           - sum_j log(1 - x_j)
           - sum_j log(1 + x_j)

We start at x0 = 0, which is strictly feasible.
"""

import numpy as np
import matplotlib.pyplot as plt

from line_search import strong_backtracking


# ---------- barrier, gradient, Hessian -------------------------------------
def make_oracle(A):
    """Returns three functions: the barrier f, its gradient and its Hessian."""

    def slacks(x):
        # Slacks for the three groups of constraints.
        s_ineq = 1.0 - A @ x
        s_up = 1.0 - x
        s_lo = 1.0 + x
        return s_ineq, s_up, s_lo

    def f(x):
        s, su, sl = slacks(x)
        # If x is not strictly feasible, return infinity.
        if np.any(s <= 0):
            return np.inf
        if np.any(su <= 0):
            return np.inf
        if np.any(sl <= 0):
            return np.inf

        total = 0.0
        for i in range(len(s)):
            total = total - np.log(s[i])
        for j in range(len(su)):
            total = total - np.log(su[j])
        for j in range(len(sl)):
            total = total - np.log(sl[j])
        return total

    def grad(x):
        s, su, sl = slacks(x)
        part1 = A.T @ (1.0 / s)
        part2 = 1.0 / su
        part3 = 1.0 / sl
        return part1 + part2 - part3

    def hess(x):
        s, su, sl = slacks(x)

        # Hessian from the inequality constraints: A^T diag(1/s^2) A.
        scaling = 1.0 / (s ** 2)
        scaled_A = A.T * scaling
        H = scaled_A @ A

        # Add the diagonal contribution from the box constraints.
        diag_extra = 1.0 / (su ** 2) + 1.0 / (sl ** 2)
        H = H + np.diag(diag_extra)
        return H

    return f, grad, hess


# ---------- gradient descent -----------------------------------------------
def gradient_descent(A, x0, eta=1e-6, max_iter=2000):
    f, grad, _ = make_oracle(A)
    x = x0.copy()

    fs = [f(x)]
    steps = []
    gnorms = []

    for k in range(max_iter):
        g = grad(x)
        gn = np.linalg.norm(g)
        gnorms.append(gn)
        if gn <= eta:
            break
        d = -g
        a, _, _ = strong_backtracking(f, grad, x, d)
        x = x + a * d
        fn = f(x)
        fs.append(fn)
        steps.append(a)

    return x, np.array(fs), np.array(steps), np.array(gnorms)


# ---------- Newton ----------------------------------------------------------
def newton(A, x0, eta=1e-8, max_iter=200):
    f, grad, hess = make_oracle(A)
    x = x0.copy()

    fs = [f(x)]
    steps = []
    decr = []

    for k in range(max_iter):
        g = grad(x)
        H = hess(x)
        d = -np.linalg.solve(H, g)

        # Newton decrement squared.
        lam2 = -g @ d
        if lam2 < 0.0:
            lam_value = 0.0
        else:
            lam_value = np.sqrt(lam2)
        decr.append(lam_value)

        if 0.5 * lam2 <= eta:
            break

        a, _, _ = strong_backtracking(f, grad, x, d)
        x = x + a * d
        fn = f(x)
        fs.append(fn)
        steps.append(a)

    return x, np.array(fs), np.array(steps), np.array(decr)


# ---------- experiment ------------------------------------------------------
def run_instance(n, m, seed=0):
    rng = np.random.default_rng(seed)

    # Sample a_i ~ N(0, I). The origin is strictly feasible since a_i^T 0 = 0 < 1.
    A = rng.standard_normal((m, n))
    x0 = np.zeros(n)

    xg, fg, sg, gng = gradient_descent(A, x0, eta=1e-4, max_iter=5000)
    xn, fn, sn, dn = newton(A, x0, eta=1e-10, max_iter=100)

    gd_msg = "[n={0}, m={1}]  GD iters={2}, f*={3:.6f}, ||g||={4:.2e}".format(
        n, m, len(fg) - 1, fg[-1], gng[-1]
    )
    nt_msg = "[n={0}, m={1}]  NT iters={2}, f*={3:.6f}, decr={4:.2e}".format(
        n, m, len(fn) - 1, fn[-1], dn[-1]
    )
    print(gd_msg)
    print(nt_msg)

    return (fg, sg, gng), (fn, sn, dn)


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "lines.linewidth": 1.8,
})

GD_COLOR = "#2C7FB8"
NT_COLOR = "#D7263D"


def plot_run(gd, nt, n, m, fname):
    fg, sg, gng = gd
    fn, sn, dn = nt

    # Use the smallest value we found as our reference for the optimal value.
    f_star = min(fg[-1], fn[-1])
    gap_g = np.maximum(fg - f_star, 1e-16)
    gap_n = np.maximum(fn - f_star, 1e-16)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    gd_label = "GD ({0} it)".format(len(fg) - 1)
    nt_label = "Newton ({0} it)".format(len(fn) - 1)

    axes[0].semilogy(gap_g, label=gd_label, color=GD_COLOR)
    axes[0].semilogy(gap_n, label=nt_label, color=NT_COLOR)
    axes[0].set_xlabel("iteration $k$")
    axes[0].set_ylabel(r"$f(x_k) - f^\star$")
    axes[0].set_title("suboptimality")
    axes[0].legend(loc="upper right")

    axes[1].semilogy(gng, label="GD", color=GD_COLOR)
    axes[1].semilogy(dn, label="Newton decr. $\\lambda$", color=NT_COLOR)
    axes[1].set_xlabel("iteration $k$")
    axes[1].set_ylabel("optimality measure")
    axes[1].set_title(r"$\|\nabla f\|_2$ / Newton decrement")
    axes[1].legend(loc="upper right")

    axes[2].plot(sg, label="GD", color=GD_COLOR, marker="o", ms=2.5, lw=1.0)
    axes[2].plot(sn, label="Newton", color=NT_COLOR, marker="s", ms=3.5, lw=1.2)
    axes[2].set_xlabel("iteration $k$")
    axes[2].set_ylabel(r"step length $\alpha_k$")
    axes[2].set_title("backtracked step")
    axes[2].set_yscale("log")
    axes[2].legend(loc="lower right")

    title = "Convergence - $n={0}$, $m={1}$".format(n, m)
    fig.suptitle(title, y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_summary(results, fname):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    num_runs = len(results)
    blues = plt.get_cmap("Blues")(np.linspace(0.45, 0.9, num_runs))
    reds = plt.get_cmap("Reds")(np.linspace(0.45, 0.9, num_runs))

    labels = list(results.keys())
    for i in range(num_runs):
        label = labels[i]
        gd, nt = results[label]
        fg, _, gng = gd
        fn, _, dn = nt

        f_star = min(fg[-1], fn[-1])
        gap_g = np.maximum(fg - f_star, 1e-16)
        gap_n = np.maximum(fn - f_star, 1e-16)

        axes[0].semilogy(gap_g, color=blues[i], label="GD " + label)
        axes[0].semilogy(gap_n, color=reds[i], label="NT " + label)
        axes[1].semilogy(gng, color=blues[i], label="GD " + label)
        axes[1].semilogy(dn, color=reds[i], label="NT " + label)

    axes[0].set_xlabel("iteration $k$")
    axes[0].set_ylabel(r"$f(x_k)-f^\star$")
    axes[0].set_title("suboptimality across sizes")

    axes[1].set_xlabel("iteration $k$")
    axes[1].set_ylabel("optimality measure")
    axes[1].set_title(r"$\|\nabla f\|$ / Newton decrement")

    for ax in axes:
        ax.legend(fontsize=8, ncol=2, loc="upper right")

    plt.tight_layout()
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run():
    instances = [(2, 5), (10, 30), (50, 200), (100, 500)]
    results = {}
    for instance in instances:
        n = instance[0]
        m = instance[1]
        gd, nt = run_instance(n, m, seed=42)
        fname = "figures/task4_n{0}_m{1}.png".format(n, m)
        plot_run(gd, nt, n, m, fname)
        key = "n={0},m={1}".format(n, m)
        results[key] = (gd, nt)
    plot_summary(results, "figures/task4_summary.png")


if __name__ == "__main__":
    run()
