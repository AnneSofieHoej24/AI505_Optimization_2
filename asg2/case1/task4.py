"""Task 4 — gradient + Newton on the log-barrier center problem with box constraints.

Problem:  a_i^T x <= 1 (i=1..m),  |x_j| <= 1 (j=1..n).
Barrier:  f(x) = -sum_i log(1 - a_i^T x) - sum_j log(1 - x_j) - sum_j log(1 + x_j).
Start at x0 = 0 (strictly feasible).
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------- barrier, gradient, Hessian -------------------------------------
def make_oracle(A):
    m, n = A.shape

    def slacks(x):
        return 1.0 - A @ x, 1.0 - x, 1.0 + x  # s_ineq, s_up, s_lo

    def f(x):
        s, su, sl = slacks(x)
        if np.any(s <= 0) or np.any(su <= 0) or np.any(sl <= 0):
            return np.inf
        return -np.log(s).sum() - np.log(su).sum() - np.log(sl).sum()

    def grad(x):
        s, su, sl = slacks(x)
        return A.T @ (1.0 / s) + 1.0 / su - 1.0 / sl

    def hess(x):
        s, su, sl = slacks(x)
        H = (A.T * (1.0 / s**2)) @ A
        H += np.diag(1.0 / su**2 + 1.0 / sl**2)
        return H

    return f, grad, hess


# ---------- backtracking line search ---------------------------------------
def backtrack(f, x, d, g, alpha0=1.0, c=1e-4, rho=0.5, max_bt=60):
    """Armijo backtracking. Also shrinks until f(x+a d) finite (feasibility)."""
    fx = f(x)
    gTd = g @ d
    a = alpha0
    for _ in range(max_bt):
        fn = f(x + a * d)
        if np.isfinite(fn) and fn <= fx + c * a * gTd:
            return a, fn
        a *= rho
    return a, f(x + a * d)


# ---------- gradient descent -----------------------------------------------
def gradient_descent(A, x0, eta=1e-6, max_iter=2000):
    f, grad, _ = make_oracle(A)
    x = x0.copy()
    fs, steps, gnorms = [f(x)], [], []
    for k in range(max_iter):
        g = grad(x)
        gn = np.linalg.norm(g)
        gnorms.append(gn)
        if gn <= eta:
            break
        d = -g
        a, fn = backtrack(f, x, d, g)
        x = x + a * d
        fs.append(fn)
        steps.append(a)
    return x, np.array(fs), np.array(steps), np.array(gnorms)


# ---------- Newton ----------------------------------------------------------
def newton(A, x0, eta=1e-8, max_iter=200):
    f, grad, hess = make_oracle(A)
    x = x0.copy()
    fs, steps, decr = [f(x)], [], []
    for k in range(max_iter):
        g = grad(x)
        H = hess(x)
        d = -np.linalg.solve(H, g)
        lam2 = -g @ d  # Newton decrement squared
        decr.append(np.sqrt(max(lam2, 0.0)))
        if 0.5 * lam2 <= eta:
            break
        a, fn = backtrack(f, x, d, g)
        x = x + a * d
        fs.append(fn)
        steps.append(a)
    return x, np.array(fs), np.array(steps), np.array(decr)


# ---------- experiment ------------------------------------------------------
def run_instance(n, m, seed=0):
    rng = np.random.default_rng(seed)
    # a_i ~ N(0, I); leaves origin strictly feasible since a_i^T 0 = 0 < 1.
    A = rng.standard_normal((m, n))
    x0 = np.zeros(n)

    xg, fg, sg, gng = gradient_descent(A, x0, eta=1e-4, max_iter=5000)
    xn, fn, sn, dn = newton(A, x0, eta=1e-10, max_iter=100)

    print(f"[n={n}, m={m}]  GD iters={len(fg)-1}, f*={fg[-1]:.6f}, "
          f"||g||={gng[-1]:.2e}")
    print(f"[n={n}, m={m}]  NT iters={len(fn)-1}, f*={fn[-1]:.6f}, "
          f"decr={dn[-1]:.2e}")
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

    f_star = min(fg[-1], fn[-1])
    gap_g = np.maximum(fg - f_star, 1e-16)
    gap_n = np.maximum(fn - f_star, 1e-16)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    axes[0].semilogy(gap_g, label=f"GD ({len(fg)-1} it)", color=GD_COLOR)
    axes[0].semilogy(gap_n, label=f"Newton ({len(fn)-1} it)", color=NT_COLOR)
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

    fig.suptitle(f"Convergence — $n={n}$, $m={m}$", y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_summary(results, fname):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap_gd = plt.get_cmap("Blues")(np.linspace(0.45, 0.9, len(results)))
    cmap_nt = plt.get_cmap("Reds")(np.linspace(0.45, 0.9, len(results)))
    for i, (label, (gd, nt)) in enumerate(results.items()):
        fg, _, gng = gd
        fn, _, dn = nt
        f_star = min(fg[-1], fn[-1])
        axes[0].semilogy(np.maximum(fg - f_star, 1e-16),
                         color=cmap_gd[i], label=f"GD {label}")
        axes[0].semilogy(np.maximum(fn - f_star, 1e-16),
                         color=cmap_nt[i], label=f"NT {label}")
        axes[1].semilogy(gng, color=cmap_gd[i], label=f"GD {label}")
        axes[1].semilogy(dn, color=cmap_nt[i], label=f"NT {label}")
    axes[0].set(xlabel="iteration $k$", ylabel=r"$f(x_k)-f^\star$",
                title="suboptimality across sizes")
    axes[1].set(xlabel="iteration $k$", ylabel="optimality measure",
                title=r"$\|\nabla f\|$ / Newton decrement")
    for ax in axes:
        ax.legend(fontsize=8, ncol=2, loc="upper right")
    plt.tight_layout()
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    instances = [(2, 5), (10, 30), (50, 200), (100, 500)]
    results = {}
    for n, m in instances:
        gd, nt = run_instance(n, m, seed=42)
        plot_run(gd, nt, n, m, f"figures/task4_n{n}_m{m}.png")
        results[f"n={n},m={m}"] = (gd, nt)
    plot_summary(results, "figures/task4_summary.png")