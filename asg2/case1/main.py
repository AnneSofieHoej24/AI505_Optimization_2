import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from scipy.optimize import minimize

A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1]], dtype=float)
b = np.array([1, 1, 1, 1, 1.5])

fig, ax = plt.subplots(figsize=(6, 6))

xx, yy = np.meshgrid(np.linspace(-1.4, 1.4, 400), np.linspace(-1.4, 1.4, 400))
pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
feasible = np.all(pts @ A.T <= b, axis=1).reshape(xx.shape)
ax.contourf(xx, yy, feasible, levels=[0.5, 1.5], colors=["#00D6FC"], alpha=0.75, label= "feasible region")
ax.contour(xx, yy, feasible, levels=[0.5], colors=["#FC2600"], linewidths=2)
ax.text(0, -0.5, "Feasible region", ha="center", va="center", fontsize=13,
        color="black",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")])


# The barrier function
def f(x):
    s = b - A @ x
    if np.any(s <= 0):
        return np.inf  # outside feasible region
    return -np.sum(np.log(s))


def grad(x):
    return A.T @ (1.0 / (b - A @ x))


def hess(x):
    s = b - A @ x
    return A.T @ np.diag(1.0 / s**2) @ A


point = (0, 0)
print("f(start) =", f(np.array(point)))

path_list = [np.array(point, dtype=float)]
res = minimize(
    f,
    point,
    jac=grad,
    hess=hess,
    method="trust-exact",
    callback=lambda xk: path_list.append(xk.copy()),
)
x_star = res.x
path = np.array(path_list)
min_slack = np.min(b - A @ x_star)
print("x* =", x_star)
print("f(x*) =", res.fun, "iters =", res.nit)
print("min slack =", min_slack)


ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.4, 1.4)
ax.set_aspect("equal")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.axhline(0, color="k", linewidth=0.5)
ax.axvline(0, color="k", linewidth=0.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
ax.scatter(*point, color="k", zorder=5, label=f"Starting point = {point}")
ax.legend(loc="upper left")
plt.savefig("figures/simpelPlot.png", dpi=200, bbox_inches="tight")

ax.plot(
    path[:, 0],
    path[:, 1],
    "o",
    color="#222",
    markersize=4,
    zorder=4,
    label="Newton path",
)
for p0, p1 in zip(path[:-1], path[1:]):
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops=dict(arrowstyle="->", linestyle=":", color="#222"),
        zorder=4,
    )
ax.scatter(
    *x_star,
    color="red",
    zorder=6,
    label="Center" + f" $ \\approx$ ({x_star[0]:.3f}, {x_star[1]:.3f})",
)
ax.legend(loc="upper left")
plt.savefig("figures/newton_path.png", dpi=200, bbox_inches="tight")


# --- Task 3.2 ---------------------------------------------------------------
# Scale the 5th constraint by gamma. Feasible set unchanged (both sides scale
# the same way). Barrier center also unchanged in theory: -log(gamma*s_i) =
# -log(gamma) - log(s_i), a constant in x, so gradient/Hessian identical.

def solve_center(A_, b_, x0=(0.0, 0.0)):
    def f_(x):
        s = b_ - A_ @ x
        if np.any(s <= 0):
            return np.inf
        return -np.sum(np.log(s))
    def g_(x):
        return A_.T @ (1.0 / (b_ - A_ @ x))
    def h_(x):
        s = b_ - A_ @ x
        return A_.T @ np.diag(1.0 / s**2) @ A_
    r = minimize(f_, x0, jac=g_, hess=h_, method="trust-exact")
    return r.x

gammas = [0.1, 1, 10, 100]
fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.contourf(xx, yy, feasible, levels=[0.5, 1.5], colors=["#00D6FC"], alpha=0.5)
ax2.contour(xx, yy, feasible, levels=[0.5], colors=["#FC2600"], linewidths=2)

x_old = x_star
ax2.scatter(*x_old, color="red", s=80, zorder=6, label=f"original ({x_old[0]:.3f}, {x_old[1]:.3f})")

colors = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
for g, c in zip(gammas, colors):
    A_g = A.copy()
    b_g = b.copy()
    A_g[4] *= g
    b_g[4] *= g
    x_new = solve_center(A_g, b_g)
    dist = np.linalg.norm(x_old - x_new)
    print(f"gamma={g:>5}: x_new = {x_new}, ||x_old - x_new|| = {dist:.3e}")
    ax2.scatter(*x_new, color=c, marker="x", s=80, zorder=5,
                label=f"$\\gamma={g}$: ({x_new[0]:.3f}, {x_new[1]:.3f})")

ax2.set_xlim(-1.4, 1.4)
ax2.set_ylim(-1.4, 1.4)
ax2.set_aspect("equal")
ax2.set_xlabel("$x_1$")
ax2.set_ylabel("$x_2$")
ax2.axhline(0, color="k", linewidth=0.5)
ax2.axvline(0, color="k", linewidth=0.5)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig("figures/task32_centers.png", dpi=200, bbox_inches="tight")
plt.show()


# --- Task 3.3 ---------------------------------------------------------------
# Normalize: divide each row by b_i so each constraint becomes a_i^T x <= 1.
# Same invariance as 3.2: per-row scaling leaves the center unchanged.

A_norm = A / b[:, None]
b_norm = np.ones_like(b)
x_norm = solve_center(A_norm, b_norm)
dist = np.linalg.norm(x_star - x_norm)
print(f"x_original = {x_star}")
print(f"x_normalized = {x_norm}")
print(f"||x_original - x_normalized|| = {dist:.3e}")

fig3, ax3 = plt.subplots(figsize=(6, 6))
ax3.contourf(xx, yy, feasible, levels=[0.5, 1.5], colors=["#00D6FC"], alpha=0.5)
ax3.contour(xx, yy, feasible, levels=[0.5], colors=["#FC2600"], linewidths=2)
ax3.scatter(*x_star, color="red", s=100, zorder=6,
            label=f"original  ({x_star[0]:.4f}, {x_star[1]:.4f})")
ax3.scatter(*x_norm, color="black", marker="x", s=100, zorder=7,
            label=f"normalized ({x_norm[0]:.4f}, {x_norm[1]:.4f})")
ax3.set_xlim(-1.4, 1.4)
ax3.set_ylim(-1.4, 1.4)
ax3.set_aspect("equal")
ax3.set_xlabel("$x_1$")
ax3.set_ylabel("$x_2$")
ax3.axhline(0, color="k", linewidth=0.5)
ax3.axvline(0, color="k", linewidth=0.5)
ax3.grid(True, alpha=0.3)
ax3.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig("figures/task33_normalized.png", dpi=200, bbox_inches="tight")
plt.show()
