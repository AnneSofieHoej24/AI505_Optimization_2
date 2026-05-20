import numpy as np

def strong_backtracking(f, nabla, x, d, alpha=1, beta=1e-4, sigma=0.1):
    """Finds a step length alpha that satisfies the Strong Wolfe Conditions.
    The algorithm consists of two phases, Bracketing and Zoom"""
    y0, g0, y_prev, alpha_prev = f(x), nabla(x) @ d, None, 0
    alpha_lo, alpha_hi = None, None
    rejected     = []   # Failed, shrink
    tried_alpha = []

    # Bracket phase
    while True:
        tried_alpha.append(alpha)
        y = f(x + alpha*d)
        if y > y0 + beta*alpha*g0 or (y_prev is not None and y >= y_prev):
            alpha_lo, alpha_hi = alpha_prev, alpha
            break
        dir_gradient = nabla(x + alpha*d) @ d
        if abs(dir_gradient) <= -sigma * g0:
            return alpha, rejected, tried_alpha
        elif dir_gradient >= 0:
            alpha_lo, alpha_hi = alpha, alpha_prev
            break
        y_prev, alpha_prev, alpha = y, alpha, 2 * alpha

    # Zoom phase
    ylo = f(x + alpha_lo*d)
    while abs(alpha_hi - alpha_lo) > 1e-10:
        alpha = (alpha_lo + alpha_hi)/2
        rejected.append(alpha)
        y = f(x + alpha*d)
        if y > y0 + beta*alpha*g0 or y >= ylo:
            alpha_hi = alpha
        else:
            g = nabla(x + alpha*d) @ d
            if abs(g) <= -sigma*g0:
                return alpha, rejected, tried_alpha
            elif g*(alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha
    return alpha_lo, rejected, tried_alpha


def backtracking_line_search(f, grad, x, d, alpha_0=0.5, p=0.5, beta=1e-4):
    """ Finds a step length alpha using the Armijo sufficient decrease condition."""
    y, g, alpha = f(x), grad(x), alpha_0    
    while ( f(x + alpha * d) > y + beta * alpha * np.dot(g, d) ) :
        alpha *= p
    return alpha