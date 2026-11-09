"""Cramer-Rao bound for point-source coordinates under the estimator's model (post hoc, not registered).

Model in one bin: y = a g(s) + e, e ~ CN(0, sigma^2 (K + rho I)), with g the free-field Green vector
G(r_m, s) = exp(ik|r_m - s|)/|r_m - s|, K the sinc kernel matrix of the homogeneous remainder, a complex amplitude
and sigma^2, rho unknown. The mean depends on (s, a) and the covariance on (sigma^2, rho), so the two blocks of the
Fisher information decouple and the bound on s does not depend on whether sigma^2 and rho are known. Profiling the
complex amplitude gives the efficient information (conditional, or deterministic, bound)

    J(s) = 2 eta Re[ D^H P D ],   D = L^{-1} dG/ds / nu,   P = I - u u^H / (u^H u),   u = L^{-1} G / nu,

with L the Cholesky factor of K + rho I, nu^2 = mean_m |G_m|^2 and eta = |a|^2 nu^2 / sigma^2, the power of the point
term relative to the remainder at the microphones (|a| refers to the raw Green vector). Coordinates shared across
bins add the per-bin informations; each bin alone uses its own. The RMS bound is sqrt(tr J^{-1}).
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import cholesky, solve_triangular

LN_RHO = (np.log(1e-7), np.log(10.0))   # the estimator's bounds on the log ridge


def green(x, s, k):
    """Green vector G (M,) and its derivative with respect to the source coordinates (M, 3)."""
    d = s[None, :] - x
    r = np.linalg.norm(d, axis=1)
    G = np.exp(1j * k * r) / r
    dG = (G * (1j * k - 1.0 / r) / r)[:, None] * d
    return G, dG


def sinc_kernel(x, k):
    r = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    return np.sinc(k * r / np.pi)


def info(x, s, k, eta, rho, K=None):
    """3 x 3 efficient Fisher information for s in one bin."""
    K = sinc_kernel(x, k) if K is None else K
    L = cholesky(K + rho * np.eye(len(x)), lower=True)
    G, dG = green(x, s, k)
    nu = np.sqrt(np.mean(np.abs(G) ** 2))
    u = solve_triangular(L, G / nu, lower=True)
    D = solve_triangular(L, dG / nu, lower=True)
    PD = D - np.outer(u, u.conj() @ D) / np.real(u.conj() @ u)
    return 2.0 * eta * np.real(D.conj().T @ PD)


def rms_bound(J):
    """sqrt(tr J^{-1}); inf when J is singular to working precision."""
    w = np.linalg.eigvalsh(0.5 * (J + J.T))
    return float(np.sqrt(np.sum(1.0 / w))) if w.min() > 1e-12 * max(w.max(), 1e-300) else float('inf')


def profile(x, y, s, k, ln_rho, K=None):
    """Profiled conditional log likelihood at fixed s and rho; returns (value, eta_hat, a_hat, sigma2_hat)."""
    K = sinc_kernel(x, k) if K is None else K
    M = len(x)
    L = cholesky(K + np.exp(ln_rho) * np.eye(M), lower=True)
    G, _ = green(x, s, k)
    u = solve_triangular(L, G, lower=True)
    z = solve_triangular(L, y, lower=True)
    a = (u.conj() @ z) / np.real(u.conj() @ u)
    s2 = np.real(np.vdot(z - a * u, z - a * u)) / M
    val = -M * np.log(s2) - 2.0 * np.sum(np.log(np.real(np.diag(L))))
    return val, float(np.abs(a) ** 2 * np.mean(np.abs(G) ** 2) / s2), a, s2


def plugin(x, y, s, k, n_grid=57):
    """Maximum-likelihood eta and rho at fixed s: grid over the log ridge, then golden-section refinement."""
    from scipy.optimize import minimize_scalar
    K = sinc_kernel(x, k)
    grid = np.linspace(*LN_RHO, n_grid)
    vals = [profile(x, y, s, k, g, K)[0] for g in grid]
    i = int(np.argmax(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, n_grid - 1)]
    r = minimize_scalar(lambda t: -profile(x, y, s, k, t, K)[0], bounds=(lo, hi), method='bounded',
                        options=dict(xatol=1e-4))
    t = r.x if -r.fun >= vals[i] else grid[i]
    val, eta, a, s2 = profile(x, y, s, k, t, K)
    return dict(ln_rho=float(t), rho=float(np.exp(t)), eta=eta, sigma2=float(s2), loglik=float(val),
                at_bound=bool(t <= LN_RHO[0] + 1e-3 or t >= LN_RHO[1] - 1e-3))
