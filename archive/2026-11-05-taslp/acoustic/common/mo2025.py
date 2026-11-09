"""Reimplementation of the source-region Gaussian-process kernel of Matsuda & Otani (JASA 157:1403, 2025).

kappa(r_i, r_j) = int_ball G(r_i|s) conj(G(r_j|s)) eta(|s - O|) ds with G = exp(i k r)/(4 pi r) and the Gaussian
source density eta(r) = (8 pi^3 sigma^6)^(-1/2) exp(-r^2 / (2 sigma^2)) restricted to the ball of radius R
(their proposed 'Gs' weight; sigma = R/3 as in their earlier figures). The integral is evaluated by quadrature
(Gauss-Legendre in radius times a Fibonacci sphere) instead of their spherical-wave series, which also covers
targets inside a source region. Several regions add. The noise variance is fixed from the known SNR,
sigma_e^2 = sigma_y^2 / (10^(SNR/10) + 1); centres and radii are tuned one region at a time by exhaustive grid
search scored by leave-one-out cross-validation, as in their Algorithm 1.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist


def fib(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack((np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)))


def ball_quadrature(radius, k, sigma_ratio=1 / 3, n_r=None, n_a=None, density='gauss'):
    """Nodes (n, 3) relative to the centre and weights eta * dV for the ball of the given radius."""
    kr = k * radius
    n_r = n_r or int(np.clip(np.ceil(kr) + 6, 8, 40))
    n_a = n_a or int(np.clip(2 * (np.ceil(kr) + 4) ** 2, 50, 1200))
    x, w = np.polynomial.legendre.leggauss(n_r)
    r = 0.5 * radius * (x + 1); wr = 0.5 * radius * w
    d = fib(n_a)
    if density == 'gauss':
        sig = sigma_ratio * radius
        eta = (8 * np.pi ** 3 * sig ** 6) ** -0.5 * np.exp(-r ** 2 / (2 * sig ** 2))
    else:
        eta = np.full_like(r, 3 / (4 * np.pi * radius ** 3))
    nodes = (r[:, None, None] * d[None, :, :]).reshape(-1, 3)
    wts = (wr * r ** 2 * eta)[:, None] * np.full(n_a, 4 * np.pi / n_a)[None, :]
    return nodes, wts.reshape(-1)


def green4pi(points, nodes, k):
    d = cdist(points, nodes)
    return np.exp(1j * k * d) / (4 * np.pi * d)


def region_factor(points, centre, radius, k, quad=None):
    nodes, wts = quad if quad is not None else ball_quadrature(radius, k)
    return green4pi(points, np.asarray(centre)[None, :] + nodes, k) * np.sqrt(wts)[None, :]


def kernel(points_a, points_b, regions, k):
    out = np.zeros((len(points_a), len(points_b)), complex)
    for c, R in regions:
        q = ball_quadrature(R, k)
        Fa = region_factor(points_a, c, R, k, q)
        Fb = Fa if points_b is points_a else region_factor(points_b, c, R, k, q)
        out += Fa @ Fb.conj().T
    return out


def loo_error(K, y, s2):
    Ci = np.linalg.inv(K + s2 * np.eye(len(y)))
    a = Ci @ y
    return float(np.sum(np.abs(a / np.real(np.diag(Ci))) ** 2))


def fit_grid(x, y, k, prior, s2, radii=(0.05, 0.10, 0.15, 0.20), step=0.05, passes=1):
    """Algorithm 1: for each region in turn, exhaustive LOO grid search over centres on a `step` lattice inside
    the prior region and over the radius menu, the other regions' kernels held fixed."""
    regions = [(np.asarray(c, float), float(R)) for c, R in prior]
    cache = {}

    def factor(c, R):
        key = (tuple(np.round(c, 6)), R)
        if key not in cache:
            cache[key] = region_factor(x, c, R, k)
        return cache[key]
    for _ in range(passes):
        for s in range(len(regions)):
            c0, R0 = prior[s]
            g = np.arange(-R0, R0 + 1e-9, step)
            cands = np.array([np.asarray(c0) + np.array(v) for v in np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
                              if np.linalg.norm(v) <= R0 + 1e-9])
            others = np.zeros((len(x), len(x)), complex)
            for t, (c, R) in enumerate(regions):
                if t != s:
                    F = factor(c, R); others += F @ F.conj().T
            best = None
            for c in cands:
                for R in radii:
                    F = factor(c, R)
                    e = loo_error(others + F @ F.conj().T, y, s2)
                    if best is None or e < best[0]:
                        best = (e, c, R)
            regions[s] = (best[1], best[2])
    return regions


def predict(x, y, q, k, regions, s2):
    K = kernel(x, x, regions, k)
    Kq = kernel(q, x, regions, k)
    return Kq @ np.linalg.solve(K + s2 * np.eye(len(x)), y)
