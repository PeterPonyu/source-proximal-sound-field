"""Exact uniform-ball cross-covariance between targets anywhere (inside, on or outside the ball) and microphones
outside it (PLAN.md). G = exp(ikd)/d; K(q, x) = (1/|B|) int_B G(q,s) conj G(x,s) ds = (3k^2/R^3) sum_n (2n+1) P_n I_n.

Radial integrals by the Lommel closed form  int t^2 u_n v_n dt = t^3/2 [u_n v_n - (u_{n-1} v_{n+1} + u_{n+1} v_{n-1})/2]
for any two solutions u, v of the spherical Bessel equation of order n (j_{-1}(t) = cos t / t, y_{-1}(t) = sin t / t).
"""
from __future__ import annotations
import numpy as np
from scipy.special import spherical_jn, spherical_yn

NMAX = 40
RQ_MIN = 1e-5   # m: targets closer than this to the centre are evaluated at this radius (keeps y_n finite; the
                # field is smooth there, so the displacement changes it negligibly)


def _jy(nmax, t):
    """j_n(t), y_n(t) for n = -1..nmax+1 (rows), t an array."""
    t = np.asarray(t, float)
    n = np.arange(0, nmax + 2)[:, None]
    j = spherical_jn(n, t[None, :]); y = spherical_yn(n, t[None, :])
    jm = (np.cos(t) / t)[None, :]; ym = (np.sin(t) / t)[None, :]
    return np.vstack((jm, j)), np.vstack((ym, y))      # index i <-> order i - 1


def _lommel(u, v, t, nmax):
    """t^3/2 [u_n v_n - (u_{n-1} v_{n+1} + u_{n+1} v_{n-1})/2] for n = 0..nmax; u, v from _jy (orders -1..nmax+1)."""
    i = np.arange(1, nmax + 2)
    return 0.5 * t[None, :] ** 3 * (u[i] * v[i] - 0.5 * (u[i - 1] * v[i + 1] + u[i + 1] * v[i - 1]))


def _legendre(u, nmax):
    P = [np.ones_like(u), u]
    for n in range(1, nmax):
        P.append(((2 * n + 1) * u * P[n] - n * P[n - 1]) / (n + 1))
    return np.stack(P[:nmax + 1])


def cross_K(q, x, c, R, k, nmax=NMAX):
    """(len(q), len(x)) cross-covariance; every microphone must lie outside the ball (r_x > R)."""
    q = np.atleast_2d(q); x = np.atleast_2d(x); c = np.asarray(c, float)
    dq = q - c; rq = np.maximum(np.linalg.norm(dq, axis=1), RQ_MIN)
    dx = x - c; rx = np.linalg.norm(dx, axis=1)
    if np.any(rx <= R):
        raise ValueError('a microphone lies inside the ball')
    u = np.clip((dq / rq[:, None]) @ (dx / rx[:, None]).T, -1, 1)
    P = _legendre(u, nmax)                                         # (n, Q, X)
    n = np.arange(nmax + 1)
    jx, yx = _jy(nmax, k * rx)
    hx_conj = (jx - 1j * yx)[1:nmax + 2]                           # conj h_n(k r_x), orders 0..nmax
    tR = np.array([k * R])
    jR, yR = _jy(nmax, tR)
    radial = np.zeros((nmax + 1, len(q)), complex)                 # radial factor of q, times k^3
    out_ = rq >= R
    if out_.any():
        jq, yq = _jy(nmax, k * rq[out_])
        hq = (jq + 1j * yq)[1:nmax + 2]
        radial[:, out_] = hq * _lommel(jR, jR, tR, nmax)
    if (~out_).any():
        t = k * rq[~out_]
        jq, yq = _jy(nmax, t)
        hq = (jq + 1j * yq)[1:nmax + 2]
        inner = _lommel(jq, jq, t, nmax)                           # int_0^{k rq} t^2 j_n^2 (zero at 0)
        hjR = _lommel(jR, jR, tR, nmax) + 1j * _lommel(yR, jR, tR, nmax)
        hjq = inner + 1j * _lommel(yq, jq, t, nmax)
        radial[:, ~out_] = hq * inner + jq[1:nmax + 2] * (hjR - hjq)
    radial /= k ** 3
    coef = (3 * k ** 2 / R ** 3) * (2 * n + 1)
    return np.einsum('n,nq,nx,nqx->qx', coef, radial, hx_conj, P)


def reference_cross_K(q1, x, c, R, k, n_dir=5000, n_t=64):
    """Singularity-free reference for one target: spherical coordinates centred on the target, Gauss-Legendre in the
    distance to the ball boundary along each of n_dir Fibonacci directions."""
    i = np.arange(n_dir) + 0.5
    phi = np.arccos(1 - 2 * i / n_dir); th = np.pi * (1 + 5 ** 0.5) * i
    w_dir = 4 * np.pi / n_dir
    D = np.stack((np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)), 1)
    a = q1 - c
    b = D @ a; cc = a @ a - R * R
    disc = b * b - cc
    tmin = np.maximum(-b - np.sqrt(np.maximum(disc, 0)), 0.0); tmax = -b + np.sqrt(np.maximum(disc, 0))
    ok = (disc > 0) & (tmax > tmin)
    g, gw = np.polynomial.legendre.leggauss(n_t)
    tot = np.zeros(len(x), complex)
    for d, t0, t1 in zip(D[ok], tmin[ok], tmax[ok]):
        t = 0.5 * (t1 - t0) * g + 0.5 * (t1 + t0); wt = 0.5 * (t1 - t0) * gw
        s = q1[None, :] + t[:, None] * d[None, :]
        Gq = np.exp(1j * k * t) / t
        dxs = np.linalg.norm(x[None, :, :] - s[:, None, :], axis=2)
        Gx = np.exp(1j * k * dxs) / dxs
        tot += (wt * t ** 2 * Gq) @ Gx.conj()
    return tot * w_dir / (4 / 3 * np.pi * R ** 3)
