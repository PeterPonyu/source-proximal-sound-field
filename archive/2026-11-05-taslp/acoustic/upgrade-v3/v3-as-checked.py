"""Estimator version 3 (development; PLAN.md): the version-2 model with a convergence fix in its fitting procedure.

Changes against `v2` (which is imported read-only and not modified):
  1. zoomed rescan around the coarse scan candidates;
  2. closed-form per-point weight reset after each refinement, kept only if it lowers the objective, then refine again;
  3. coordinates (and ball centres) optimised in units of 1/k_mid; iteration cap 1000 for points and balls.
`Opts(zoom=False, resets=0, scale=False, maxiter=200, region_maxiter=300)` reproduces version 2 exactly.
"""
from __future__ import annotations
from dataclasses import dataclass
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import estimators as E  # noqa: E402
import v2  # noqa: E402


@dataclass(frozen=True)
class Opts:
    zoom: bool = True
    resets: int = 3
    scale: bool = True
    maxiter: int = 1000
    region_maxiter: int = 1000
    zoom_max_nodes: int = 30000
    reset_tol: float = 1e-9


V2_OPTS = Opts(zoom=False, resets=0, scale=False, maxiter=200, region_maxiter=300)
V3_OPTS = Opts()


# ------------------------------------------------------------------ scaled bounded minimisation

def _min(fun, x0, bounds, scale, maxiter, args=()):
    """L-BFGS-B on z = x / scale (elementwise); returns the result with x, jac in the original variables."""
    lb = np.array([b[0] for b in bounds]); ub = np.array([b[1] for b in bounds])
    x0 = np.clip(x0, lb, ub)

    def f(z):
        v, g = fun(z * scale, *args)
        return v, g * scale
    r = minimize(f, x0 / scale, jac=True, method='L-BFGS-B', bounds=list(zip(lb / scale, ub / scale)),
                 options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-6))
    r.x = np.clip(r.x * scale, lb, ub)
    return r


def _point_scale(band, K, opts):
    s = np.ones(3 * K + K * band.F + band.F)
    if opts.scale:
        s[:3 * K] = band.lam_mid() / (2 * np.pi)
    return s


# ------------------------------------------------------------------ scan with zoom

def grid_spacing(nodes):
    ax = np.unique(np.round(nodes[:, 0], 9))
    return float(ax[1] - ax[0]) if len(ax) > 1 else 0.0


def scan(band, nodes, S, W, R, starts, suppress, opts):
    cands = v2.band_scan(band, nodes, S, W, R, starts, suppress)
    if not opts.zoom:
        return cands
    h = grid_spacing(nodes)
    fine = min(0.5 * h, band.c / band.freqs.max() / 6)
    out = []
    for node, wn, gain in cands:
        zn = E.grid_nodes(node - h, node + h, fine, max_nodes=opts.zoom_max_nodes)
        z = v2.band_scan(band, zn, S, W, R, 1, 0.0)
        out.append(z[0] if z and z[0][2] >= gain else (node, wn, gain))
    return out


# ------------------------------------------------------------------ weight reset and refinement cycle

def reset_weights(band, theta, K):
    """Each point's per-bin weights in turn set to the closed-form optimum of the rank-one evidence gain given the
    other fitted terms (Eq. 7 of the manuscript), ridges held."""
    S, lw, lr = v2._split(theta, K, band.F)
    S = S.reshape(K, 3); lw = lw.copy(); R = np.exp(lr)
    for j in range(K):
        oth = [i for i in range(K) if i != j]
        z = v2.band_scan(band, S[j:j + 1], S[oth], np.exp(lw[:, oth]), R, 1, 0.0)
        if z:
            lw[:, j] = np.log(np.clip(z[0][1], np.exp(v2.LOGW2[0]), np.exp(v2.LOGW2[1])))
    return np.concatenate((S.ravel(), lw.ravel(), lr))


def refine_points(band, th0, K, opts, stats):
    lo, hi = band.search_box()
    bounds = v2._bounds_points(band, K, lo, hi)
    sc = _point_scale(band, K, opts)
    r = _min(v2.joint_objective, th0, bounds, sc, opts.maxiter, args=(band, K))
    stats.append(dict(K=K, nit=int(r.nit), cap=bool(r.nit >= opts.maxiter), success=bool(r.success), stage='refine'))
    for _ in range(opts.resets):
        th_r = reset_weights(band, r.x, K)
        v_r = v2.joint_objective(th_r, band, K)[0]
        if not v_r < r.fun - opts.reset_tol:
            break
        r2 = _min(v2.joint_objective, th_r, bounds, sc, opts.maxiter, args=(band, K))
        stats.append(dict(K=K, nit=int(r2.nit), cap=bool(r2.nit >= opts.maxiter), success=bool(r2.success), stage='reset'))
        if r2.fun < r.fun - opts.reset_tol:
            r = r2
        else:
            break
    return r


def fit_band(band, kmax=v2.KMAX, starts=3, opts=V3_OPTS, stats=None):
    """Greedy sequential fit with coordinates shared across the bins (version 2 structure); models for 0..kmax."""
    stats = [] if stats is None else stats
    lo, hi = band.search_box()
    nodes = E.grid_nodes(lo, hi, 0.25 * band.lam_mid(), max_nodes=v2.SCAN_NODES)
    R0 = np.array([v2.remainder_rho(p) for p in band.probs])
    ev0 = float(sum(v2.remainder_evidence(p, r) for p, r in zip(band.probs, R0)))
    models = {0: dict(S=np.zeros((0, 3)), W=np.zeros((band.F, 0)), R=R0, evidence=ev0, converged=True)}
    S, W, R = np.zeros((0, 3)), np.zeros((band.F, 0)), R0
    for K in range(1, kmax + 1):
        cands = scan(band, nodes, S, W, R, starts, 0.5 * band.lam_mid(), opts)
        best, bstats = None, None
        for node, wn, _ in cands:
            S0 = np.vstack((S, node[None, :]))
            lw0 = np.log(np.maximum(np.hstack((W, wn[:, None])), 1e-4))
            th0 = np.concatenate((S0.ravel(), lw0.ravel(), np.log(R)))
            st = []
            r = refine_points(band, th0, K, opts, st)
            if best is None or r.fun < best.fun:
                best, bstats = r, st
        stats.extend(bstats)
        Sv, lw, lr = v2._split(best.x, K, band.F)
        S, W, R = Sv.reshape(K, 3), np.exp(lw), np.exp(lr)
        models[K] = dict(S=S.copy(), W=W.copy(), R=R.copy(), evidence=-float(best.fun), converged=bool(best.success),
                         theta=best.x.copy(), cap=any(s['cap'] for s in bstats[-1:]))
    return models


def cv_order(band, kmax, folds, opts=V3_OPTS, **kw):
    err = np.zeros(kmax + 1)
    for held in folds:
        train = np.setdiff1d(np.arange(band.M), held)
        sub = band.subset(train)
        models = fit_band(sub, kmax, opts=opts, **kw)
        for K, m in models.items():
            for f, p in enumerate(band.probs):
                pred = v2.predict_points(sub, m, f, q=band.x[held])
                err[K] += float(np.sum(np.abs(p.y[held] - pred) ** 2)) / float(np.vdot(p.y, p.y).real)
    return int(np.argmin(err)), err


# ------------------------------------------------------------------ ball extension

def fit_regions(band, J, starts, opts=V3_OPTS, stats=None):
    lo, hi = band.search_box()
    bounds = ([(lo[c], hi[c]) for _ in range(J) for c in range(3)] + [v2.LOGRAD] * J + [v2.LOGW2] * (J * band.F)
              + [v2.LOGRHO2] * band.F)
    sc = np.ones(len(bounds))
    if opts.scale:
        sc[:3 * J] = band.lam_mid() / (2 * np.pi)
    best = None
    for cen0, W0, R0 in starts:
        t0 = np.concatenate((np.asarray(cen0).ravel(), np.full(J, np.log(0.05)), np.log(np.maximum(W0, 1e-4)).ravel(),
                             np.log(R0)))
        r = _min(v2.region_joint_objective, t0, bounds, sc, opts.region_maxiter, args=(band, J))
        if stats is not None:
            stats.append(dict(J=J, nit=int(r.nit), cap=bool(r.nit >= opts.region_maxiter), success=bool(r.success)))
        if best is None or r.fun < best.fun:
            best = r
    cen, lrad, lw, lr = v2._rsplit(best.x, J, band.F)
    return dict(centres=cen.reshape(J, 3), radii=np.exp(lrad), lw=lw, R=np.exp(lr), evidence=-float(best.fun),
                converged=bool(best.success), pruned=int(np.sum(np.all(lw <= v2.LOGW2[0] + 1e-3, axis=0))))


def extend(band, m, K, opts=V3_OPTS, stats=None):
    starts = [(m['S'], m['W'], m['R'])]
    cen, W, R0 = v2.scan_start(band, K)
    if len(cen) == K:
        starts.append((cen, W, R0))
    return fit_regions(band, K, starts, opts, stats)


# ------------------------------------------------------------------ primary estimator

def primary(band, seed, opts=V3_OPTS):
    """Shared coordinates, order by five-fold microphone cross-validation (cap 5), extension to balls.
    Returns (predict(f) -> target prediction, info)."""
    pstats, rstats = [], []
    models = fit_band(band, 5, opts=opts, stats=pstats)
    K, cv_err = cv_order(band, 5, E.make_folds(band.M, 5, seed), opts=opts)
    info = dict(K=K, cv_err=cv_err.tolist(), evidence={k: m['evidence'] for k, m in models.items()},
                point_stats=pstats)
    if K == 0:
        return (lambda f: v2.predict_points(band, models[0], f)), models, info
    rm = extend(band, models[K], K, opts, rstats)
    info['region_stats'] = rstats
    return (lambda f: v2.predict_regions(band, rm, f)), models, info
