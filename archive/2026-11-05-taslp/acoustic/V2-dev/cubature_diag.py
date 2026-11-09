"""Post hoc descriptive diagnostic (labelled; development data only; no registered rule changes).

The registered region kernel (estimators.region_factor: K = F F^H) is a 33-node cubature of a ball with a uniform
source density. Anderson, Teal & Poletti (2016, eq. 40) give its exact spherical-wave series; the cubature's error grows
with kR. This script
  (a) records the fitted radii R and k_max R of the three registered methods that use the region kernel: the extension
      step of j_cv_ext (v2.extend), mo26_cap5 (v2.mo26_cap) and gsmf (v2.gsmf_fit);
  (b) reruns the same three with the exact series in place of the cubature (the same code: estimators.mo26_objective
      and estimators.region_predict are swapped for series versions with analytic gradients) and reports the
      per-problem dB change, paired with a cubature run in the same process.
The cubature runs must reproduce the development rows (V2-dev/dev1-box) bit for bit; that is checked.

OPENBLAS_NUM_THREADS=1 nice -n 10 python3 -B cubature_diag.py OUT_DIR [--workers 4] [--check]
"""
from __future__ import annotations
import argparse
import contextlib
import json
import math
import sys
import time
import traceback
import warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import spherical_jn, spherical_yn
from scipy.spatial.distance import cdist
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common')); sys.path.insert(0, str(HERE))
import estimators as E  # noqa: E402
import mo2025 as MO  # noqa: E402
import v2  # noqa: E402
import run_dev as RD  # noqa: E402
import prox_dev as PD  # noqa: E402

NMAX = 40
FINE = (24, 1200)


# ------------------------------------------------------------------ exact uniform-ball kernel (G = exp(ikd)/d)

def a_uniform(R, k, deriv=False):
    """Coefficients a_n = (3 k^2 / 2)(2n+1)[j_n(kR)^2 - j_{n-1}(kR) j_{n+1}(kR)] (Anderson et al. 2016, eq. 40, times
    16 pi^2 for G = exp(ikd)/d), and d a_n / dR."""
    n = np.arange(NMAX + 1)
    x = k * R
    jn = spherical_jn(n, x); jp1 = spherical_jn(n + 1, x)
    jm1 = np.r_[np.cos(x) / x, jn[:-1]]
    a = 1.5 * k ** 2 * (2 * n + 1) * (jn ** 2 - jm1 * jp1)
    if not deriv:
        return a
    djn = spherical_jn(n, x, derivative=True); djp1 = spherical_jn(n + 1, x, derivative=True)
    djm1 = np.r_[-np.sin(x) / x - np.cos(x) / x ** 2, djn[:-1]]
    da = 1.5 * k ** 2 * (2 * n + 1) * k * (2 * jn * djn - djm1 * jp1 - jm1 * djp1)
    return a, da


def _geom(x, c, k):
    d = x - c
    r = np.linalg.norm(d, axis=1)
    dh = d / r[:, None]
    u = np.clip(dh @ dh.T, -1, 1)
    n = np.arange(NMAX + 1)[:, None]
    h = spherical_jn(n, k * r) + 1j * spherical_yn(n, k * r)
    hp = spherical_jn(n, k * r, derivative=True) + 1j * spherical_yn(n, k * r, derivative=True)
    return r, dh, u, h, hp


def _legendre(u):
    P = [np.ones_like(u), u]
    D = [np.zeros_like(u), np.ones_like(u)]
    for n in range(1, NMAX):
        P.append(((2 * n + 1) * u * P[n] - n * P[n - 1]) / (n + 1))
        D.append(D[n - 1] + (2 * n + 1) * P[n])
    return P, D


def series_K(x, c, R, k):
    r, dh, u, h, hp = _geom(x, c, k)
    a = a_uniform(R, k)
    P, _ = _legendre(u)
    return sum(a[n] * (h[n][:, None] * h[n].conj()[None, :]) * P[n] for n in range(NMAX + 1))


def series_grad(x, c, R, k, B):
    """Re tr(B K) and its gradients with respect to the centre (3,) and the radius."""
    a, da = a_uniform(R, k, deriv=True)
    r, dh, u, h, hp = _geom(x, c, k)
    P, D = _legendre(u)
    BT = B.T
    trK = 0.0; trKr = 0.0
    gc = np.zeros(3, complex)
    Rm = np.zeros((len(x), len(x)), complex)
    for n in range(NMAX + 1):
        hh = h[n][:, None] * h[n].conj()[None, :]
        Q = BT * P[n]
        t = np.sum(Q * hh)
        trK += a[n] * t; trKr += da[n] * t
        gc += -k * a[n] * ((hp[n] * (Q @ h[n].conj()))[:, None] * dh).sum(0)
        gc += -k * a[n] * ((hp[n].conj() * (Q.T @ h[n]))[:, None] * dh).sum(0)
        Rm += a[n] * BT * hh * D[n]
    ir = 1 / r
    du = (np.einsum('ij,jc->c', Rm * ir[:, None], dh) - np.einsum('ij,ic->c', Rm * u * ir[:, None], dh)
          + np.einsum('ij,ic->c', Rm * ir[None, :], dh) - np.einsum('ij,jc->c', Rm * u * ir[None, :], dh))
    gc += -du
    return float(np.real(trK)), np.real(gc), float(np.real(trKr))


def cross_K(q, x, c, R, k):
    """Targets q against microphones x: the series where the target is more than 2 cm outside the ball, a dense uniform
    cubature otherwise."""
    a = a_uniform(R, k)
    dx = x - c; rx = np.linalg.norm(dx, axis=1)
    n = np.arange(NMAX + 1)[:, None]
    hx = spherical_jn(n, k * rx) + 1j * spherical_yn(n, k * rx)
    dq = q - c; rq = np.linalg.norm(dq, axis=1)
    out = np.zeros((len(q), len(x)), complex)
    far = rq > R + 0.02
    if far.any():
        hq = spherical_jn(n, k * rq[far]) + 1j * spherical_yn(n, k * rq[far])
        u = np.clip((dq[far] / rq[far, None]) @ (dx / rx[:, None]).T, -1, 1)
        P, _ = _legendre(u)
        out[far] = sum(a[m] * hq[m][:, None] * hx[m].conj()[None, :] * P[m] for m in range(NMAX + 1))
    if (~far).any():
        nodes, w = MO.ball_quadrature(R, k, n_r=FINE[0], n_a=FINE[1], density='uniform')
        Fq = E.green(q[~far], c[None] + nodes, k) * np.sqrt(w)
        Fx = E.green(x, c[None] + nodes, k) * np.sqrt(w)
        out[~far] = Fq @ Fx.conj().T
    return out


# ------------------------------------------------------------------ series versions of the two registered functions

def mo26_objective_series(t, prob, J):
    """estimators.mo26_objective with the exact series kernel (same parameters, penalty, bounds and evidence)."""
    cen = t[:3 * J].reshape(J, 3); lrad = t[3 * J:4 * J]; lw = t[4 * J:5 * J]; rho = float(np.exp(t[5 * J]))
    rad = np.exp(lrad)
    M = prob.M
    C = prob.H + (rho + E.JITTER) * np.eye(M)
    store = []
    pen, g_pen = 0.0, np.zeros(3 * J + J)
    for j in range(J):
        dist = np.linalg.norm(prob.x - cen[j], axis=1) - rad[j]
        short = np.maximum(E.GUARD - dist, 0.0)
        pen += 1e6 * float(np.sum(short ** 2))
        u = (cen[j][None, :] - prob.x) / np.maximum(np.linalg.norm(prob.x - cen[j], axis=1), 1e-12)[:, None]
        g_pen[3 * j:3 * j + 3] += -2e6 * (short[:, None] * u).sum(axis=0)
        g_pen[3 * J + j] += 2e6 * float(np.sum(short)) * rad[j]
        if dist.min() <= 0.0:                     # a microphone inside a ball: the series is invalid there
            return 1e30, np.zeros_like(t)
        K = series_K(prob.x, cen[j], rad[j], prob.k)
        nrm = float(np.mean(np.real(np.diag(K))))
        sc = np.exp(lw[j]) / nrm
        C = C + sc * K
        store.append((K, nrm, sc))
    try:
        fac = cho_factor(C, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return 1e30, np.zeros_like(t)
    alpha = cho_solve(fac, prob.y)
    Q = float(np.vdot(prob.y, alpha).real)
    ev = -M * np.log(Q) - 2 * float(np.sum(np.log(np.diag(fac[0]).real)))
    Ci = cho_solve(fac, np.eye(M))
    G = (M / Q) * np.outer(alpha, alpha.conj()) - Ci          # d ev = Re tr(G dC)
    grad = np.zeros_like(t)
    Im = np.eye(M) / M
    for j, (K, nrm, sc) in enumerate(store):
        tGK, gcK, grK = series_grad(prob.x, cen[j], rad[j], prob.k, G)
        _, gcn, grn = series_grad(prob.x, cen[j], rad[j], prob.k, Im)
        grad[4 * J + j] = sc * tGK
        grad[3 * j:3 * j + 3] = sc * (gcK - tGK * gcn / nrm)
        grad[3 * J + j] = rad[j] * sc * (grK - tGK * grn / nrm)
    grad[5 * J] = rho * (M * float(np.vdot(alpha, alpha).real) / Q - float(np.real(np.trace(Ci))))
    gvec = -grad
    gvec[:4 * J] += g_pen
    return -ev + pen, gvec


def region_predict_series(prob, cen, rad, lw, rho):
    C = prob.H + (rho + E.JITTER) * np.eye(prob.M)
    Cq = prob.Hq.astype(complex)
    for j in range(len(cen)):
        K = series_K(prob.x, cen[j], rad[j], prob.k)
        s = np.exp(lw[j]) / float(np.mean(np.real(np.diag(K))))
        C = C + s * K
        Cq = Cq + s * cross_K(prob.q, prob.x, cen[j], rad[j], prob.k)
    return Cq @ np.linalg.solve(C, prob.y)


@contextlib.contextmanager
def exact_kernel():
    orig = E.mo26_objective, E.region_predict
    E.mo26_objective, E.region_predict = mo26_objective_series, region_predict_series
    try:
        yield
    finally:
        E.mo26_objective, E.region_predict = orig


# ------------------------------------------------------------------ checks and the diagnostic

def checks():
    """(1) the series against a dense uniform cubature away from the ball, and the registered 33-node cubature's error
    by kR; (2) the series objective's gradient against finite differences."""
    out = {}
    rng = np.random.default_rng(3)
    c = np.zeros(3)
    for f, R in ((125.0, 0.05), (500.0, 0.1), (1000.0, 0.15), (1000.0, 0.3), (1400.0, 0.3)):
        k = 2 * np.pi * f / 343.0
        dirs = rng.standard_normal((10, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        x = dirs * rng.uniform(R + 0.15, R + 0.6, 10)[:, None]
        Ks = series_K(x, c, R, k)
        nodes, w = MO.ball_quadrature(R, k, n_r=40, n_a=3000, density='uniform')
        F = E.green(x, c[None] + nodes, k) * np.sqrt(w)
        Kd = F @ F.conj().T
        F33 = E.region_factor(x, c, R, k)
        K33 = F33 @ F33.conj().T
        out[f'{int(f)}Hz_R{R}_kR{k * R:.2f}'] = dict(series_vs_dense=float(np.linalg.norm(Ks - Kd) / np.linalg.norm(Kd)),
                                                    cubature33_vs_series=float(np.linalg.norm(K33 - Ks) / np.linalg.norm(Ks)))
    band, T, *_ = RD.sim_band('A2-f500-M24', 400920)
    p = band.probs[2]
    J = 2
    t = np.r_[np.array([[0.6, 0.6, -0.5], [1.5, 0.2, -0.4]]).ravel(), np.log([0.12, 0.2]), [0.5, -0.3], np.log(1e-2)]
    f0, g = mo26_objective_series(t, p, J)
    fd = np.empty_like(t); h = 1e-6
    for i in range(len(t)):
        tp, tm = t.copy(), t.copy(); tp[i] += h; tm[i] -= h
        fd[i] = (mo26_objective_series(tp, p, J)[0] - mo26_objective_series(tm, p, J)[0]) / (2 * h)
    out['objective_gradient_max_rel_err'] = float(np.max(np.abs(g - fd)) / np.max(np.abs(fd)))
    return out


def tasks():
    return PD.tasks('quick')


def run_one(task):
    warnings.simplefilter('ignore')
    try:
        if task[0] == 'sim':
            band, T, *_ = RD.sim_band(task[1], task[2])
            key = dict(kind='sim', cell=task[1], seed=task[2])
        else:
            _, spk, fc, M, d, seed = task
            band, T, _, _ = RD.meas_band(spk, fc, M, seed)
            key = dict(kind='meas', source=spk, band=fc, M=M, draw=d, seed=seed)
        kmax = 2 * np.pi * float(band.freqs.max()) / band.c
        row = dict(key, M_mics=band.M, k_max=kmax, nmse={}, radii={}, k_max_R={}, seconds={})
        models = v2.fit_band(band, 5)
        kcv, _ = v2.cv_order(band, 5, E.make_folds(band.M, 5, task[-1]))
        row['K_cv'] = int(kcv)
        for tag, ctx in (('cubature', contextlib.nullcontext), ('series', exact_kernel)):
            with ctx():
                a = time.process_time()
                if kcv > 0:
                    ext = v2.extend(band, models[kcv], kcv)
                    pj = [E.nmse(v2.predict_regions(band, ext, f), T[:, f]) for f in range(band.F)]
                    rj = [float(r) for r in ext['radii']]
                else:
                    pj = [E.nmse(v2.predict_points(band, models[0], f), T[:, f]) for f in range(band.F)]
                    rj = []
                mo = [v2.mo26_cap(p) for p in band.probs]
                gs = v2.gsmf_fit(band)
                row['nmse'][tag] = dict(j_cv_ext=pj, mo26_cap5=[E.nmse(r['pred'], T[:, f]) for f, r in enumerate(mo)],
                                        gsmf=[E.nmse(v2.predict_regions(band, gs, f), T[:, f]) for f in range(band.F)])
                row['radii'][tag] = dict(j_cv_ext=rj, mo26_cap5=[r['radii'] for r in mo], gsmf=[float(r) for r in gs['radii']])
                allr = rj + [x for r in mo for x in r['radii']] + [float(r) for r in gs['radii']]
                row['k_max_R'][tag] = max(allr) * kmax if allr else 0.0
                row['seconds'][tag] = round(time.process_time() - a, 1)
        return row
    except Exception:
        return dict(task=list(task), error=traceback.format_exc())


def init():
    warnings.simplefilter('ignore')
    RD.init()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.check:
        RD.init()
        res = checks()
        (out / 'checks.json').write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps(res, indent=1, default=float))
        return
    T = tasks()
    t0 = time.time()
    with Pool(a.workers, initializer=init) as pool, open(out / 'rows.jsonl', 'w') as fh:
        for n, row in enumerate(pool.imap_unordered(run_one, T), 1):
            fh.write(json.dumps(row, default=float) + '\n'); fh.flush()
            print(f'{n}/{len(T)} {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
