"""WP1 post hoc diagnostic of the spherical-array breakdown (PLAN.md). K = 1 only; no registered file is changed.

    OPENBLAS_NUM_THREADS=1 python3 -B posthoc/upgrade-wp1/wp1.py OUT.jsonl --workers 10
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
A = HERE.parents[1]
sys.path.insert(0, str(A / 'common')); sys.path.insert(0, str(A / 'posthoc' / 'review3'))
import estimators as E  # noqa: E402
import v2  # noqa: E402
import reproduce as RP  # noqa: E402

AT_BOUND = v2.LOGW2[0] + 1e-6


def make_band(seed):
    src, q = RP.trial(seed)
    Y, T = zip(*[RP.observe(RP.X_MIC, src, q, f, seed) for f in RP.FREQS])
    return v2.Band(RP.X_MIC, np.stack(Y, 1), q, RP.FREQS, c=RP.C0, box=RP.BOX), T, src


def errs_db(band, m, T):
    return [float(10 * np.log10(E.nmse(v2.predict_points(band, m, i), T[i]))) for i in range(band.F)]


def refine(band, th0, K=1, maxiter=200, fixed_S=None):
    lo, hi = band.search_box()
    bounds = v2._bounds_points(band, K, lo, hi)
    lb, ub = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    th0 = np.clip(th0, lb, ub)
    if fixed_S is None:
        r = minimize(v2.joint_objective, th0, args=(band, K), jac=True, method='L-BFGS-B', bounds=bounds,
                     options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-6))
        return r, r.x
    n = 3 * K

    def obj(t):
        v, g = v2.joint_objective(np.concatenate((fixed_S.ravel(), t)), band, K)
        return v, g[n:]
    r = minimize(obj, th0[n:], jac=True, method='L-BFGS-B', bounds=bounds[n:],
                 options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-6))
    return r, np.concatenate((fixed_S.ravel(), r.x))


def model_of(band, th, K=1):
    S, lw, lr = v2._split(th, K, band.F)
    return dict(S=S.reshape(K, 3), W=np.exp(lw), R=np.exp(lr))


def closed_form_weights(band, S, R):
    """Per-bin optimal scan weight of one point at S on top of the source-free model (band_scan at one node)."""
    out = v2.band_scan(band, np.atleast_2d(S), np.zeros((0, 3)), np.zeros((band.F, 0)), R, 1, 0.0)
    return out[0][1] if out else np.full(band.F, np.exp(v2.LOGW2[0]))


def one(seed):
    t0 = time.process_time()
    try:
        band, T, src = make_band(seed)
        lo, hi = band.search_box()
        lam_mid = band.lam_mid()
        nodes = E.grid_nodes(lo, hi, 0.25 * lam_mid, max_nodes=v2.SCAN_NODES)
        ax = np.unique(np.round(nodes[:, 0], 9))
        spacing = float(ax[1] - ax[0])
        R0 = np.array([v2.remainder_rho(p) for p in band.probs])
        cands = v2.band_scan(band, nodes, np.zeros((0, 3)), np.zeros((band.F, 0)), R0, 3, 0.5 * lam_mid)
        # baseline replication of fit_band at K = 1
        runs = []
        for node, wn, _ in cands:
            th0 = np.concatenate((node, np.log(np.maximum(wn, 1e-4)), np.log(R0)))
            r, th = refine(band, th0)
            runs.append((r, th, th0, node))
        r, th, th0, node = min(runs, key=lambda z: z[0].fun)
        ref = v2.fit_band(band, 1)[1]
        rep_ok = bool(np.allclose(ref['S'].ravel(), th[:3], atol=1e-9) and np.allclose(np.log(ref['W'][:, 0]), th[3:3 + band.F], atol=1e-9))
        base = model_of(band, th)
        out = dict(seed=seed, src=src.tolist(), spacing_m=spacing, replicated=rep_ok,
                   best_node_dist_m=float(np.linalg.norm(node - src)),
                   init_log10w=(th0[3:3 + band.F] / np.log(10)).tolist(),
                   init_at_bound=(th0[3:3 + band.F] <= AT_BOUND).tolist(),
                   base=dict(dist_m=float(np.linalg.norm(base['S'][0] - src)), log10w=np.log10(base['W'][:, 0]).tolist(),
                             nit=int(r.nit), msg=str(r.message), fun=float(r.fun), db=errs_db(band, base, T)))
        # E-fix (a): true position, baseline initial weights
        ra, tha = refine(band, np.concatenate((src, th0[3:])), fixed_S=src[None, :])
        ma = model_of(band, tha)
        out['fix_a'] = dict(log10w=np.log10(ma['W'][:, 0]).tolist(), fun=float(ra.fun), nit=int(ra.nit), db=errs_db(band, ma, T))
        # E-fix (b): true position, closed-form weights at the source
        wb = closed_form_weights(band, src, R0)
        rb, thb = refine(band, np.concatenate((src, np.log(np.maximum(wb, 1e-4)), np.log(R0))), fixed_S=src[None, :])
        mb = model_of(band, thb)
        out['fix_b'] = dict(init_log10w=np.log10(np.maximum(wb, 1e-4)).tolist(), log10w=np.log10(mb['W'][:, 0]).tolist(),
                            fun=float(rb.fun), nit=int(rb.nit), db=errs_db(band, mb, T))
        # E-reset: closed-form weights at the fitted coordinate, refine again
        wr = closed_form_weights(band, base['S'][0], R0)
        rr, thr = refine(band, np.concatenate((base['S'][0], np.log(np.maximum(wr, 1e-4)), np.log(R0))))
        mr = model_of(band, thr)
        out['reset'] = dict(dist_m=float(np.linalg.norm(mr['S'][0] - src)), log10w=np.log10(mr['W'][:, 0]).tolist(),
                            fun=float(rr.fun), nit=int(rr.nit), db=errs_db(band, mr, T))
        # E-zoom: fine rescan around each coarse candidate, then the baseline refinement
        lam_min = band.c / band.freqs.max()
        zruns = []
        for node, _, _ in cands:
            zn = E.grid_nodes(node - spacing, node + spacing, lam_min / 6, max_nodes=30000)
            zc = v2.band_scan(band, zn, np.zeros((0, 3)), np.zeros((band.F, 0)), R0, 1, 0.0)
            zn0, zw, _ = zc[0]
            rz, thz = refine(band, np.concatenate((zn0, np.log(np.maximum(zw, 1e-4)), np.log(R0))))
            zruns.append((rz, thz, zn0, zw))
        rz, thz, zn0, zw = min(zruns, key=lambda z: z[0].fun)
        mz = model_of(band, thz)
        out['zoom'] = dict(node_dist_m=float(np.linalg.norm(zn0 - src)), init_at_bound=(np.log(np.maximum(zw, 1e-4)) <= AT_BOUND).tolist(),
                           dist_m=float(np.linalg.norm(mz['S'][0] - src)), log10w=np.log10(mz['W'][:, 0]).tolist(),
                           fun=float(rz.fun), nit=int(rz.nit), db=errs_db(band, mz, T))
        out['seconds'] = round(time.process_time() - t0, 1)
        return out
    except Exception:
        return dict(seed=seed, error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--seeds', type=int, nargs='*', default=None)
    a = ap.parse_args()
    seeds = a.seeds or list(RP.SEEDS)
    with Pool(a.workers, initializer=RP.init) as pool, open(a.out, 'w') as fh:
        for row in pool.imap_unordered(one, seeds):
            fh.write(json.dumps(row) + '\n'); fh.flush()
            print(row['seed'], 'error' if 'error' in row else row['seconds'], flush=True)


if __name__ == '__main__':
    main()
