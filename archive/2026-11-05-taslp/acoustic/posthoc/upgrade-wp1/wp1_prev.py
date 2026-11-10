"""WP1 prevalence probe (post hoc, PLAN.md addendum): does the bound-start / iteration-cap defect of the greedy shared fit
occur on registered room-grid problems, and what does a closed-form weight reset change?

Sample: MeshRIR room grid, every band x microphone count, loudspeakers 0, 8, 16, 24, draw 0 (64 problems), the
registered task seeds. For each order K = 1..5 of the registered greedy fit (replicated from v2.fit_band) record the
iterations, whether the cap was hit and how many bins of the new point started at the weight bound; then the reset
variant: after each order's refinement, the newest point's per-bin weights are set to their closed-form optimum given
the other fitted points, and the refinement is repeated once (same cap). Point-model errors per order for both.

    OPENBLAS_NUM_THREADS=1 python3 -B posthoc/upgrade-wp1/wp1_prev.py OUT.jsonl --workers 10
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
sys.path.insert(0, str(A / 'common')); sys.path.insert(0, str(A))
import estimators as E  # noqa: E402
import v2  # noqa: E402
import problems_v2 as PB  # noqa: E402
import run_v2 as RV  # noqa: E402

AT_BOUND = v2.LOGW2[0] + 1e-6
MAXITER = 200


def refine(band, th0, K):
    lo, hi = band.search_box()
    bounds = v2._bounds_points(band, K, lo, hi)
    lb, ub = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    return minimize(v2.joint_objective, np.clip(th0, lb, ub), args=(band, K), jac=True, method='L-BFGS-B', bounds=bounds,
                    options=dict(maxiter=MAXITER, ftol=1e-10, gtol=1e-6))


def greedy(band, reset):
    lo, hi = band.search_box()
    nodes = E.grid_nodes(lo, hi, 0.25 * band.lam_mid(), max_nodes=v2.SCAN_NODES)
    R0 = np.array([v2.remainder_rho(p) for p in band.probs])
    S, W, R = np.zeros((0, 3)), np.zeros((band.F, 0)), R0
    models, info = {}, {}
    for K in range(1, 6):
        cands = v2.band_scan(band, nodes, S, W, R, 3, 0.5 * band.lam_mid())
        best, bstart = None, None
        for node, wn, _ in cands:
            lw0 = np.log(np.maximum(np.hstack((W, wn[:, None])), 1e-4))
            th0 = np.concatenate((np.vstack((S, node[None, :])).ravel(), lw0.ravel(), np.log(R)))
            r = refine(band, th0, K)
            if best is None or r.fun < best.fun:
                best, bstart = r, int(np.sum(np.log(np.maximum(wn, 1e-4)) <= AT_BOUND))
        rec = dict(nit=int(best.nit), cap=bool(best.nit >= MAXITER), new_at_bound=bstart, fun=float(best.fun))
        if reset:
            Sv, lw, lr = v2._split(best.x, K, band.F)
            Sv = Sv.reshape(K, 3)
            out = v2.band_scan(band, Sv[-1:], Sv[:-1], np.exp(lw[:, :-1]), np.exp(lr), 1, 0.0)
            if out:
                lw = lw.copy(); lw[:, -1] = np.log(np.maximum(out[0][1], 1e-4))
                r2 = refine(band, np.concatenate((Sv.ravel(), lw.ravel(), lr)), K)
                rec.update(nit2=int(r2.nit), cap2=bool(r2.nit >= MAXITER), fun2=float(r2.fun))
                if r2.fun < best.fun:
                    best = r2
        Sv, lw, lr = v2._split(best.x, K, band.F)
        S, W, R = Sv.reshape(K, 3), np.exp(lw), np.exp(lr)
        models[K] = dict(S=S.copy(), W=W.copy(), R=R.copy())
        info[K] = rec
    return models, info


def one(task):
    t0 = time.process_time()
    try:
        _, spk, fc, M, d, seed = task
        band, T, ref, ts = PB.meshrir_band(RV.DATA['mesh'], spk, fc, M, seed)
        res = dict(task=list(task))
        for tag, reset in (('base', False), ('reset', True)):
            models, info = greedy(band, reset)
            res[tag] = {K: dict(info[K], db=[float(10 * np.log10(E.nmse(v2.predict_points(band, m, f), T[:, f])))
                                             for f in range(band.F)]) for K, m in models.items()}
        res['seconds'] = round(time.process_time() - t0, 1)
        return res
    except Exception:
        return dict(task=list(task), error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--workers', type=int, default=10)
    a = ap.parse_args()
    T = [t for t in RV.tasks('TB-M2') if t[0] == 'mesh' and t[1] in (0, 8, 16, 24) and t[4] == 0]
    assert len(T) == 64, len(T)
    with Pool(a.workers, initializer=RV.init, initargs=('TB-M2',)) as pool, open(a.out, 'w') as fh:
        for n, row in enumerate(pool.imap_unordered(one, T), 1):
            fh.write(json.dumps(row) + '\n'); fh.flush()
            print(n, row.get('seconds', 'error'), flush=True)


if __name__ == '__main__':
    main()
