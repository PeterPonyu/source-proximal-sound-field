"""Outcome rule 3 of PLAN.md: the effect of the failing kernel blocks, on development problems only.

The 18 development problems of the exact-kernel diagnostic (V2-dev, at most 48 microphones) are fitted with both
custom region kernels and the exact kernel, exactly as in the registered exact-kernel follow-up. Then, at those
fitted parameters and without a refit, every ball of radius 0.2 m or more gets refined blocks: the microphone and
far-target series at order 80 in 50-digit arithmetic, and the targets inside or within 2 cm of the ball by the
target-centred reference quadrature. This covers the failing corner of the check (R >= 0.4 m) with margin. Reported:
the change of each problem's error, 10 log10 of the mean NMSE over the bins, refined minus registered.

    OPENBLAS_NUM_THREADS=1 python3 -B studies/AUDITS/kernel-envelope-2026-11-06/effect.py --workers 4
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
S = HERE.parents[2]
sys.path.insert(0, str(S / 'common')); sys.path.insert(0, str(S / 'V2-dev')); sys.path.insert(0, str(HERE))
import check_kernel as CK  # noqa: E402
import cubature_diag as CD  # noqa: E402
import estimators as E  # noqa: E402
import prox_dev as PD  # noqa: E402
import run_dev as RD  # noqa: E402
import v2  # noqa: E402

REFINE_R = 0.2


def refined_blocks(prob, c, R):
    x = prob.x - c; q = prob.q - c
    rx = np.linalg.norm(x, axis=1); rq = np.linalg.norm(q, axis=1)
    far = rq > R + 0.02
    V, sgn = CK.mp_vectors(list(rx) + list(rq[far]), prob.k, R)
    Vx, Vq = V[:, :len(rx)], V[:, len(rx):]
    dx = x / rx[:, None]
    K = CK.series_orders(Vx, Vx, sgn, np.clip(dx @ dx.T, -1, 1), orders=(80,))[80]
    Kq = np.zeros((len(q), len(x)), complex)
    if far.any():
        dq = q[far] / rq[far, None]
        Kq[far] = CK.series_orders(Vq, Vx, sgn, np.clip(dq @ dx.T, -1, 1), orders=(80,))[80]
    for i in np.where(~far)[0]:
        Kq[i] = CK.ref_row(q[i], x, R, prob.k, 64, 64, 128)[0]
    return K, Kq, int((~far).sum())


def predict(prob, cen, rad, lw, rho, refine):
    C = prob.H + (rho + E.JITTER) * np.eye(prob.M)
    Cq = prob.Hq.astype(complex)
    n_ref = n_near = 0
    for j in range(len(cen)):
        c = np.asarray(cen[j], float); R = float(rad[j])
        if refine and R >= REFINE_R:
            K, Kq, nn = refined_blocks(prob, c, R); n_ref += 1; n_near += nn
        else:
            K = CD.series_K(prob.x, c, R, prob.k); Kq = CD.cross_K(prob.q, prob.x, c, R, prob.k)
        s = np.exp(lw[j]) / float(np.mean(np.real(np.diag(K))))
        C = C + s * K; Cq = Cq + s * Kq
    return Cq @ np.linalg.solve(C, prob.y), n_ref, n_near


def run_one(task):
    warnings.simplefilter('ignore')
    try:
        if task[0] == 'sim':
            band, T, *_ = RD.sim_band(task[1], task[2]); key = dict(kind='sim', cell=task[1], seed=task[2])
        else:
            _, spk, fc, M, d, seed = task
            band, T, _, _ = RD.meas_band(spk, fc, M, seed); key = dict(kind='meas', source=spk, band=fc, M=M, seed=seed)
        calls = []

        def capture(prob, cen, rad, lw, rho):
            calls.append((prob, np.asarray(cen, float), np.asarray(rad, float), np.asarray(lw, float), float(rho)))
            return CD.region_predict_series(prob, cen, rad, lw, rho)
        out = dict(key, M_mics=band.M, k_max=float(2 * np.pi * band.freqs.max() / band.c), methods={})
        with CD.exact_kernel():
            E.region_predict = capture
            gs = v2.gsmf_fit(band)
            gcalls = []
            for f in range(band.F):
                calls.clear(); v2.predict_regions(band, gs, f); gcalls.append(calls[-1])
            calls.clear()
            mo = [v2.mo26_cap(p) for p in band.probs]
            mcalls = list(calls)
        assert len(mcalls) == band.F
        for name, cl in (('gsmf', gcalls), ('mo26_cap5', mcalls)):
            reg, ref, nref, nnear, rmax = [], [], 0, 0, 0.0
            for f, (prob, cen, rad, lw, rho) in enumerate(cl):
                p0, _, _ = predict(prob, cen, rad, lw, rho, refine=False)
                p1, a, b = predict(prob, cen, rad, lw, rho, refine=True)
                reg.append(E.nmse(p0, T[:, f])); ref.append(E.nmse(p1, T[:, f])); nref += a; nnear += b
                rmax = max(rmax, float(rad.max()) if len(rad) else 0.0)
            out['methods'][name] = dict(db_registered=float(10 * np.log10(np.mean(reg))), db_refined=float(10 * np.log10(np.mean(ref))),
                                        change_db=float(10 * np.log10(np.mean(ref)) - 10 * np.log10(np.mean(reg))),
                                        balls_refined=nref, near_targets_refined=nnear, max_radius=rmax,
                                        max_kR=rmax * out['k_max'])
        return out
    except Exception:
        import traceback
        return dict(task=list(task), error=traceback.format_exc())


def init():
    warnings.simplefilter('ignore')
    RD.init()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    tasks = PD.tasks('quick')
    with Pool(a.workers, initializer=init) as pool:
        rows = pool.map(run_one, tasks, chunksize=1)
    ok = [r for r in rows if 'error' not in r]
    ch = [m['change_db'] for r in ok for m in r['methods'].values()]
    summary = dict(problems=len(tasks), errors=len(rows) - len(ok), refine_radius_m=REFINE_R,
                   balls_refined=sum(m['balls_refined'] for r in ok for m in r['methods'].values()),
                   max_abs_change_db=float(np.max(np.abs(ch))) if ch else None,
                   max_radius_m=max((m['max_radius'] for r in ok for m in r['methods'].values()), default=None),
                   seconds=round(time.time() - t0, 1))
    (HERE / 'effect.json').write_text(json.dumps(dict(summary=summary, rows=rows), indent=1) + '\n')
    print(json.dumps(summary, indent=1))
    for r in rows:
        if 'error' in r:
            print('ERROR', r['task'], r['error'][-400:])


if __name__ == '__main__':
    main()
