"""Post hoc diagnostic of reproduction/ (PLAN-C2-DIAGNOSTIC.md): weights, ridges and the sinc kernel's conditioning in
the bins where the proposed estimator failed. Deterministic refits of the same tasks.

    OPENBLAS_NUM_THREADS=1 nice -n 10 python3 -B posthoc/review3/reproduce_diag.py OUT.jsonl --workers 4
"""
from __future__ import annotations
import argparse
import json
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'common')); sys.path.insert(0, str(HERE))
import estimators as E  # noqa: E402
import v2  # noqa: E402
import reproduce as RP  # noqa: E402


def describe(band, m, T, f, src):
    p = band.probs[f]
    lam = np.linalg.eigvalsh(p.H)
    e = E.nmse(v2.predict_points(band, m, f) if len(m['S']) == 0 else v2.predict_points(band, m, f), T)
    w = m['W'][f] if len(m['S']) else np.zeros(0)
    d = float(np.min(np.linalg.norm(m['S'] - src, axis=1))) if len(m['S']) else None
    return dict(db=float(10 * np.log10(e)), log10w=[float(np.log10(x)) for x in w], ridge=float(m['R'][f]),
                dist_m=d, sinc_cond=float(lam[-1] / max(lam[0], 1e-300)), sinc_min_eig=float(lam[0]))


def one(task):
    try:
        kind, seed, f = task
        src, q = RP.trial(seed)
        if kind == 'bin':
            y, truth = RP.observe(RP.X_MIC, src, q, f, seed)
            band = v2.Band(RP.X_MIC, y[:, None], q, np.array([f]), c=RP.C0, box=RP.BOX)
            models = v2.fit_band(band, 5)
            K = v2.cv_order(band, 5, E.make_folds(band.M, 5, seed))[0]
            return dict(task=list(task), K=int(K), final=describe(band, models[K], truth, 0, src),
                        K1=describe(band, models[1], truth, 0, src))
        Y, T = zip(*[RP.observe(RP.X_MIC, src, q, f_, seed) for f_ in RP.FREQS])
        band = v2.Band(RP.X_MIC, np.stack(Y, 1), q, RP.FREQS, c=RP.C0, box=RP.BOX)
        models = v2.fit_band(band, 5)
        K = v2.cv_order(band, 5, E.make_folds(band.M, 5, seed))[0]
        return dict(task=list(task), K=int(K), bins=[describe(band, models[K], T[i], i, src) for i in range(len(RP.FREQS))])
    except Exception:
        return dict(task=list(task), error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    T = [('band', s, None) for s in RP.SEEDS] + [('bin', s, float(f)) for s in RP.SEEDS for f in RP.FREQS[::-1]]
    with Pool(a.workers, initializer=RP.init) as pool, open(a.out, 'w') as fh:
        for row in pool.imap_unordered(one, T):
            fh.write(json.dumps(row) + '\n'); fh.flush()


if __name__ == '__main__':
    main()
