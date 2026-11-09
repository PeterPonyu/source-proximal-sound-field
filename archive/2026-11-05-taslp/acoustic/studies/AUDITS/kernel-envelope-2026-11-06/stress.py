"""Stress case for the failing corner, ADDED AFTER the development effect estimate (effect.json) was read.

Not in PLAN.md. No development fit had a ball in the corner where the check failed (R >= 0.4 m at the top wavenumbers
with targets in or near the ball), so effect.json could not measure the effect there. This script forces one ball
into the corner on development problems. It takes the five development problems of the 1 kHz band, fitted with
the custom group-sparse kernel and the exact kernel as registered, and enlarges their largest ball to R = 0.5 m. It
moves the ball to the centre that holds the most targets inside or within 2 cm of it while every microphone stays
at least 5 cm outside it, and keeps every other fitted parameter. It then compares the errors with the registered
kernel blocks and with refined ones: order 80 in 50-digit arithmetic, and the reference quadrature for the targets
in or near the ball. The fitted state is perturbed, so the errors themselves mean nothing; only their difference
between the two kernel evaluations is the quantity of interest.

    OPENBLAS_NUM_THREADS=1 python3 -B studies/AUDITS/kernel-envelope-2026-11-06/stress.py --workers 4
"""
from __future__ import annotations
import argparse
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
S = HERE.parents[2]
sys.path.insert(0, str(S / 'common')); sys.path.insert(0, str(S / 'V2-dev')); sys.path.insert(0, str(HERE))
import cubature_diag as CD  # noqa: E402
import effect as EF  # noqa: E402
import estimators as E  # noqa: E402
import prox_dev as PD  # noqa: E402
import run_dev as RD  # noqa: E402
import v2  # noqa: E402

R_STRESS = 0.5


def place(x, q, R, spacing=0.05):
    """The centre holding the most targets within R + 0.02 m, with every microphone at least R + 0.05 m away."""
    pts = np.vstack((x, q))
    lo, hi = pts.min(0) - R, pts.max(0) + R
    g = np.stack(np.meshgrid(*[np.arange(a, b + 1e-9, spacing) for a, b in zip(lo, hi)], indexing='ij'), -1).reshape(-1, 3)
    ok = np.min(np.linalg.norm(g[:, None, :] - x[None, :, :], axis=2), axis=1) >= R + 0.05
    g = g[ok]
    n_in = np.array([int(np.sum(np.linalg.norm(q - c, axis=1) <= R + 0.02)) for c in g])
    i = int(np.argmax(n_in))
    return g[i], int(n_in[i])


def run_one(task):
    warnings.simplefilter('ignore')
    try:
        if task[0] == 'sim':
            band, T, *_ = RD.sim_band(task[1], task[2]); key = dict(kind='sim', cell=task[1], seed=task[2])
        else:
            _, spk, fc, M, d, seed = task
            band, T, _, _ = RD.meas_band(spk, fc, M, seed); key = dict(kind='meas', source=spk, band=fc, M=M, seed=seed)
        with CD.exact_kernel():
            gs = v2.gsmf_fit(band)
        cen = np.array(gs['centres'], float); rad = np.array(gs['radii'], float)
        j = int(np.argmax(rad))
        c_new, n_in = place(band.x, band.q, R_STRESS)
        cen[j] = c_new; rad[j] = R_STRESS
        reg, ref = [], []
        for f, p in enumerate(band.probs):
            p0, _, _ = EF.predict(p, cen, rad, gs['lw'][f], float(gs['R'][f]), refine=False)
            p1, _, _ = EF.predict(p, cen, rad, gs['lw'][f], float(gs['R'][f]), refine=True)
            reg.append(E.nmse(p0, T[:, f])); ref.append(E.nmse(p1, T[:, f]))
        return dict(key, targets=int(len(band.q)), targets_in_or_near=n_in, k_max=float(2 * np.pi * band.freqs.max() / band.c),
                    db_registered=float(10 * np.log10(np.mean(reg))), db_refined=float(10 * np.log10(np.mean(ref))),
                    change_db=float(10 * np.log10(np.mean(ref)) - 10 * np.log10(np.mean(reg))),
                    max_bin_change_db=float(np.max(np.abs(10 * np.log10(np.array(ref) / np.array(reg))))))
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
    tasks = [t for t in PD.tasks('quick') if (t[0] == 'sim' and RD.sim.cells()[t[1]]['f'] == 1000.0) or (t[0] == 'meas' and t[2] == 1000.0)]
    with Pool(a.workers, initializer=init) as pool:
        rows = pool.map(run_one, tasks, chunksize=1)
    ok = [r for r in rows if 'error' not in r]
    summary = dict(problems=len(tasks), errors=len(rows) - len(ok), radius_m=R_STRESS,
                   max_abs_change_db=float(max(abs(r['change_db']) for r in ok)) if ok else None,
                   max_abs_bin_change_db=float(max(r['max_bin_change_db'] for r in ok)) if ok else None,
                   added_after_reading='effect.json')
    (HERE / 'stress.json').write_text(json.dumps(dict(summary=summary, rows=rows), indent=1) + '\n')
    print(json.dumps(summary, indent=1))
    for r in rows:
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k != 'error'} if 'error' not in r else r['error'][-500:])


if __name__ == '__main__':
    main()
