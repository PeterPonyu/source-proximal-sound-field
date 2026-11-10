"""Version 3b development check on fresh data (PLAN-3b.md); version 2 run alongside on the same machine.
Adapted from dev_run.py (version 3 check). Development numbers only, never paper numbers.

Tasks: the spherical post hoc configuration (20 trials, shared band, full pipeline), development simulation cells on
seeds 400920-400927 (120 problems), measured development bins S4/S7/S8 draw 0 (48 problems). Per task: version 3
primary estimator (per-bin NMSE, order, evidence per order, convergence records, process time) and the version-2 full-data
greedy fit replicated with convergence records (evidence per order).

    OPENBLAS_NUM_THREADS=1 nice -n 10 python3 -B upgrade-v3/dev_run_3b.py OUT.jsonl --workers 18
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

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(A / 'common')); sys.path.insert(0, str(A / 'posthoc' / 'review3'))
sys.path.insert(0, str(A / 'V2-dev'))
import v3  # noqa: E402
import v2  # noqa: E402
import estimators as E  # noqa: E402
import reproduce as RP  # noqa: E402
import run_dev as RD  # noqa: E402
import sim  # noqa: E402

SIM_SEEDS = range(410100, 410108)


def init():
    RP.init(); RD.init()


def tasks():
    T = [('sphere', s) for s in RP.SEEDS]
    T += [('sim', c, s) for c in RD.SIM_CELLS for s in SIM_SEEDS]
    i = 0
    for spk in ('S4', 'S7', 'S8'):
        for M in (24, 48, 96, 192):
            for d in range(2):
                seed = RD.MEAS_SEEDS[i]; i += 1
                if d == 1:
                    T += [('meas', spk, fc, M, d, seed) for fc in (125.0, 250.0, 500.0, 1000.0)]
    return T


def size(t):
    return t[3] if t[0] == 'meas' else (sim.cells()[t[1]]['M'] if t[0] == 'sim' else 36)


def one(task):
    try:
        if task[0] == 'sphere':
            seed = task[1]
            src, q = RP.trial(seed)
            Y, Tl = zip(*[RP.observe(RP.X_MIC, src, q, f, seed) for f in RP.FREQS])
            band = v2.Band(RP.X_MIC, np.stack(Y, 1), q, RP.FREQS, c=RP.C0, box=RP.BOX)
            T = np.stack(Tl, 1)
        elif task[0] == 'sim':
            _, cell, seed = task
            band, T, _, _, _ = RD.sim_band(cell, seed)
        else:
            _, spk, fc, M, d, seed = task
            band, T, _, _ = RD.meas_band(spk, fc, M, seed)
        res = dict(task=list(task))
        for tag, opts in (('v2', v3.V2_OPTS), ('v3b', v3.V3B_OPTS)):
            t0 = time.process_time()
            pred, models, info = v3.primary(band, seed, opts=opts)
            secs = time.process_time() - t0
            res[tag] = dict(nmse=[E.nmse(pred(f), T[:, f]) for f in range(band.F)], K=info['K'], K_argmin=info['K_argmin'],
                            cv_fold_err=info['cv_fold_err'], evidence={str(k): v for k, v in info['evidence'].items()},
                            point_stats=info['point_stats'], region_stats=info.get('region_stats', []), seconds=round(secs, 1))
        return res
    except Exception:
        return dict(task=list(task), error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--workers', type=int, default=18)
    a = ap.parse_args()
    T = sorted(tasks(), key=lambda t: -size(t))
    t0 = time.time()
    with Pool(a.workers, initializer=init) as pool, open(a.out, 'w') as fh:
        for n, row in enumerate(pool.imap_unordered(one, T), 1):
            fh.write(json.dumps(row, default=float) + '\n'); fh.flush()
            print(f'{n}/{len(T)} {time.time() - t0:.0f}s', 'ERR' if 'error' in row else '', flush=True)


if __name__ == '__main__':
    main()
