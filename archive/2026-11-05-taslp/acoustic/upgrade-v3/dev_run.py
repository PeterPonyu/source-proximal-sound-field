"""Version 3 development check (PLAN.md). Development numbers only, never paper numbers.

Tasks: the spherical post hoc configuration (20 trials, shared band, full pipeline), development simulation cells on
seeds 400920-400927 (120 problems), measured development bins S4/S7/S8 draw 0 (48 problems). Per task: version 3
primary estimator (per-bin NMSE, order, evidence per order, convergence records, process time) and the version-2 full-data
greedy fit replicated with convergence records (evidence per order).

    OPENBLAS_NUM_THREADS=1 nice -n 10 python3 -B upgrade-v3/dev_run.py OUT.jsonl --workers 18
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

SIM_SEEDS = range(400920, 400928)


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
                if d == 0:
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
        t0 = time.process_time()
        pred, models, info = v3.primary(band, seed)
        secs = time.process_time() - t0
        nmse = [E.nmse(pred(f), T[:, f]) for f in range(band.F)]
        st2 = []
        m2 = v3.fit_band(band, 5, opts=v3.V2_OPTS, stats=st2)
        return dict(task=list(task), nmse_v3=nmse, K_v3=info['K'], evidence_v3={str(k): v for k, v in info['evidence'].items()},
                    evidence_v2={str(k): m['evidence'] for k, m in m2.items()}, point_stats_v3=info['point_stats'],
                    region_stats_v3=info.get('region_stats', []), point_stats_v2=st2, seconds_v3=round(secs, 1))
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
