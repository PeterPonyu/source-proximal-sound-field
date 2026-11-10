"""Run time of the arrival-time point baseline and a single fitted point (PLAN.md). Post hoc; no registered output
changes.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
        python3 -B posthoc/upgrade-timing/timing.py posthoc/upgrade-timing/timing.jsonl --workers 4
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
S = HERE.parents[1]
sys.path.insert(0, str(S / 'common')); sys.path.insert(0, str(S / 'V2-dev'))
import estimators as E  # noqa: E402
import problems_v2 as PB  # noqa: E402
import v2  # noqa: E402

DATA = {}
GATE = 1e-6


def init():
    z = np.load(S / 'B3-measured/meshrir-prep/meshrir-prep.npz', allow_pickle=False)
    DATA['mesh'] = {k: z[k] for k in z.files}


def stored():
    out = {}
    for l in open(S / 'studies/TB-M2/primary/rows.jsonl'):
        r = json.loads(l)
        if 'error' in r or r.get('kind') != 'mesh':
            continue
        out[(r['source'], r['band'], r['M'], r['draw'], r['seed'])] = dict(
            j_K1=r['nmse']['j_K1'], at=r['nmse'].get('two_stage_point'), j_fit=r['timing']['j_fit'])
    return out


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def one(task):
    key, st = task
    spk, fc, M, draw, seed = key
    try:
        prep = DATA['mesh']
        band, T, _, _ = PB.meshrir_band(prep, spk, fc, M, seed)
        # the builder's microphone draw, repeated to recover the training positions and arrival times
        rng = np.random.default_rng(np.random.SeedSequence([seed, 9311]))
        pos = prep['pos_mic']; mic = np.sort(rng.choice(len(pos), M, replace=False))
        assert np.array_equal(pos[mic], band.x)
        box = (pos.min(0), pos.max(0))
        a = time.process_time()
        ts = PB.two_stage_centre(pos[mic], prep['arrival_s'][spk][mic], box)
        at = None
        if ts:
            c2 = np.asarray(ts['centre'])[None, :]
            at = [E.nmse(E.oracle_fit(band.probs[f], c2)['pred'], T[:, f]) for f in range(band.F)]
        t_at = time.process_time() - a
        a = time.process_time()
        m1 = v2.fit_band(band, 1)
        k1 = [E.nmse(v2.predict_points(band, m1[1], f), T[:, f]) for f in range(band.F)]
        t_k1 = time.process_time() - a
        a = time.process_time()
        v2.fit_band(band, 5)
        t_f5 = time.process_time() - a
        return dict(key=list(key), M=M, at_seconds=t_at, k1_seconds=t_k1, jfit5_seconds=t_f5, jfit5_recorded=st['j_fit'],
                    gate_k1=rel(k1, st['j_K1']), gate_at=None if at is None or st['at'] is None else rel(at, st['at']),
                    at_ok=bool(ts))
    except Exception:
        return dict(key=list(key), error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    st = stored()
    keys = [tuple(json.loads(l)['task'][1:]) for l in open(S / 'posthoc/upgrade-wp1/wp1-prev-roomgrid.jsonl')]
    tasks = [(k, st[k]) for k in keys]
    with Pool(a.workers, initializer=init) as pool, open(a.out, 'w') as fh:
        for r in pool.imap_unordered(one, tasks):
            fh.write(json.dumps(r) + '\n'); fh.flush()
    rows = [json.loads(l) for l in open(a.out)]
    bad = [r for r in rows if 'error' in r or r['gate_k1'] > GATE or (r['gate_at'] is not None and r['gate_at'] > GATE)]
    print(len(rows), 'rows;', len(bad), 'gate failures or errors')


if __name__ == '__main__':
    main()
