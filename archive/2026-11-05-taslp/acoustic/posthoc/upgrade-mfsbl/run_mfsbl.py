"""Post hoc runs of the multi-frequency SBL comparator on registered problems (PLAN.md); A2 runs inside TB-M4 instead.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python3 -B run_mfsbl.py OUT --workers N
    [--dev | --sets A1,R2,room,sim] [--extract DIR]
--dev: the 8 room-grid and 8 MIRACLE A1 problems fixed in PLAN.md (development numbers, not paper numbers).
--sets: every problem of the stored registered rows of those sets (TB-M3 A1/R2, TB-M2 room grid, TB-S2 simulation).
Each problem is rebuilt with the registered builder and seed. A row holds the comparator's per-bin NMSE, its info and
process seconds, and the stored errors (dB) of the primary and the single-frequency SBL of the same problem.
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
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(S)); sys.path.insert(0, str(S / 'studies' / 'TB-M3-MIRACLE'))
import mfsbl as MF  # noqa: E402
import run_m3 as RM  # noqa: E402
import problems_v2 as PB  # noqa: E402
import analyze_v2 as AV  # noqa: E402

ROWS = dict(miracle=S / 'studies/TB-M3-MIRACLE/primary/rows.jsonl', room=S / 'studies/TB-M2/primary/rows.jsonl',
            roomfix=S / 'studies/TB-M2-SBLFIX/primary/rows.jsonl', sim=S / 'studies/TB-S2/primary/rows.jsonl')
STORED = ('j_cv_ext', 'sbl_em_common', 'sbl_mackay_common', 'sbl_em_guard', 'sbl_mackay_guard', 'sinc_ev')
MESH = {}


def key(set_, r):
    return (set_, r['cell'], r['seed']) if set_ == 'sim' else (set_, r['source'], float(r['band']), r['M'], r['draw'], r['seed'])


def stored(sets):
    """Problem key -> stored errors (dB) of the registered rows."""
    out = {}
    for s in sets:
        if s in ('A1', 'R2'):
            src = [(r, s) for r in map(json.loads, open(ROWS['miracle'])) if r.get('scenario') == s]
        elif s == 'room':
            src = [(r, s) for r in map(json.loads, open(ROWS['room'])) if r.get('kind') == 'mesh'] + \
                  [(r, s) for r in map(json.loads, open(ROWS['roomfix']))]
        else:
            src = [(r, s) for r in map(json.loads, open(ROWS['sim'])) if r.get('kind') == 'sim']
        for r, s_ in src:
            if 'error' in r:
                continue
            e = out.setdefault(key(s_, r), {})
            e.update({m: AV.db(v) for m, v in r['nmse'].items() if m in STORED})
    return out


def dev_keys(st):
    srcs = np.sort(np.random.default_rng(414600).choice(64, 32, replace=False))
    first = {'A1': int(srcs[0]), 'room': 0}
    return [k for k in st if k[0] in first and k[1] == first[k[0]] and k[2] in (250.0, 1000.0) and k[4] == 0]


def init(extract, sets):
    mir = [s for s in sets if s in ('A1', 'R2')]
    if mir:
        RM.init(extract, mir)
    if 'room' in sets:
        z = np.load(S / 'B3-measured/meshrir-prep/meshrir-prep.npz', allow_pickle=False)
        MESH.update({k: z[k] for k in z.files})


def run(job):
    k, ref = job
    t0 = time.process_time()
    try:
        if k[0] == 'sim':
            band, T, _, _ = PB.sim_band(k[1], k[2])
        elif k[0] == 'room':
            band, T, _, _ = PB.meshrir_band(MESH, k[1], k[2], k[3], k[5])
        else:
            band, T, _, _ = RM.build(k[0], k[1], k[2], k[3], k[5])
        out, info, timing = MF.add_methods(band, T, {}, {}, {})
        return dict(key=list(k), post_hoc=True, F=band.F, M_mics=band.M, nmse=out, info=info, timing=timing,
                    e={m: AV.db(v) for m, v in out.items()}, stored_e=ref, process_seconds=round(time.process_time() - t0, 1))
    except Exception:
        return dict(key=list(k), error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--workers', type=int, required=True)
    ap.add_argument('--dev', action='store_true'); ap.add_argument('--sets', default='A1,R2,room,sim')
    ap.add_argument('--extract', default=str(S / 'studies/TB-M3-MIRACLE/extract'))
    a = ap.parse_args()
    sets = ['A1', 'room'] if a.dev else a.sets.split(',')
    st = stored(sets)
    keys = dev_keys(st) if a.dev else sorted(st, key=str)
    keys = sorted(keys, key=lambda k: -k[3] if k[0] != 'sim' else 0)          # largest problems first
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    (out / 'started.json').write_text(json.dumps(dict(tasks=len(keys), sets=sets, dev=a.dev, workers=a.workers, post_hoc=True,
                                                      started=time.strftime('%Y-%m-%dT%H:%M:%S%z')), indent=1))
    t0 = time.time()
    with Pool(a.workers, initializer=init, initargs=(a.extract, sets)) as pool, open(out / 'rows.jsonl', 'w') as fh:
        for n, row in enumerate(pool.imap_unordered(run, [(k, st[k]) for k in keys]), 1):
            fh.write(json.dumps(row, default=float) + '\n'); fh.flush()
            print(f'{n}/{len(keys)} {time.time() - t0:.0f} s', row.get('key'), 'ERROR' if 'error' in row else
                  {m: round(v, 2) for m, v in row['e'].items()}, flush=True)


if __name__ == '__main__':
    main()
