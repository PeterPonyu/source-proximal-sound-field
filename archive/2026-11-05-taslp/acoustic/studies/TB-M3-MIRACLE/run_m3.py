"""TB-M3-MIRACLE runner (PROTOCOL.md). The registered measured method set (methods_v2.evaluate, measured=True) with the
target-guarded sparse Bayesian comparators and the clip-masked adaptive stand-in; the proposed estimator unchanged.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python3 -B run_m3.py EXTRACT_DIR OUT --workers N
    [--limit N] [--scenarios A1,R2]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
sys.path.insert(0, str(S / 'common')); sys.path.insert(0, str(S / 'posthoc' / 'review3'))
import estimators as E  # noqa: E402
import problems_v2 as PB  # noqa: E402
import methods_v2 as MV  # noqa: E402
import v2  # noqa: E402
import sbl_guard as G  # noqa: E402
import adk_fix as AF  # noqa: E402

BANDS = (250.0, 500.0, 1000.0)
MS = (24, 48, 96, 192)
SEED0 = {'A1': 414000, 'R2': 414256}
DATA = {}

# the two corrections fixed by the protocol, applied to every problem
v2.sbl_common = lambda p, rule: G.sbl_guarded(p, rule)
E.adk_objective = lambda t, prob, Ed, Yd: AF.adk_objective_masked(t, prob, Ed, Yd)


def tasks(scenarios):
    T = []
    for sc in scenarios:
        srcs = np.sort(np.random.default_rng(414600).choice(64, 32, replace=False))
        s = SEED0[sc]
        for j in srcs:
            for M in MS:
                for d in range(2):
                    T += [(sc, int(j), fc, M, d, s) for fc in BANDS]
                    s += 1
        assert s == SEED0[sc] + 256
    return T


def init(extract_dir, scenarios):
    for sc in scenarios:
        z = np.load(Path(extract_dir) / f'miracle-{sc}.npz', allow_pickle=False)
        DATA[sc] = {k: z[k] for k in z.files}


def build(sc, spk, fc, M, seed):
    prep = DATA[sc]
    inband = np.where(prep['band'] == fc)[0]
    sel = inband[1::2]                                            # odd-indexed 6 of the 12 frequencies, as registered
    rng = np.random.default_rng(np.random.SeedSequence([seed, 9311]))
    pos = prep['pos_mic']
    box = (pos.min(0), pos.max(0))
    band, T, ts = PB._measured_band(pos, prep['H'][spk], prep['arrival_s'][spk], sel, prep['freqs'], M, rng, box, float(prep['c0']))
    return band, T, prep['pos_src'][spk][None, :], ts


def run_task(task):
    t0 = time.process_time()
    try:
        sc, spk, fc, M, d, seed = task
        mk = lambda: build(sc, spk, fc, M, seed)
        band, T, ref, ts = mk()
        out, info, timing = MV.evaluate(band, T, ref, ref[0], seed, measured=True, two_stage=ts, band_factory=lambda: mk()[0])
        return dict(kind='miracle', scenario=sc, source=spk, band=fc, M=M, draw=d, seed=seed, c_used=band.c, two_stage_ok=bool(ts),
                    F=band.F, M_mics=band.M, nmse=out, info=info, timing=timing, process_seconds=round(time.process_time() - t0, 1))
    except Exception:
        return dict(task=list(task), error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('extract'); ap.add_argument('out'); ap.add_argument('--workers', type=int, required=True)
    ap.add_argument('--limit', type=int, default=None); ap.add_argument('--scenarios', default='A1,R2')
    a = ap.parse_args()
    sc = a.scenarios.split(',')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    T = tasks(sc)
    T = sorted(T, key=lambda t: -t[3])                            # largest problems first
    if a.limit:
        T = sorted(T, key=lambda t: t[3])[:a.limit]               # the cross-machine check uses the smallest problems
    commit = subprocess.run(['git', '-C', str(S), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() or None
    (out / 'started.json').write_text(json.dumps(dict(tasks=len(T), workers=a.workers, commit=commit, started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                                      machine=dict(node=platform.node(), cpu=platform.processor(), numpy=np.__version__),
                                                      extracts={s: hashlib.sha256((Path(a.extract) / f'miracle-{s}.npz').read_bytes()).hexdigest() for s in sc},
                                                      inputs={n: hashlib.sha256((S / n).read_bytes()).hexdigest() for n in
                                                              ('common/estimators.py', 'common/v2.py', 'common/methods_v2.py', 'common/problems_v2.py',
                                                               'common/sbl_guard.py', 'posthoc/review3/adk_fix.py', 'studies/TB-M3-MIRACLE/run_m3.py',
                                                               'studies/TB-M3-MIRACLE/PROTOCOL.md')}), indent=1))
    t0 = time.time(); n_err = 0
    with Pool(a.workers, initializer=init, initargs=(a.extract, sc)) as pool, open(out / 'rows.jsonl', 'w') as fh:
        for n, row in enumerate(pool.imap_unordered(run_task, T), 1):
            n_err += 'error' in row
            fh.write(json.dumps(row, default=float) + '\n'); fh.flush()
            if n % 50 == 0 or n == len(T):
                print(f'{n}/{len(T)} tasks, {time.time() - t0:.0f} s, errors {n_err}', flush=True)
    print('done', n_err, 'errors')


if __name__ == '__main__':
    main()
