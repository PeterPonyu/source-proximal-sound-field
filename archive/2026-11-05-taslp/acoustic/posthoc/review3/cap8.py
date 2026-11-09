"""Post hoc: the order cap raised to eight (PLAN-F-CAP8.md). Laptop or box; same code, gate against the stored errors.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 python3 -B posthoc/review3/cap8.py OUT --workers N [--limit N]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
sys.path.insert(0, str(S / 'common')); sys.path.insert(0, str(S / 'V2-dev')); sys.path.insert(0, str(HERE))
import estimators as E  # noqa: E402
import v2  # noqa: E402
import refit as RF  # noqa: E402

KMAX8 = 8


def one(task):
    t0 = time.process_time()
    key, stored = task['key'], task['stored']
    try:
        band, T = RF.build(key)
        models = v2.fit_band(band, KMAX8)
        perK = {K: [E.nmse(v2.predict_points(band, m, f), T[:, f]) for f in range(band.F)] for K, m in models.items()}
        gate = {f'j_K{K}': RF.relmax(perK[K], stored['nmse'][f'j_K{K}']) for K in range(6)}
        seed = key['seed']
        K8, err = v2.cv_order(band, KMAX8, E.make_folds(band.M, 5, seed))
        K5 = int(np.argmin(err[:6]))
        row = dict(study=task['study'], key=key, gate_rel=gate, K5_cv_check=bool(K5 == int(stored['K_cv'])),
                   K8=int(K8), cv_err=[float(x) for x in err], points_K6_K8={str(K): perK[K] for K in (6, 7, 8)})
        if K8 == int(stored['K_cv']):
            row['j_cv8_ext'] = stored['nmse']['j_cv_ext']; row['reused'] = True
        elif K8 == 0:
            row['j_cv8_ext'] = perK[0]; row['reused'] = False
        else:
            rm = v2.extend(band, models[K8], K8)
            row['j_cv8_ext'] = [E.nmse(v2.predict_regions(band, rm, f), T[:, f]) for f in range(band.F)]
            row['reused'] = False; row['radii'] = rm['radii'].tolist()
        row['seconds'] = round(time.process_time() - t0, 1)
        return row
    except Exception:
        return dict(study=task['study'], key=key, error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--workers', type=int, default=20); ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    T = [t for t in RF.registered_rows() if t['key']['kind'] in ('mesh', 'sim')]
    import sim
    T = sorted(T, key=lambda t: -(t['key']['M'] if t['key']['M'] else sim.cells()[t['key']['cell']]['M']))
    if a.limit:
        T = T[:a.limit]
    (out / 'started.json').write_text(json.dumps(dict(tasks=len(T), workers=a.workers, started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                                      machine=dict(cpu=platform.processor() or platform.machine(), python=platform.python_version(),
                                                                   numpy=np.__version__),
                                                      inputs={n: hashlib.sha256((S / n).read_bytes()).hexdigest() for n in
                                                              ('common/estimators.py', 'common/v2.py', 'posthoc/review3/cap8.py',
                                                               'posthoc/review3/refit.py')}), indent=1))
    t0 = time.time(); n_err = 0
    with Pool(a.workers, initializer=RF.init) as pool, open(out / 'rows.jsonl', 'w') as fh:
        for n, row in enumerate(pool.imap_unordered(one, T), 1):
            n_err += 'error' in row
            fh.write(json.dumps(row) + '\n'); fh.flush()
            if n % 100 == 0 or n == len(T):
                print(f'{n}/{len(T)} tasks, {time.time() - t0:.0f} s, errors {n_err}', flush=True)
    print('done', n_err, 'errors')


if __name__ == '__main__':
    main()
