"""WP2 use 2 (PLAN.md, descriptive): the proposed estimator's ball extension refitted AND predicted with the exact kernel
(fit: cubature_diag.mo26_objective_series, the series evidence with analytic gradient; predict: exact_ball everywhere),
at the registered cross-validated order, from the registered greedy point fit. Sample: the WP1 room-grid sample
(loudspeakers 0, 8, 16, 24, draw 0, every band x M: 64) and 64 TB-S2 problems (every 4th cell in sorted order, which happens to include
the far control A1-beside1.2; seeds 402000-402003). Compared with the registered rows (j_cv_ext).

    OPENBLAS_NUM_THREADS=1 nice -n 10 python3 -B posthoc/upgrade-wp2/refit_exact.py OUT.jsonl --workers 16
"""
import argparse, json, sys, time, traceback, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; A = HERE.parents[1]
for p in (HERE, A / 'common', A / 'V2-dev', A / 'posthoc' / 'review3'):
    sys.path.insert(0, str(p))
import estimators as E  # noqa: E402
import v2  # noqa: E402
import cubature_diag as CD  # noqa: E402
import refit as RF  # noqa: E402
import stress as ST  # noqa: E402


def keys():
    rows = [json.loads(l) for l in open(A / 'posthoc/review3/refit/rows.jsonl')]
    mesh = [r for r in rows if r['key']['kind'] == 'mesh' and r['key']['source'] in (0, 8, 16, 24) and r['key']['draw'] == 0]
    sims = sorted({r['key']['cell'] for r in rows if r['key']['kind'] == 'sim'})
    pick = sims[::4]
    sim = [r for r in rows if r['key']['kind'] == 'sim' and r['key']['cell'] in pick and r['key']['seed'] < 402004]
    return mesh + sim


def one(row):
    warnings.simplefilter('ignore')
    key = row['key']
    try:
        st = row['proposed']; K = int(st['K_cv'])
        if K == 0:
            return dict(key=key, K=0, skipped=True)
        band, T = RF.build(key)
        t0 = time.process_time()
        models = v2.fit_band(band, 5)
        with CD.exact_kernel():
            rm = v2.extend(band, models[K], K)
        secs = time.process_time() - t0
        pred = [ST.predict(band.probs[f], rm['centres'], rm['radii'], rm['lw'][f], rm['R'][f], band.q, 'exact') for f in range(band.F)]
        nm = [E.nmse(p, T[:, f]) for f, p in enumerate(pred)]
        b = st['balls']
        moved = float(max(np.min(np.linalg.norm(np.array(b['centres'])[:, None] - rm['centres'][None], axis=2), axis=1)))
        return dict(key=key, K=K, nmse_exact_refit=nm, nmse_registered=st['j_cv_ext'], radii_exact=rm['radii'].tolist(),
                    radii_registered=b['radii'], max_centre_shift_m=moved, converged=bool(rm['converged']), seconds=round(secs, 1))
    except Exception:
        return dict(key=key, error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    ks = keys(); print(len(ks), 'problems', flush=True)
    with Pool(a.workers, initializer=RF.init) as pool, open(a.out, 'w') as fh:
        for n, r in enumerate(pool.imap_unordered(one, ks), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush(); print(n, flush=True)


if __name__ == '__main__':
    main()
