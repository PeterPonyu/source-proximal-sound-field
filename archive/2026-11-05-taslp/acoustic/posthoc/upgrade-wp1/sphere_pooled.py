"""WP1 (post hoc): the published spherical configuration (posthoc/review3/reproduce.py, shared band, 20 trials) with the
registered fitting (v3.V2_OPTS, which reproduces version 2 exactly) and with converged fitting (zoomed scan, closed-form
weight reset, scaled coordinates, iteration caps 1000 with ball restarts; the order rule unchanged: cross-validated argmin).
Squared error and truth power per bin, so the pooled error matches Table III's definition. Gate: the registered fitting
reproduces the stored rows of posthoc/review3/reproduction.

    OPENBLAS_NUM_THREADS=1 nice -n 10 python3 -B posthoc/upgrade-wp1/sphere_pooled.py OUT.jsonl --workers 6
"""
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; A = HERE.parents[1]
for p in (A / 'upgrade-v3', A / 'common', A / 'posthoc' / 'review3'):
    sys.path.insert(0, str(p))
import v3  # noqa: E402
import v2  # noqa: E402
import reproduce as RP  # noqa: E402

CONVERGED = v3.Opts(region_restarts=2)          # V3 fitting with ball restarts; order rule 'argmin' as registered


def one(seed):
    src, q = RP.trial(seed)
    Y, T = zip(*[RP.observe(RP.X_MIC, src, q, f, seed) for f in RP.FREQS])
    band = v2.Band(RP.X_MIC, np.stack(Y, 1), q, RP.FREQS, c=RP.C0, box=RP.BOX)
    out = dict(seed=seed)
    for tag, opts in (('registered', v3.V2_OPTS), ('converged', CONVERGED)):
        t0 = time.process_time()
        pred, models, info = v3.primary(band, seed, opts=opts)
        out[tag] = dict(K=info['K'], sq=[RP.sq(pred(i), T[i]) for i in range(band.F)], seconds=round(time.process_time() - t0, 1),
                        point_caps=sum(s['cap'] for s in info['point_stats']), point_refinements=len(info['point_stats']),
                        region_caps=sum(s['cap'] for s in info.get('region_stats', [])), region_refinements=len(info.get('region_stats', [])))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    with Pool(a.workers, initializer=RP.init) as pool, open(a.out, 'w') as fh:
        for r in pool.imap_unordered(one, list(RP.SEEDS)):
            fh.write(json.dumps(r) + '\n'); fh.flush(); print(r['seed'], flush=True)
