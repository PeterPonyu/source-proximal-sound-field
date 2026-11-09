"""Development runs of the author-recipe comparators after the ICASSP 2026 full text (common/prox2.py), for
PROX-DESIGN.md: the zeta sweep over the paper's grid {10^l: l = -3..2}, on development problems only (seeds
400920-400959), never on registered data. For each problem the dictionary and kernels are built once and both
variants run at every zeta.

OPENBLAS_NUM_THREADS=1 nice -n 10 python3 -B prox2_dev.py OUT_DIR [--workers N] [--config JSON]
Development numbers only; never paper numbers.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import traceback
import warnings
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common')); sys.path.insert(0, str(HERE))
import estimators as E  # noqa: E402
import prox2 as P2  # noqa: E402
import run_dev as RD  # noqa: E402
import prox_dev as PD  # noqa: E402

CFG = {}


SIGMA2_GRID = (P2.GSMF['sigma2'], 1e-2)   # gsmf-prox: the ICASSP-unit value and a smaller noise variance


def tasks():
    """The quick subset (at most 48 microphones: 6 simulation, 12 measured problems): one job per problem, variant,
    noise variance and zeta for gsmf-prox; one job per problem for mo26-prox (its JASA setting is fixed)."""
    T = []
    for t in PD.tasks('quick'):
        T += [t + ('gsmf', s2, z) for s2 in SIGMA2_GRID for z in P2.ZETA_GRID]
        T.append(t + ('mo26', P2.MO26['sigma2'], P2.MO26['zeta']))
    return T


def init(cfg):
    warnings.simplefilter('ignore')
    CFG.update(cfg)
    for k, v in cfg.items():
        setattr(P2, k, v)
    RD.init()


def run_task(job):
    task, (variant, s2, z) = job[:-3], job[-3:]
    import resource
    try:
        if task[0] == 'sim':
            band, T, *_ = RD.sim_band(task[1], task[2])
            key = dict(kind='sim', cell=task[1], seed=task[2])
        else:
            _, spk, fc, M, d, seed = task
            band, T, _, _ = RD.meas_band(spk, fc, M, seed)
            key = dict(kind='meas', source=spk, band=fc, M=M, draw=d, seed=seed)
        a = time.process_time()
        cfg = dict(P2.GSMF if variant == 'gsmf' else P2.MO26, sigma2=s2, zeta=z)
        prep = P2.Prepared(band, cfg)
        out = dict(key, variant=variant, sigma2=s2, zeta=z, M_mics=band.M, S=len(prep.cen), delta=prep.delta, rho=prep.rho,
                   prepare_s=round(time.process_time() - a, 1))
        a = time.process_time()
        if variant == 'gsmf':
            g = P2.gsmf_prox(prep)
            out.update(nmse=[E.nmse(p, T[:, f]) for f, p in enumerate(g['preds'])], info=g['info'])
        else:
            mo = [P2.mo26_prox(prep, f) for f in range(band.F)]
            out.update(nmse=[E.nmse(r['pred'], T[:, f]) for f, r in enumerate(mo)], info=[r['info'] for r in mo])
        out['fit_s'] = round(time.process_time() - a, 1)
        out['peak_rss_gb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576
        out['config'] = CFG
        return out
    except Exception:
        return dict(task=list(job), error=traceback.format_exc(), config=CFG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--config', default='{}')
    a = ap.parse_args()
    cfg = json.loads(a.config)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    T = sorted(tasks(), key=lambda t: -(t[3] if t[0] == 'meas' else RD.sim.cells()[t[1]]['M']))   # largest first
    RD.init()
    (out / 'config.json').write_text(json.dumps(dict(config=cfg, tasks=len(T), zeta_grid=P2.ZETA_GRID, sigma2_grid=SIGMA2_GRID), indent=1))
    t0 = time.time()
    with Pool(a.workers, initializer=init, initargs=(cfg,)) as pool, open(out / 'rows.jsonl', 'w') as fh:
        for n, row in enumerate(pool.imap_unordered(run_task, T, chunksize=1), 1):
            fh.write(json.dumps(row, default=float) + '\n'); fh.flush()
            print(f'{n}/{len(T)} {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
