"""Development runs of the author-recipe comparators (common/prox.py) for PROX-DESIGN.md: free choices only, on
development problems of V2-dev (seeds 400920-400959), never on registered data.

OPENBLAS_NUM_THREADS=1 nice -n 10 python3 -B prox_dev.py OUT_DIR CONFIG_JSON [--workers N] [--subset small|full]
CONFIG_JSON sets module constants of prox (for example {"Q_MAX": 512, "LOSS": "mean", "ETA_C": 0.01, "ETA_R": 0.001}).
Development numbers only; never paper numbers.
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
sys.path.insert(0, str(HERE.parent / 'common')); sys.path.insert(0, str(HERE))
import estimators as E  # noqa: E402
import prox as PX  # noqa: E402
import run_dev as RD  # noqa: E402

CFG = {}


def tasks(subset):
    if subset == 'quick':      # the small subset's problems with at most 48 microphones (low cost, low frequencies included)
        return [t for t in tasks('small') if (t[0] == 'sim' and RD.sim.cells()[t[1]]['M'] <= 48) or (t[0] == 'meas' and t[3] <= 48)]
    sim_seeds = list(RD.SIM_SEEDS)[:2] if subset == 'small' else list(RD.SIM_SEEDS)[:4]
    T = [('sim', c, s) for c in RD.SIM_CELLS for s in sim_seeds]
    i = 0
    for spk in ('S4', 'S7', 'S8'):
        for M in (24, 48, 96, 192):
            for d in range(2):
                seed = RD.MEAS_SEEDS[i]; i += 1
                if d == 0 and (M in (48, 96) if subset == 'small' else True):
                    T += [('meas', spk, fc, M, d, seed) for fc in (125.0, 250.0, 500.0, 1000.0)]
    return T


def init(cfg):
    CFG.update(cfg)
    for k, v in cfg.items():
        setattr(PX, k, v)
    RD.init()


def run_task(task):
    try:
        if task[0] == 'sim':
            band, T, *_ = RD.sim_band(task[1], task[2])
            key = dict(kind='sim', cell=task[1], seed=task[2])
        else:
            _, spk, fc, M, d, seed = task
            band, T, _, _ = RD.meas_band(spk, fc, M, seed)
            key = dict(kind='meas', source=spk, band=fc, M=M, draw=d, seed=seed)
        tm = {}
        a = time.process_time()
        mo = [PX.mo26_prox(p) for p in band.probs]
        tm['mo26_prox'] = round(time.process_time() - a, 1)
        a = time.process_time()
        gs = PX.gsmf_prox(band)
        tm['gsmf_prox'] = round(time.process_time() - a, 1)
        nmse = dict(mo26_prox=[E.nmse(r['pred'], T[:, f]) for f, r in enumerate(mo)],
                    gsmf_prox=[E.nmse(gs['preds'][f], T[:, f]) for f in range(band.F)])
        info = dict(mo26_regions=[r['info']['regions'] for r in mo], mo26_stage1=[r['info']['after_stage1'] for r in mo],
                    mo26_dictionary=[r['info']['dictionary'] for r in mo], gsmf_regions=gs['info']['regions'],
                    gsmf_stage1=gs['info']['after_stage1'], gsmf_dictionary=gs['info']['dictionary'],
                    min_target_distance_m=[r['info']['min_target_distance_m'] for r in mo] + [gs['info']['min_target_distance_m']])
        return dict(key, M_mics=band.M, nmse=nmse, info=info, timing=tm, config=CFG)
    except Exception:
        return dict(task=list(task), error=traceback.format_exc(), config=CFG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('config')
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--subset', default='small')
    a = ap.parse_args()
    cfg = json.loads(a.config)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    T = tasks(a.subset)
    T = sorted(T, key=lambda t: -(96 if t[0] == 'sim' else t[3]))
    (out / 'config.json').write_text(json.dumps(dict(config=cfg, subset=a.subset, tasks=len(T)), indent=1))
    t0 = time.time()
    with Pool(a.workers, initializer=init, initargs=(cfg,)) as pool, open(out / 'rows.jsonl', 'w') as fh:
        for n, row in enumerate(pool.imap_unordered(run_task, T), 1):
            fh.write(json.dumps(row, default=float) + '\n'); fh.flush()
            if n % 10 == 0 or n == len(T):
                print(f'{n}/{len(T)} {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
