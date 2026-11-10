"""Revised fitting on every registered problem (PLAN.md; post hoc, descriptive).

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python3 -B posthoc/upgrade-revfit/revfit.py STUDY OUT.jsonl --workers N [--gate]

STUDY is mesh, miracle or sim. Rows already in OUT.jsonl are skipped, so an interrupted run resumes.
--gate runs the registered fitting (v3.V2_OPTS) on the first small problems instead and compares with the stored rows.
"""
import argparse, json, sys, time, traceback, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; A = HERE.parents[1]
for p in (A / 'upgrade-v3', A / 'common', A / 'posthoc' / 'review3', A / 'studies' / 'TB-M3-MIRACLE'):
    sys.path.insert(0, str(p))
import v3  # noqa: E402
import estimators as E  # noqa: E402
import refit as RF  # noqa: E402
import run_m3 as RM  # noqa: E402

REVISED = v3.Opts(region_restarts=2)       # as posthoc/upgrade-wp1/sphere_pooled.py; order rule 'argmin' as registered
OPTS = {'revised': REVISED, 'registered': v3.V2_OPTS}
STUDY = {}


def init(study):
    STUDY['s'] = study
    if study == 'miracle':
        RM.init(A / 'studies/TB-M3-MIRACLE/extract', ['A1', 'R2'])
    else:
        RF.init()


def keys(study):
    if study == 'miracle':
        ks = []
        for l in open(A / 'studies/TB-M3-MIRACLE/primary/rows.jsonl'):
            r = json.loads(l)
            ks.append(dict(kind='miracle', scenario=r['scenario'], source=r['source'], band=r['band'], M=r['M'], draw=r['draw'], seed=r['seed'],
                           reg=r['nmse']['j_cv_ext'], K_reg=r['info']['j_K']['cv']))
        return ks
    kind = 'mesh' if study == 'mesh' else 'sim'
    return [dict(r['key'], reg=r['stored']['nmse']['j_cv_ext'], K_reg=r['stored']['K_cv']) for r in RF.registered_rows() if r['key']['kind'] == kind]


def ident(k):
    return json.dumps({a: k[a] for a in sorted(k) if a not in ('reg', 'K_reg')})


def one(arg):
    warnings.simplefilter('ignore')
    k, tag = arg
    key = {a: k[a] for a in k if a not in ('reg', 'K_reg')}
    try:
        if k['kind'] == 'miracle':
            band, T, _, _ = RM.build(k['scenario'], k['source'], k['band'], k['M'], k['seed'])
        else:
            band, T = RF.build(key)
        t0 = time.process_time()
        pred, models, info = v3.primary(band, k['seed'], opts=OPTS[tag])
        nm = [E.nmse(pred(f), T[:, f]) for f in range(band.F)]
        secs = time.process_time() - t0
        ps, rs = info['point_stats'], info.get('region_stats', [])
        return dict(key=key, fitting=tag, M=band.M, K=info['K'], K_reg=k['K_reg'], nmse=nm, nmse_reg=k['reg'], cv_err=info['cv_err'],
                    point_caps=sum(s['cap'] for s in ps), point_refinements=len(ps),
                    region_caps=sum(s['cap'] for s in rs), region_refinements=len(rs), seconds=round(secs, 1))
    except Exception:
        return dict(key=key, fitting=tag, error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('study', choices=('mesh', 'miracle', 'sim')); ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=16); ap.add_argument('--gate', action='store_true')
    a = ap.parse_args()
    init(a.study)
    ks = keys(a.study)
    # Task order only (largest first); simulation cell names do not all carry the microphone count, so it is read from the stored rows.
    simM = {}
    if a.study == 'sim':
        for l in open(A / 'studies/TB-S2/primary/rows.jsonl'):
            r = json.loads(l)
            simM[(r['cell'], r['seed'])] = r.get('M_mics', 0)
    msize = lambda k: k['M'] if k.get('M') else simM.get((k['cell'], k['seed']), 0)
    if a.gate:
        small = sorted([k for k in ks if k.get('M') == 24], key=ident)[:16 if a.study == 'mesh' else 12]
        todo = [(k, 'registered') for k in small]
    else:
        done = set()
        if Path(a.out).exists():
            done = {json.dumps(json.loads(l)['key'], sort_keys=True) for l in open(a.out) if 'error' not in json.loads(l)}
        todo = [(k, 'revised') for k in sorted(ks, key=lambda k: -msize(k)) if ident(k) not in done]
    print(len(todo), 'problems', flush=True)
    t0 = time.time()
    with Pool(a.workers, initializer=init, initargs=(a.study,)) as pool, open(a.out, 'a') as fh:
        for n, r in enumerate(pool.imap_unordered(one, todo), 1):
            fh.write(json.dumps(r, default=float) + '\n'); fh.flush()
            if n % 20 == 0 or n == len(todo) or 'error' in r:
                print(n, f'{time.time() - t0:.0f}s', 'ERR' if 'error' in r else '', flush=True)


if __name__ == '__main__':
    main()
