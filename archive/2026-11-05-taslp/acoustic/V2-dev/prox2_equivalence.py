"""Equivalence of the faster stage-2 code with the version that produced the development sweep (V2-dev/prox-dev/N3,
run on the box with the bundle of SHA-256 fd6a1cf2...): both variants at the registered settings on the four
24-microphone development problems; the active sets (count and radii of the live regions per bin) must be identical
and every bin's NMSE equal within 1e-6 dB. The laptop and the box agree bit for bit on the pinned stack
(remote/cross_check.py), so the box rows are the reference.

OPENBLAS_NUM_THREADS=1 nice -n 10 python3 -B prox2_equivalence.py OUT.json
"""
import json, math, sys, warnings
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common')); sys.path.insert(0, str(HERE))
import estimators as E  # noqa: E402
import prox2 as P2  # noqa: E402
import run_dev as RD  # noqa: E402

PROBLEMS = [('A2-f125-M24', 400920), ('A2-f125-M24', 400921), ('A2-f1000-M24', 400920), ('A2-f1000-M24', 400921)]


def job(args):
    warnings.simplefilter('ignore')
    (cell, seed), variant = args
    band, T, *_ = RD.sim_band(cell, seed)
    if variant == 'gsmf':
        prep = P2.Prepared(band, P2.GSMF)
        g = P2.gsmf_prox(prep)
        return dict(cell=cell, seed=seed, variant=variant, nmse=[E.nmse(p, T[:, f]) for f, p in enumerate(g['preds'])], info=g['info'])
    prep = P2.Prepared(band, P2.MO26)
    mo = [P2.mo26_prox(prep, f) for f in range(band.F)]
    return dict(cell=cell, seed=seed, variant=variant, nmse=[E.nmse(r['pred'], T[:, f]) for f, r in enumerate(mo)], info=[r['info'] for r in mo])


def main(out):
    ref = {}
    for l in open(HERE / 'prox-dev/N3/rows.jsonl'):
        r = json.loads(l)
        if r.get('kind') != 'sim':
            continue
        if r['variant'] == 'gsmf' and not (abs(r['sigma2'] - P2.GSMF['sigma2']) < 1e-12 and r['zeta'] == P2.GSMF['zeta']):
            continue
        ref[(r['cell'], r['seed'], r['variant'])] = r
    with Pool(8) as pool:
        new = pool.map(job, [(p, v) for p in PROBLEMS for v in ('gsmf', 'mo26')])
    res = []
    for n in new:
        o = ref[(n['cell'], n['seed'], n['variant'])]
        ddb = max(abs(10 * math.log10(a) - 10 * math.log10(b)) for a, b in zip(n['nmse'], o['nmse']))
        if n['variant'] == 'gsmf':
            same = n['info']['regions'] == o['info']['regions']
            detail = dict(regions=(n['info']['regions'], o['info']['regions']))
        else:
            same = all(a['regions'] == b['regions'] and a['after_stage1'] == b['after_stage1'] and a['iterations'] == b['iterations']
                       and len(a['radii']) == len(b['radii']) and all(abs(x - y) < 1e-9 for x, y in zip(a['radii'], b['radii']))
                       for a, b in zip(n['info'], o['info']))
            detail = dict(regions=([a['regions'] for a in n['info']], [b['regions'] for b in o['info']]),
                          iterations=([a['iterations'] for a in n['info']], [b['iterations'] for b in o['info']]))
        res.append(dict(cell=n['cell'], seed=n['seed'], variant=n['variant'], max_abs_db_diff=ddb, active_sets_identical=bool(same), **detail))
    passed = all(r['active_sets_identical'] and r['max_abs_db_diff'] <= 1e-6 for r in res)
    Path(out).write_text(json.dumps(dict(problems=res, passed=passed), indent=1, default=float))
    for r in res:
        print(r['cell'], r['seed'], r['variant'], 'identical' if r['active_sets_identical'] else 'DIFFERENT', f"{r['max_abs_db_diff']:.2e} dB")
    print('PASSED' if passed else 'FAILED')


if __name__ == '__main__':
    main(sys.argv[1])
