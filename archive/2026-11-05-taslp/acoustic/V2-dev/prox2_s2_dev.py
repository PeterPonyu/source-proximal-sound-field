"""Development probe for TB-PROX-S2 (before its registration; development problems only, never reported as results).

mo26-prox with the paper's stage-2 stop (common/prox2_s2.py) on the four 24-microphone development problems of the
equivalence check, at the registered settings. Records per bin the stage lengths, active sets, process time and NMSE,
and checks that stage 1 is unchanged against the development sweep's rows (V2-dev/prox-dev/N3, registered stop).
Purpose: the cost estimate of the registered follow-up (stage 2 dominates mo26-prox's time, and the stop sets its length).

OPENBLAS_NUM_THREADS=1 nice -n 10 python3 -B prox2_s2_dev.py OUT_DIR
"""
import json, math, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common')); sys.path.insert(0, str(HERE))
import estimators as E  # noqa: E402
import prox2 as P2  # noqa: E402
import prox2_s2 as S2  # noqa: E402
import run_dev as RD  # noqa: E402

PROBLEMS = [('A2-f125-M24', 400920), ('A2-f125-M24', 400921), ('A2-f1000-M24', 400920), ('A2-f1000-M24', 400921)]


def job(p):
    warnings.simplefilter('ignore')
    cell, seed = p
    band, T, *_ = RD.sim_band(cell, seed)
    a = time.process_time()
    prep = P2.Prepared(band, P2.MO26)
    prep_s = time.process_time() - a
    bins = []
    for f in range(band.F):
        b = time.process_time()
        r = S2.mo26_prox_s2(prep, f)
        bins.append(dict(after_stage1=r['info']['after_stage1'], regions=r['info']['regions'], iterations=r['info']['iterations'],
                         nmse=E.nmse(r['pred'], T[:, f]), seconds=round(time.process_time() - b, 1)))
    return dict(cell=cell, seed=seed, prepare_s=round(prep_s, 1), fit_s=round(time.process_time() - a - prep_s, 1), bins=bins)


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    ref = {}
    for l in open(HERE / 'prox-dev/N3/rows.jsonl'):
        r = json.loads(l)
        if r.get('kind') == 'sim' and r['variant'] == 'mo26':
            ref[(r['cell'], r['seed'])] = r
    with Pool(4) as pool, open(out / 'rows.jsonl', 'w') as fh:
        for row in pool.imap_unordered(job, PROBLEMS):
            o = ref[(row['cell'], row['seed'])]
            row['registered_stop'] = dict(fit_s=o['fit_s'], machine='box', bins=[dict(after_stage1=b['after_stage1'], regions=b['regions'],
                                          iterations=b['iterations']) for b in o['info']], nmse=o['nmse'])
            row['stage1_unchanged'] = all(a['after_stage1'] == b['after_stage1'] and a['iterations'][0] == b['iterations'][0]
                                          for a, b in zip(row['bins'], o['info']))
            fh.write(json.dumps(row, default=float) + '\n'); fh.flush()
            print(row['cell'], row['seed'], 'stage1 unchanged', row['stage1_unchanged'], 'fit_s', row['fit_s'],
                  'stage2 lengths', [b['iterations'][1] for b in row['bins']], 'registered', [b['iterations'][1] for b in o['info']], flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
