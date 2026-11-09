"""Post hoc diagnostic (labelled; development seeds only): does the target guard change sparse Bayesian learning in the
simulated boxes? For development fields of four TB-S2 cells, the common-domain sparse Bayesian comparators of TB-S2
(guard from microphones only) and the corrected ones (guard from microphones and targets) are run on the same bins.

OPENBLAS_NUM_THREADS=1 nice -n 15 python3 -B diag_sbl_sim_guard.py OUT.json
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common'))
import estimators as E, v2, sbl_guard as G, problems_v2 as PB
from scipy.spatial.distance import cdist

CELLS = ('base', 'A1-in', 'A2-f125-M24', 'A2-f1000-M192')
SEEDS = range(400990, 400994)


def one(task):
    cell, seed = task
    band, T, _, _ = PB.sim_band(cell, seed)
    out = dict(cell=cell, seed=seed, nearest_atom_to_target_m=[], removed_by_targets_only=[])
    for rule, a, b in (('em', 'sbl_em_common', 'sbl_em_guard'), ('mackay', 'sbl_mackay_common', 'sbl_mackay_guard')):
        out[a] = [E.nmse(v2.sbl_common(p, rule)['pred'], T[:, f]) for f, p in enumerate(band.probs)]
        st = []
        out[b] = [E.nmse(G.sbl_guarded(p, rule, st)['pred'], T[:, f]) for f, p in enumerate(band.probs)]
        if rule == 'em':
            out['removed_by_targets_only'] = [s['removed_by_targets_only'] for s in st]
    for p in band.probs:
        with v2._common_domain_dictionary():
            nodes, _ = E.sbl_dictionary(p)
        out['nearest_atom_to_target_m'].append(float(cdist(p.q, nodes).min()))
    return out


if __name__ == '__main__':
    with Pool(8) as pool:
        rows = pool.map(one, [(c, s) for c in CELLS for s in SEEDS])
    summ = {}
    for c in CELLS:
        rs = [r for r in rows if r['cell'] == c]
        for a, b in (('sbl_em_common', 'sbl_em_guard'), ('sbl_mackay_common', 'sbl_mackay_guard')):
            d = [10 * np.log10(np.mean(r[a]) / np.mean(r[b])) for r in rs]
            summ[f'{c}:{a}'] = dict(median_db_unguarded_minus_guarded=float(np.median(d)), max_abs_db=float(np.max(np.abs(d))))
        summ[f'{c}:nearest_atom_m'] = float(np.min([min(r['nearest_atom_to_target_m']) for r in rs]))
        summ[f'{c}:removed_by_targets_only'] = int(sum(sum(r['removed_by_targets_only']) for r in rs))
    json.dump(dict(rows=rows, summary=summ), open(sys.argv[1], 'w'), indent=1)
    print(json.dumps(summ, indent=1))
