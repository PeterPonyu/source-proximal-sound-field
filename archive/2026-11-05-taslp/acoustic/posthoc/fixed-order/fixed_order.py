"""Post hoc, descriptive: the cross-validated order against fixed orders, and a stricter classification (PLAN.md).

Uses the registered statistics of analyze_v2.py, imported unchanged. Reads the registered TB-S2 and TB-M2 rows.

    python3 -B posthoc/fixed-order/fixed_order.py
"""
from __future__ import annotations
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
sys.path.insert(0, str(S))
import analyze_v2 as A  # noqa: E402

ORDERS = range(6)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def finite(t):
    return all(t[k] is not None and math.isfinite(t[k]) for k in ('mean', 'low', 'high'))


def study_rows(study):
    d = S / 'studies' / study / 'primary'
    assert sha(d / 'rows.jsonl') == json.loads((d / 'run-receipt.json').read_text())['outputs']['rows.jsonl'], study
    rows, errs = A.load(d)
    if study == 'TB-S2':
        for r in rows:
            r['cell_key'] = r['cell']
        cells = list(dict.fromkeys(r['cell'] for r in rows))
        return rows, errs, 'sim', [c for c in cells if c != A.FAR], cells
    rows = [r for r in rows if r['kind'] == 'mesh']
    for r in rows:
        r['cell_key'] = f"{int(r['band'])}Hz-M{r['M']}"
    cells = sorted({r['cell_key'] for r in rows}, key=lambda c: (int(c.split('Hz')[0]), int(c.split('M')[1])))
    return rows, errs, 'mesh', cells, cells


def compare(rows, kind, family, a, b):
    """Per-condition registered classification over the family, the pooled comparison and the range of cell means."""
    alpha = A.ALPHA / len(family)
    by = defaultdict(list)
    for r in rows:
        by[r['cell_key']].append(r)
    per = {}
    for c in family:
        m, lo, hi = A.tint(list(A.cell_diff(by[c], kind, a, b).values()), alpha)
        per[c] = dict(mean=m, low=lo, high=hi)
        per[c]['cls'] = A.classify(m, lo, hi) if finite(per[c]) else 'undefined'
    cl = [v['cls'] for v in per.values()]
    means = [v['mean'] for v in per.values()]
    return dict(wins=cl.count('win'), ties=cl.count('tie'), losses=cl.count('loss'), undefined=cl.count('undefined'),
                cells=len(cl), pooled=A.pooled(rows, kind, family, a, b), cell_mean_range=[min(means), max(means)],
                per_condition=per)


def strict(t):
    """Win only if the whole interval is below -DELTA; loss only if it is above +DELTA; undefined kept apart."""
    if not finite(t):
        return 'undefined'
    return 'win' if t['high'] < -A.DELTA else ('loss' if t['low'] > A.DELTA else 'tie')


def classification(T, family, key_of):
    reg, st = [], []
    for c in family:
        t = T[c]['comparisons'][key_of(c)]
        reg.append(t['cls'] if finite(t) else 'undefined'); st.append(strict(t))
    cnt = lambda v: dict(wins=v.count('win'), ties=v.count('tie'), losses=v.count('loss'), undefined=v.count('undefined'))
    return dict(registered=cnt(reg), strict=cnt(st), cells=len(family))


def main():
    out = dict(status='post hoc, descriptive, not registered (PLAN.md)', delta_db=A.DELTA, alpha=A.ALPHA,
               provenance={p: sha(S / p) for p in ('analyze_v2.py', 'posthoc/fixed-order/PLAN.md', 'posthoc/fixed-order/fixed_order.py',
                                                   'studies/TB-S2/primary/rows.jsonl', 'studies/TB-M2/primary/rows.jsonl')},
               studies={})
    for study in ('TB-S2', 'TB-M2'):
        rows, errs, kind, family, allcells = study_rows(study)
        mism = sum(r['nmse']['j_cv'] != r['nmse'][f"j_K{r['info']['j_K']['cv']}"] for r in rows)
        assert mism == 0 and not errs, (study, mism, len(errs))
        res = dict(rows=len(rows), error_rows=len(errs), family_cells=len(family), cv_equals_selected_fixed_order_mismatches=int(mism))
        res['cv_vs_fixed'] = {f'K{K}': compare(rows, kind, family, 'j_cv', f'j_K{K}') for K in ORDERS}
        res['fixed_vs_fixed'] = {'K5-vs-K1': compare(rows, kind, family, 'j_K5', 'j_K1'),
                                 'K2-vs-K1': compare(rows, kind, family, 'j_K2', 'j_K1')}
        best = max(ORDERS, key=lambda K: res['cv_vs_fixed'][f'K{K}']['pooled']['mean'])
        res['best_fixed_order'] = dict(K=int(best), cv_vs_best=res['cv_vs_fixed'][f'K{best}']['pooled'])
        K_sel = np.bincount([r['info']['j_K']['cv'] for r in rows if r['cell_key'] in family], minlength=6)
        res['selected_order_fraction'] = (K_sel / K_sel.sum()).tolist()
        T = A.analyse(rows, kind, family, len(family))
        bg = lambda c: f"{A.PRIMARY}-vs-{T[c]['best_generic']}"
        res['strict'] = {'best_generic': classification(T, family, bg),
                         'gsmf': classification(T, family, lambda c: f'{A.PRIMARY}-vs-gsmf'),
                         'mo26_cap5': classification(T, family, lambda c: f'{A.PRIMARY}-vs-mo26_cap5')}
        Ts = A.analyse(rows, kind, allcells, len(allcells)) if kind == 'sim' else T
        res['strict']['sinc_ev'] = classification(Ts, allcells, lambda c: f'{A.PRIMARY}-vs-sinc_ev')
        out['studies'][study] = res
        print(study, 'rows', len(rows), 'mismatches', mism, 'best fixed K', best)
        for k, v in list(res['cv_vs_fixed'].items()) + list(res['fixed_vs_fixed'].items()):
            p = v['pooled']
            print(f"  {k:9s} {v['wins']}/{v['ties']}/{v['losses']}  pooled {p['mean']:+.3f} [{p['low']:+.3f}, {p['high']:+.3f}]"
                  f"  cells {v['cell_mean_range'][0]:+.2f} to {v['cell_mean_range'][1]:+.2f}")
        for k, v in res['strict'].items():
            print(f"  strict {k:12s} registered {v['registered']}  strict {v['strict']}  of {v['cells']}")
    (HERE / 'fixed-order.json').write_text(json.dumps(out, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
