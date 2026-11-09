"""Post hoc sensitivity analysis of TB-M2 (labelled; not a registered rule). Written 2026-11-05 after the run.

Two comparators returned non-finite errors on MeshRIR. The common-domain sparse Bayesian dictionary has monopoles on
or within millimetres of target points of the planar grid (a defect of the comparator implementation: the guard
protected microphones only), and the cap-5 region kernel failed numerically in one bin of one problem.
1. The best generic estimator chosen with each generic mean over its finite problems only; pairs with both finite.
2. The primary against the cap-5 region kernel without its single non-finite problem.
Same intervals as the registered analysis: Student-t over loudspeakers, Bonferroni over 16 cells, delta = 0.1 dB.

python3 -B sensitivity_nonfinite.py studies/TB-M2/primary
"""
import json, sys
from collections import defaultdict
import numpy as np
from scipy import stats

GEN = ('sinc_ev', 'rbf', 'adk_standin', 'sbl_em_common', 'sbl_mackay_common')
ALPHA = 0.05 / 16


def e(v):
    v = np.asarray(v, float)
    return 10 * np.log10(np.mean(v)) if np.all(np.isfinite(v)) and np.all(v < 1e6) else np.nan


def tint(x, a=ALPHA):
    x = np.asarray(x); n = len(x); m = x.mean(); h = stats.t.ppf(1 - a / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(m), float(m - h), float(m + h)


def cls(m, lo, hi):
    return 'win' if (hi < 0 and m < -0.1) else ('loss' if (lo > 0 and m > 0.1) else 'tie')


def main():
    out = sys.argv[1]
    rows = [json.loads(l) for l in open(f'{out}/rows.jsonl')]
    mesh = [r for r in rows if r['kind'] == 'mesh' and 'error' not in r]
    cells = defaultdict(list)
    for r in mesh:
        cells[(r['band'], r['M'])].append(r)
    nonfinite = {m: sum(not np.isfinite(e(r['nmse'][m])) for r in mesh) for m in GEN + ('mo26_cap5', 'gsmf', 'j_cv_ext')}
    s1, s2, per = {}, {}, defaultdict(list)
    for c in sorted(cells):
        rs = cells[c]
        with np.errstate(all='ignore'):
            fin = {m: np.nanmean([e(r['nmse'][m]) for r in rs]) for m in GEN}
        bg = min((m for m in GEN if np.isfinite(fin[m])), key=lambda m: fin[m])
        for target, store in ((bg, s1), ('mo26_cap5', s2)):
            by = defaultdict(list)
            for r in rs:
                a, b = e(r['nmse']['j_cv_ext']), e(r['nmse'][target])
                if np.isfinite(a) and np.isfinite(b):
                    by[r['source']].append(a - b)
            d = {u: float(np.mean(v)) for u, v in by.items()}
            m, lo, hi = tint(list(d.values()))
            store[f'{int(c[0])}Hz-M{c[1]}'] = dict(comparator=target, mean=m, low=lo, high=hi, cls=cls(m, lo, hi), non_inferior=hi < 0.2)
            if target == 'mo26_cap5':
                for u, v in d.items():
                    per[u].append(v)
    vals = [np.mean(v) for v in per.values()]
    pm, plo, phi = tint(vals, 0.05)
    res = dict(nonfinite_tasks_of_1024=nonfinite,
               best_generic_over_finite_tasks=dict(per_cell=s1, wins=sum(v['cls'] == 'win' for v in s1.values()),
                                                   ties=sum(v['cls'] == 'tie' for v in s1.values()), losses=sum(v['cls'] == 'loss' for v in s1.values())),
               mo26_cap5_without_nonfinite=dict(per_cell=s2, non_inferior=sum(v['non_inferior'] for v in s2.values()),
                                                wins=sum(v['cls'] == 'win' for v in s2.values()), ties=sum(v['cls'] == 'tie' for v in s2.values()),
                                                losses=sum(v['cls'] == 'loss' for v in s2.values()), pooled=dict(mean=pm, low=plo, high=phi)))
    json.dump(res, open(f'{out}/../sensitivity-nonfinite.json', 'w'), indent=1, default=float)
    print(json.dumps({k: (v if k == 'nonfinite_tasks_of_1024' else {kk: vv for kk, vv in v.items() if kk != 'per_cell'}) for k, v in res.items()}, indent=1, default=float))


if __name__ == '__main__':
    main()
