"""Cross-machine agreement of the development run: the same tasks run on the laptop and on the rented box.

python3 -B compare_machines.py LAPTOP_ROWS BOX_ROWS OUT.json
For every task present on both machines and every method and bin, d = |10 log10(box / laptop)| in dB. Reports, per
method and overall, the fraction of entries with d <= 0.001, 0.01, 0.05, 0.5 dB, the median and the maximum, and the
order selections that differ. This calibrates the tolerance of remote/cross_check.py before any registered run.
"""
from __future__ import annotations
import json
import sys
import numpy as np


def key(r):
    return (r['kind'], r.get('cell'), r.get('source'), r.get('band'), r.get('M'), r.get('draw'), r['seed'])


def main():
    lap = {key(r): r for r in map(json.loads, open(sys.argv[1])) if 'error' not in r}
    box = {key(r): r for r in map(json.loads, open(sys.argv[2])) if 'error' not in r}
    common = sorted(set(lap) & set(box), key=str)
    per, allv, ksel = {}, [], []
    for k in common:
        a, b = lap[k]['nmse'], box[k]['nmse']
        for m in a:
            if m in b:
                d = np.abs(10 * np.log10(np.asarray(b[m]) / np.asarray(a[m])))
                per.setdefault(m, []).extend(d.tolist()); allv.extend(d.tolist())
        for blk in ('j_K', 'fj_K'):
            if blk in lap[k] and blk in box[k]:
                ksel.append(int(lap[k][blk] != box[k][blk]))
        if lap[k].get('family') != box[k].get('family'):
            ksel.append(1)

    def stats(v):
        v = np.asarray(v)
        return dict(n=len(v), within_0001=float(np.mean(v <= 0.001)), within_001=float(np.mean(v <= 0.01)),
                    within_005=float(np.mean(v <= 0.05)), within_05=float(np.mean(v <= 0.5)), median=float(np.median(v)),
                    p99=float(np.quantile(v, 0.99)), max=float(v.max()))
    res = dict(tasks=len(common), overall=stats(allv), per_method={m: stats(v) for m, v in sorted(per.items())},
               selection_or_family_differences=int(sum(ksel)), selection_comparisons=len(ksel))
    json.dump(res, open(sys.argv[3], 'w'), indent=1)
    print(json.dumps(dict(tasks=res['tasks'], overall=res['overall'], selection_or_family_differences=res['selection_or_family_differences'],
                          worst_methods=sorted(((m, s['within_005'], round(s['max'], 3)) for m, s in res['per_method'].items()), key=lambda t: t[1])[:8]), indent=1))


if __name__ == '__main__':
    main()
