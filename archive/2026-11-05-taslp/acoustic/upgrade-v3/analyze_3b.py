"""Reading of the version 3b fresh development check against B1-B4 of PLAN-3b.md (development numbers only).

    python3 -B upgrade-v3/analyze_3b.py upgrade-v3/dev3b-rows.jsonl upgrade-v3/dev3b-summary.json
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy import stats


def db(x):
    return float(10 * np.log10(np.mean(x)))


def tci(d):
    d = np.asarray(d, float); n = len(d); m = d.mean(); h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return [float(m), float(m - h), float(m + h), n]


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    out = dict(rows=len(rows), errors=sum('error' in r for r in rows))
    rows = [r for r in rows if 'error' not in r]
    sph = [r for r in rows if r['task'][0] == 'sphere']
    dev = [r for r in rows if r['task'][0] != 'sphere']
    f = {t: int(sum(10 * np.log10(e) > -10 for r in sph for e in r[t]['nmse'])) for t in ('v2', 'v3b')}
    out['B1'] = dict(trials=len(sph), failing_bins=f, pass_=bool(f['v3b'] == 0 and len(sph) == 20))
    pc = defaultdict(lambda: [0, 0]); rc = [0, 0]; pc2 = defaultdict(lambda: [0, 0]); rc2 = [0, 0]
    for r in dev:
        for s in r['v3b']['point_stats']:
            pc[s['K']][0] += 1; pc[s['K']][1] += int(not s['cap'])
        for s in r['v3b']['region_stats']:
            rc[0] += 1; rc[1] += int(not s['cap'])
        for s in r['v2']['point_stats']:
            pc2[s['K']][0] += 1; pc2[s['K']][1] += int(not s['cap'])
        for s in r['v2']['region_stats']:
            rc2[0] += 1; rc2[1] += int(not s['cap'])
    pf = {k: v[1] / v[0] for k, v in sorted(pc.items())}
    out['B2'] = dict(point_v3b=pf, region_v3b=rc[1] / max(rc[0], 1), point_v2={k: v[1] / v[0] for k, v in sorted(pc2.items())},
                     region_v2=rc2[1] / max(rc2[0], 1), pass_=bool(all(x >= 0.95 for x in pf.values()) and rc[1] / max(rc[0], 1) >= 0.95))
    unit = defaultdict(list); cell = defaultdict(list); s2, s3 = [], []
    for r in dev:
        t = r['task']; d = db(r['v3b']['nmse']) - db(r['v2']['nmse'])
        if t[0] == 'sim':
            unit[('sim', t[2])].append(d); cell[('sim', t[1])].append(d)
        else:
            unit[('meas', t[1], t[3])].append(d); cell[('meas', t[2], t[3])].append(d)
        s2.append(r['v2']['seconds']); s3.append(r['v3b']['seconds'])
    b3 = {}
    for kind in ('sim', 'meas'):
        u = [np.mean(v) for k, v in unit.items() if k[0] == kind]
        cells = {str(k[1:]): float(np.mean(v)) for k, v in cell.items() if k[0] == kind}
        b3[kind] = dict(pooled=tci(u), worst_cell=max(cells.values()), best_cell=min(cells.values()),
                        cells_worse_0_1=[k for k, v in cells.items() if v > 0.1], cells=cells)
    out['B3'] = dict(by_kind=b3, pass_=bool(all(b3[k]['pooled'][2] <= 0.05 and not b3[k]['cells_worse_0_1'] for k in b3)))
    ratio = float(np.median(s3) / np.median(s2))
    out['B4'] = dict(median_seconds_v2=float(np.median(s2)), median_seconds_v3b=float(np.median(s3)), ratio=ratio, pass_=bool(ratio <= 2.5))
    # descriptive
    desc = {}
    for kind in ('sim', 'meas', 'sphere'):
        rs = [r for r in rows if r['task'][0] == kind]
        desc[kind] = {t: dict(K=dict(sorted(Counter(r[t]['K'] for r in rs).items())), at_cap=float(np.mean([r[t]['K'] == 5 for r in rs])))
                      for t in ('v2', 'v3b')}
        desc[kind]['v3b_argmin'] = dict(sorted(Counter(r['v3b']['K_argmin'] for r in rs).items()))
    ev = {}
    for K in map(str, range(1, 6)):
        g = [r['v3b']['evidence'][K] - r['v2']['evidence'][K] for r in dev]
        ev[K] = dict(frac_not_lower=float(np.mean([x >= -1e-6 * abs(r['v2']['evidence'][K]) for x, r in zip(g, dev)])), median_gain=float(np.median(g)))
    desc['evidence_by_order'] = ev
    out['descriptive'] = desc
    out['all_pass'] = all(out[k]['pass_'] for k in ('B1', 'B2', 'B3', 'B4'))
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
    show = json.loads(json.dumps(out))
    for k in show['B3']['by_kind'].values():
        k.pop('cells')
    print(json.dumps(show, indent=1))


if __name__ == '__main__':
    main()
