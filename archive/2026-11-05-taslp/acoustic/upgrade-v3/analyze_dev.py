"""Reading of the version 3 development check against the pre-set criteria A1-A5 of PLAN.md.

    python3 -B upgrade-v3/analyze_dev.py upgrade-v3/dev-rows.jsonl upgrade-v3/dev-summary.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

A = Path(__file__).resolve().parents[1]


def db(x):
    return float(10 * np.log10(np.mean(x)))


def tci(d):
    d = np.asarray(d, float); n = len(d); m = d.mean(); h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return [float(m), float(m - h), float(m + h), n]


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    errs = [r for r in rows if 'error' in r]
    rows = [r for r in rows if 'error' not in r]
    v2dev = {}
    for l in open(A / 'V2-dev/dev1-box/rows.jsonl'):
        r = json.loads(l)
        key = ('sim', r['cell'], r['seed']) if r['kind'] == 'sim' else ('meas', r['source'], r['band'], r['M'], r['draw'], r['seed'])
        v2dev[key] = r
    rep = {}
    for l in open(A / 'posthoc/review3/reproduction/rows.jsonl'):
        r = json.loads(l)
        if r['task'][0] == 'band':
            rep[r['task'][1]] = r
    out = dict(rows=len(rows), errors=len(errs))
    # A1 spherical
    sph = [r for r in rows if r['task'][0] == 'sphere']
    f3 = sum(10 * np.log10(e) > -10 for r in sph for e in r['nmse_v3'])
    f2 = sum(10 * np.log10(a / b) > -10 for r in sph for a, b in rep[r['task'][1]]['proposed_shared'])
    d_sph = [np.mean([10 * np.log10(e) for e in r['nmse_v3']]) - np.mean([10 * np.log10(a / b) for a, b in rep[r['task'][1]]['proposed_shared']])
             for r in sph]
    out['A1'] = dict(trials=len(sph), failing_bins_v3=int(f3), failing_bins_v2=int(f2), mean_db_change=tci(d_sph), pass_=bool(f3 == 0 and len(sph) == 20))
    dev = [r for r in rows if r['task'][0] != 'sphere']
    # A2 convergence
    conv = defaultdict(lambda: [0, 0]); rconv = [0, 0]; conv2 = defaultdict(lambda: [0, 0])
    for r in dev:
        for s in r['point_stats_v3']:
            if s['stage'] == 'refine' or True:
                conv[s['K']][0] += 1; conv[s['K']][1] += int(not s['cap'])
        for s in r['point_stats_v2']:
            conv2[s['K']][0] += 1; conv2[s['K']][1] += int(not s['cap'])
        for s in r['region_stats_v3']:
            rconv[0] += 1; rconv[1] += int(not s['cap'])
    frac = {k: v[1] / v[0] for k, v in sorted(conv.items())}
    out['A2'] = dict(point_refinements_converged_v3=frac, point_refinements_converged_v2={k: v[1] / v[0] for k, v in sorted(conv2.items())},
                     region_refinements_converged_v3=rconv[1] / max(rconv[0], 1),
                     pass_=bool(all(x >= 0.95 for x in frac.values()) and rconv[1] / max(rconv[0], 1) >= 0.95))
    # A3 evidence
    a3 = {}
    for K in map(str, range(1, 6)):
        ok = [r['evidence_v3'][K] >= r['evidence_v2'][K] - 1e-6 * abs(r['evidence_v2'][K]) for r in dev]
        gain = [r['evidence_v3'][K] - r['evidence_v2'][K] for r in dev]
        a3[K] = dict(frac_not_lower=float(np.mean(ok)), median_gain=float(np.median(gain)), max_loss=float(min(gain)))
    out['A3'] = dict(by_order=a3, pass_=bool(all(v['frac_not_lower'] >= 0.95 for v in a3.values())))
    # A4 accuracy vs stored version 2 primary
    unit = defaultdict(list); cell = defaultdict(list); secs3, secs2 = [], []
    for r in dev:
        t = r['task']
        key = ('sim', t[1], t[2]) if t[0] == 'sim' else tuple(t)
        old = v2dev[key]
        d = db(r['nmse_v3']) - db(old['nmse']['j_cv_ext'])
        if t[0] == 'sim':
            unit[('sim', t[2])].append(d); cell[('sim', t[1])].append(d)
        else:
            unit[('meas', t[1], t[3])].append(d); cell[('meas', t[2], t[3])].append(d)
        secs3.append(r['seconds_v3'])
        tm = old['timing']; K2 = old['j_K']['cv'] if isinstance(old.get('j_K'), dict) else None
        secs2.append(tm.get('j_fit', 0) + tm.get('j_cv', 0) + (tm.get(f'j_ext_K{K2}', 0) if K2 else 0))
    a4 = {}
    for kind in ('sim', 'meas'):
        u = [np.mean(v) for k, v in unit.items() if k[0] == kind]
        cells = {str(k[1:]): float(np.mean(v)) for k, v in cell.items() if k[0] == kind}
        a4[kind] = dict(pooled=tci(u), worst_cell=max(cells.values()), best_cell=min(cells.values()),
                        cells_worse_0_1=[k for k, v in cells.items() if v > 0.1], cells=cells)
    out['A4'] = dict(by_kind=a4, pass_=bool(all(a4[k]['pooled'][2] <= 0.05 and not a4[k]['cells_worse_0_1'] for k in a4)))
    # A5 cost
    ratio = float(np.median(secs3) / np.median(secs2))
    out['A5'] = dict(median_seconds_v3=float(np.median(secs3)), median_seconds_v2=float(np.median(secs2)), ratio=ratio, pass_=bool(ratio <= 2.5))
    out['all_pass'] = all(out[k]['pass_'] for k in ('A1', 'A2', 'A3', 'A4', 'A5'))
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ('cells',)}) for k, v in out.items()}, indent=1)[:6000])


if __name__ == '__main__':
    main()
