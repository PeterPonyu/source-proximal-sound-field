"""Analysis of the point-plus-dipole run (posthoc/review4/PLAN.md, part B), with the registered statistics of
analyze_v2.py imported unchanged and the reading fixed by the plan before the run.

    python3 -B posthoc/review4/analyze_dipole.py posthoc/review4/dipole-run posthoc/review4/dipole-summary.json
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
sys.path.insert(0, str(S))
import analyze_v2 as AV  # noqa: E402

P = AV.PRIMARY
KEY = ('scenario', 'source', 'band', 'M', 'draw', 'seed')
PAIRS = [(P, 'two_stage_dipole'), ('two_stage_dipole', 'two_stage_point'), ('j_K5', 'two_stage_point'),
         ('two_stage_dipole', 'j_K5')]


def main():
    run, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    prim = {tuple(r[k] for k in KEY): r for r in map(json.loads, open(S / 'studies/TB-M3-MIRACLE/primary/rows.jsonl'))}
    new = [json.loads(l) for l in open(run / 'rows.jsonl')]
    errors = [r for r in new if 'error' in r]
    rows, refit_dev, skipped = [], [], 0
    for r in new:
        if 'error' in r:
            continue
        if not r['two_stage_ok']:
            skipped += 1; continue
        p = prim[tuple(r[k] for k in KEY)]
        stored = np.array(p['nmse']['two_stage_point']); refit = np.array(r['nmse']['two_stage_point_refit'])
        refit_dev.append(float(np.max(np.abs(refit - stored) / stored)))
        e = {m: AV.db(v) for m, v in p['nmse'].items()}
        e['two_stage_dipole'] = AV.db(r['nmse']['two_stage_dipole'])
        rows.append(dict(scenario=r['scenario'], source=r['source'], band=r['band'], M=r['M'], e=e,
                         cell_key=f"{int(r['band'])}Hz-M{r['M']}", dev=r['info']['log10_w_rho_dev']))
    res = dict(status='post hoc; posthoc/review4/PLAN.md part B', rows=len(rows), errors=len(errors), skipped_no_two_stage=skipped,
               refit_point_vs_stored_max_rel=max(refit_dev) if refit_dev else None)
    for sc in ('A1', 'R2'):
        rs = [r for r in rows if r['scenario'] == sc]
        by = defaultdict(list)
        for r in rs:
            by[r['cell_key']].append(r)
        cells = sorted(by, key=lambda c: (int(c.split('Hz')[0]), int(c.split('M')[1])))
        per_cell = {}
        for c in cells:
            d = {}
            for a, b in PAIRS:
                m, lo, hi = AV.tint(list(AV.cell_diff(by[c], 'mesh', a, b).values()), AV.ALPHA / len(cells))
                d[f'{a}-minus-{b}'] = dict(mean=m, low=lo, high=hi, cls=AV.classify(m, lo, hi))
            per_cell[c] = d
        per_band = {}
        for bd in sorted({int(c.split('Hz')[0]) for c in cells}):
            cs = [c for c in cells if c.startswith(f'{bd}Hz')]
            pl = {}
            for a, b in PAIRS:
                per = defaultdict(list)
                for c in cs:
                    for u, v in AV.cell_diff(by[c], 'mesh', a, b).items():
                        per[u].append(v)
                m, lo, hi = AV.tint([np.mean(v) for v in per.values()], AV.ALPHA)
                pl[f'{a}-minus-{b}'] = dict(mean=m, low=lo, high=hi, units=len(per))
            gap = pl['j_K5-minus-two_stage_point']['mean']
            closed = pl['two_stage_dipole-minus-two_stage_point']['mean'] / gap if gap < 0 else None
            mean_db = {m: float(np.mean([r['e'][m] for c in cs for r in by[c]])) for m in (P, 'j_K5', 'j_K1', 'two_stage_point', 'two_stage_dipole')}
            dev_gain = [x[-1] for c in cs for r in by[c] for x in r['dev']]
            per_band[str(bd)] = dict(pooled=pl, fraction_of_K5_gap_closed_by_dipole=closed, mean_db=mean_db,
                                     median_log_evidence_gain_of_dipole=float(np.median(dev_gain)))
        res[sc] = dict(per_cell=per_cell, per_band=per_band)
    fr = [res[sc]['per_band']['1000']['fraction_of_K5_gap_closed_by_dipole'] for sc in ('A1', 'R2')]
    if all(f is not None and f >= 0.5 for f in fr):
        reading = 'supported: a first-order directional term accounts for most of the extra points\' benefit at 1 kHz'
    elif all(f is not None and f < 0.25 for f in fr):
        reading = 'not supported: first-order directivity does not explain the extra points\' benefit at 1 kHz'
    else:
        reading = 'partial'
    res['reading'] = dict(fractions_1kHz=dict(A1=fr[0], R2=fr[1]), result=reading,
                          rule='>= 0.5 in both scenarios supported; < 0.25 in both not supported; otherwise partial (PLAN.md part B)')
    out_path.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res['reading'], indent=1))


if __name__ == '__main__':
    main()
