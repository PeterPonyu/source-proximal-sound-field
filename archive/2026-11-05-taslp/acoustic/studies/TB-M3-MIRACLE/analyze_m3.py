"""TB-M3-MIRACLE registered analysis (PROTOCOL.md), with the statistics of analyze_v2.py imported unchanged.

python3 -B analyze_m3.py RUN_DIR   ->  RUN_DIR/summary.json
In the rows the sparse Bayesian comparators keep their registered names (sbl_em_common, sbl_mackay_common) but are the
target-guarded versions, and adk_standin uses the clip-masked gradient, as the protocol fixes.
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import analyze_v2 as AV  # noqa: E402

P = AV.PRIMARY


def main():
    out_dir = Path(sys.argv[1])
    rows, errs = AV.load(out_dir)
    res = dict(n_rows=len(rows), n_errors=len(errs), note='sbl_* are the target-guarded comparators; adk_standin has the corrected gradient')
    for sc in ('A1', 'R2'):
        rs = [r for r in rows if r['scenario'] == sc]
        if not rs:
            continue
        for r in rs:
            r['cell_key'] = f"{int(r['band'])}Hz-M{r['M']}"
        cells = sorted({r['cell_key'] for r in rs}, key=lambda c: (int(c.split('Hz')[0]), int(c.split('M')[1])))
        T = AV.analyse(rs, 'mesh', cells, len(cells))
        by = defaultdict(list)
        for r in rs:
            by[r['cell_key']].append(r)
        bg = [T[c]['comparisons'][f"{P}-vs-{T[c]['best_generic']}"]['cls'] for c in cells]
        def cnt(key):
            return AV.count(T, cells, key)
        every = 0
        for c in cells:
            ok = True
            for m in AV.GENERIC:
                d = AV.cell_diff(by[c], 'mesh', P, m)
                mu, lo, hi = AV.tint(list(d.values()), AV.ALPHA / len(cells))
                ok &= AV.classify(mu, lo, hi) == 'win'
            every += ok
        s = dict(problems=len(rs), cells=len(cells), units=len({r['source'] for r in rs}),
                 best_generic=dict(wins=bg.count('win'), ties=bg.count('tie'), losses=bg.count('loss')),
                 best_generic_by_cell={c: T[c]['best_generic'] for c in cells},
                 every_member_wins=every,
                 sinc=cnt(f'{P}-vs-sinc_ev'), gsmf=dict(**cnt(f'{P}-vs-gsmf'), pooled=AV.pooled(rs, 'mesh', cells, P, 'gsmf')),
                 mo26_cap5=dict(**cnt(f'{P}-vs-mo26_cap5'), pooled=AV.pooled(rs, 'mesh', cells, P, 'mo26_cap5')),
                 v1=dict(**cnt(f'{P}-vs-v1'), pooled=AV.pooled(rs, 'mesh', cells, P, 'v1')),
                 two_stage_point=dict(**cnt(f'{P}-vs-two_stage_point'), pooled=AV.pooled(rs, 'mesh', cells, P, 'two_stage_point')),
                 two_stage_region=dict(**cnt(f'{P}-vs-two_stage_region'), pooled=AV.pooled(rs, 'mesh', cells, P, 'two_stage_region')),
                 ref_point=cnt(f'{P}-vs-ref_point'), ref_region=cnt(f'{P}-vs-ref_region'),
                 pooled_vs_best_generic_task=AV.pooled(rs, 'mesh', cells, P, 'best_generic_task'),
                 pooled_vs_best_generic_cell=pooled_bg(by, cells, {c: T[c]['best_generic'] for c in cells}),
                 by_band={b: pooled_bg(by, [c for c in cells if c.startswith(b)], {c: T[c]['best_generic'] for c in cells})
                          for b in ('250', '500', '1000')},
                 K_cv_hist=np.bincount([r['info']['j_K']['cv'] for r in rs], minlength=6).tolist(),
                 median_error_db=dict(proposed=float(np.median([r['e'][P] for r in rs])),
                                      best_generic_of_condition=float(np.median([r['e'][T[r['cell_key']]['best_generic']] for r in rs])),
                                      arrival_time_point=float(np.median([r['e']['two_stage_point'] for r in rs]))),
                 cells_table=T)
        res[sc] = s
    a1, r2 = res.get('A1'), res.get('R2')
    res['rules'] = dict(
        G1=dict(rule='A1: beats the best generic estimator in >= 9 of 12 conditions, loses in none',
                passed=bool(a1 and a1['best_generic']['wins'] >= 9 and a1['best_generic']['losses'] == 0)),
        G2=dict(rule='A1: no loss to the sinc kernel', passed=bool(a1 and a1['sinc']['losses'] == 0)),
        G3=dict(rule='A1: pooled superior to the shared five-ball variant', passed=bool(a1 and a1['gsmf']['pooled']['high'] < 0)),
        G4=dict(rule='R2: beats the best generic estimator in >= 9 of 12 conditions, loses in none',
                passed=bool(r2 and r2['best_generic']['wins'] >= 9 and r2['best_generic']['losses'] == 0)))
    res['stop_rule'] = dict(rule='report in the abstract if pooled worse than the best generic estimator in A1',
                            triggered=bool(a1 and a1['pooled_vs_best_generic_cell']['low'] > 0))
    (out_dir / 'summary.json').write_text(json.dumps(res, indent=1, default=float))
    for k, v in res['rules'].items():
        print(k, 'PASS' if v['passed'] else 'FAIL', '-', v['rule'])


def pooled_bg(by, cells, best):
    per = defaultdict(list)
    for c in cells:
        for u, v in AV.cell_diff(by[c], 'mesh', P, best[c]).items():
            per[u].append(v)
    m, lo, hi = AV.tint([np.mean(v) for v in per.values()], AV.ALPHA)
    return dict(mean=m, low=lo, high=hi, units=len(per), cells=len(cells))


if __name__ == '__main__':
    main()
