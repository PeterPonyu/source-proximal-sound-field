"""Descriptive companion to the registered analyses (labelled; not a registered rule): the pooled margins of M1 as
registered and of M1-corrected under both definitions of the best generic estimator, with the registered functions
of analyze_followups.py.

python3 -B studies/TB-M2-SBLFIX/m1_side_by_side.py  ->  studies/TB-M2-SBLFIX/m1-side-by-side.json

Per-cell best: the member of the generic set with the lowest mean error in the cell; a member with any non-finite
error in the cell has no mean there (the registered TB-M2 definition). Per-problem best: the lowest finite error of the
set on each problem. Units, intervals and pairwise exclusion as in analyze_followups.py.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import analyze_followups as AF  # noqa: E402

SETS = {'M1 (registered generic set)': ('sinc_ev', 'rbf', 'adk_standin', 'sbl_em_common', 'sbl_mackay_common'),
        'M1-corrected': ('sinc_ev', 'rbf', 'adk_standin', 'sbl_em_guard', 'sbl_mackay_guard')}


def main():
    reg = {AF.key(r): r for r in map(json.loads, open(HERE / 'studies/TB-M2/primary/rows.jsonl')) if r.get('kind') == 'mesh' and 'error' not in r}
    fix = {AF.key(r): r for r in map(json.loads, open(HERE / 'studies/TB-M2-SBLFIX/primary/rows.jsonl')) if 'error' not in r}
    res = {}
    for name, G in SETS.items():
        cells = defaultdict(list)
        for k, r in reg.items():
            e = {m: AF.err(v) for m, v in r['nmse'].items()}
            e.update({m: AF.err(v) for m, v in fix[k]['nmse'].items()})
            cells[f"{int(r['band'])}Hz-M{r['M']}"].append(dict(source=r['source'], e=e))
        best = {}
        for c, rs in cells.items():
            mean = {m: float(np.mean([t['e'][m] for t in rs])) for m in G}
            best[c] = min((m for m in G if np.isfinite(mean[m])), key=lambda m: mean[m])
            for t in rs:
                t['e']['best_cell'] = t['e'][best[c]]
                fin = [t['e'][m] for m in G if np.isfinite(t['e'][m])]
                t['e']['best_problem'] = min(fin) if fin else np.nan
        pc = AF.compare(cells, AF.PRIMARY, 'best_cell', len(cells))
        pp = AF.compare(cells, AF.PRIMARY, 'best_problem', len(cells))
        res[name] = dict(best_generic_per_cell=best, wins=pc['wins'], ties=pc['ties'], losses=pc['losses'],
                         pooled_vs_per_cell_best=pc['pooled'], pooled_vs_per_problem_best=pp['pooled'],
                         undefined_problems_per_cell_best=pc['undefined_problems'])
    out = HERE / 'studies/TB-M2-SBLFIX/m1-side-by-side.json'
    out.write_text(json.dumps(res, indent=1, default=float))
    for name, v in res.items():
        a, b = v['pooled_vs_per_cell_best'], v['pooled_vs_per_problem_best']
        print(f"{name}: {v['wins']}/{v['ties']}/{v['losses']}; per-cell best {a['mean']:+.2f} [{a['low']:+.2f}, {a['high']:+.2f}];"
              f" per-problem best {b['mean']:+.2f} [{b['low']:+.2f}, {b['high']:+.2f}]")


if __name__ == '__main__':
    main()
