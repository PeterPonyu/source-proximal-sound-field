"""Registered analysis of the TB-M2 follow-ups (PROTOCOLS/TB-M2-SBLFIX.md, PROTOCOLS/TB-M2-RIB.md).

python3 -B analyze_followups.py TB-M2-SBLFIX|TB-M2-RIB studies/<study>/primary studies/TB-M2/primary
Same metric, units and intervals as analyze_v2.py: per-problem error 10 log10(mean NMSE over the bins used), paired
differences in dB, loudspeakers as units (draws averaged), Student-t intervals with a Bonferroni correction over the
cells of the rule, practical margin 0.1 dB, non-inferiority margin 0.2 dB, pooled 95 % interval over loudspeakers.
A non-finite or > 1e6 error makes that problem's comparison undefined; it is counted and excluded pairwise.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

DELTA, NI, ALPHA = 0.1, 0.2, 0.05
PRIMARY = 'j_cv_ext'


def key(r):
    return (r['source'], r['band'], r['M'], r['draw'])


def err(v, bins=None):
    v = np.asarray(v, float)
    if bins is not None:
        v = v[bins]
    return 10 * np.log10(np.mean(v)) if np.all(np.isfinite(v)) and np.all(v < 1e6) else np.nan


def tint(x, a):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return float(np.mean(x)) if n else np.nan, -np.inf, np.inf
    m = x.mean(); h = stats.t.ppf(1 - a / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(m), float(m - h), float(m + h)


def cls(m, lo, hi):
    return 'win' if (hi < 0 and m < -DELTA) else ('loss' if (lo > 0 and m > DELTA) else 'tie')


def compare(cells, a, b, family):
    out, per, undefined = {}, defaultdict(list), 0
    for c, rs in cells.items():
        by = defaultdict(list)
        for r in rs:
            x, y = r['e'].get(a, np.nan), r['e'].get(b, np.nan)
            if np.isfinite(x) and np.isfinite(y):
                by[r['source']].append(x - y)
            else:
                undefined += 1
        d = {u: float(np.mean(v)) for u, v in by.items()}
        m, lo, hi = tint(list(d.values()), ALPHA / family)
        out[c] = dict(mean=m, low=lo, high=hi, units=len(d), cls=cls(m, lo, hi), non_inferior=bool(hi < NI))
        for u, v in d.items():
            per[u].append(v)
    pm, plo, phi = tint([np.mean(v) for v in per.values()], ALPHA)
    cl = [v['cls'] for v in out.values()]
    return dict(per_cell=out, wins=cl.count('win'), ties=cl.count('tie'), losses=cl.count('loss'),
                non_inferior=sum(v['non_inferior'] for v in out.values()), undefined_problems=undefined,
                pooled=dict(mean=pm, low=plo, high=phi))


def main():
    study, out, m2 = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    reg = {key(r): r for r in map(json.loads, open(m2 / 'rows.jsonl')) if r.get('kind') == 'mesh' and 'error' not in r}
    new = [json.loads(l) for l in open(out / 'rows.jsonl')]
    errs = [r for r in new if 'error' in r]; new = [r for r in new if 'error' not in r]
    res = dict(study=study, rows=len(new), errors=len(errs))
    cells = defaultdict(list)
    if study == 'TB-M2-SBLFIX':
        G = ('sinc_ev', 'rbf', 'adk_standin', 'sbl_em_guard', 'sbl_mackay_guard')
        for r in new:
            base = reg[key(r)]
            e = {m: err(v) for m, v in base['nmse'].items()}
            e.update({m: err(v) for m, v in r['nmse'].items()})
            cells[f"{int(r['band'])}Hz-M{r['M']}"].append(dict(source=r['source'], e=e))
        res['nonfinite_guarded'] = {m: int(sum(not np.isfinite(t['e'][m]) for rs in cells.values() for t in rs)) for m in ('sbl_em_guard', 'sbl_mackay_guard')}
        best = {}
        for c, rs in cells.items():
            with np.errstate(all='ignore'):
                mean = {m: np.nanmean([t['e'][m] for t in rs]) for m in G}
            best[c] = min((m for m in G if np.isfinite(mean[m])), key=lambda m: mean[m])
            for t in rs:
                t['e']['best_generic_cell'] = t['e'][best[c]]
                t['e']['best_generic_task'] = np.nanmin([t['e'][m] for m in G])
        c1 = compare(cells, PRIMARY, 'best_generic_cell', len(cells))
        res['M1_corrected'] = dict(rule='primary beats the best generic estimator, with the corrected sparse Bayesian comparators, in >= 12 of 16 MeshRIR cells and loses in none',
                                   best_generic=best, **{k: v for k, v in c1.items() if k != 'per_cell'}, per_cell=c1['per_cell'],
                                   passed=bool(c1['wins'] >= 12 and c1['losses'] == 0))
        res['pooled_vs_task_best_generic'] = compare(cells, PRIMARY, 'best_generic_task', len(cells))['pooled']
        for m in ('sbl_em_guard', 'sbl_mackay_guard'):
            res[f'primary_vs_{m}'] = {k: v for k, v in compare(cells, PRIMARY, m, len(cells)).items() if k != 'per_cell'}
    else:
        for r in new:
            base = reg[key(r)]; b = r['bin']
            e = {m: err(v, [b]) for m, v in base['nmse'].items()}
            e['ribeiro_official'] = err(r['nmse']['ribeiro_official'])
            cells[f"{int(r['band'])}Hz-M{r['M']}"].append(dict(source=r['source'], e=e))
        # amendment 1 (PROTOCOLS/TB-M2-RIB-AMENDMENT-1.md): pooled superiority only; per-cell intervals are descriptive
        res['R1'] = dict(rule='primary pooled superior to the official adaptive kernel (95% Student-t over the 6 loudspeakers, upper bound < 0); per-cell intervals descriptive',
                         **(c := compare(cells, PRIMARY, 'ribeiro_official', len(cells))))
        res['R1']['passed'] = bool(c['pooled']['high'] < 0)
        res['R2_standin_vs_official'] = {k: v for k, v in compare(cells, 'adk_standin', 'ribeiro_official', len(cells)).items() if k != 'per_cell'}
        res['official_vs_sinc'] = {k: v for k, v in compare(cells, 'ribeiro_official', 'sinc_ev', len(cells)).items() if k != 'per_cell'}
        res['fit_seconds'] = dict(median=float(np.median([r['fit_seconds'] for r in new])) if new else None,
                                  max=float(np.max([r['fit_seconds'] for r in new])) if new else None)
    (out / 'summary.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ('M1_corrected', 'R1')}, indent=1, default=float)[:3000])
    for k in ('M1_corrected', 'R1'):
        if k in res:
            print(k, 'PASS' if res[k]['passed'] else 'FAIL', {kk: res[k][kk] for kk in ('wins', 'ties', 'losses', 'non_inferior', 'pooled')})


if __name__ == '__main__':
    main()
