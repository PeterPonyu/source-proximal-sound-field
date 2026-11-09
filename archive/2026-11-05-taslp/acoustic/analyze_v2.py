"""Registered analysis of TB-S2 and TB-M2 (PROTOCOLS/TB-S2.md, PROTOCOLS/TB-M2.md).

python3 -B analyze_v2.py TB-S2|TB-M2 OUTPUT_DIR   ->  summary.json and cell-table.csv in OUTPUT_DIR

Per task, a method's error is 10 log10 of its mean NMSE over the bins of the band. A paired difference is the
difference of two such errors (dB; negative favours the first method). The independent units are fields (TB-S2,
the 32 seeds of a cell) and loudspeakers (TB-M2 MeshRIR: the two draws of a loudspeaker are averaged; held-out
S4/S7/S8: loudspeaker-draw pairs, descriptive only).
Per cell, a Student-t interval on the mean paired difference is computed at the family-wise level 0.05 with a
Bonferroni correction over the cells of the rule's family. With the practical margin DELTA = 0.1 dB:
  win  = upper bound < 0 and mean < -DELTA;  loss = lower bound > 0 and mean > +DELTA;  tie otherwise;
  non-inferior = upper bound < NI_MARGIN = 0.2 dB.
Pooled comparisons average the cell means within each independent unit that all cells share (TB-S2: seed;
TB-M2: loudspeaker) and take a 95 % Student-t interval over the units.
"""
from __future__ import annotations
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

DELTA, NI_MARGIN, ALPHA = 0.1, 0.2, 0.05
GENERIC = ('sinc_ev', 'rbf', 'adk_standin', 'sbl_em_common', 'sbl_mackay_common')
PRIMARY, SECONDARY = 'j_cv_ext', 'fj_cv_ext'
FAR = 'A1-beside1.2'


def db(v):
    return float(10 * np.log10(np.mean(v)))


def tint(x, alpha):
    x = np.asarray(x, float); n = len(x)
    m = float(x.mean())
    if n < 2:
        return m, -np.inf, np.inf
    h = float(stats.t.ppf(1 - alpha / 2, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return m, m - h, m + h


def classify(m, lo, hi):
    return 'win' if (hi < 0 and m < -DELTA) else ('loss' if (lo > 0 and m > DELTA) else 'tie')


def load(out):
    rows = [json.loads(l) for l in open(Path(out) / 'rows.jsonl')]
    ok = [r for r in rows if 'error' not in r]
    for r in ok:
        r['e'] = {m: db(v) for m, v in r['nmse'].items()}
        r['e']['best_generic_task'] = min(r['e'][m] for m in GENERIC)
    return ok, [r for r in rows if 'error' in r]


def units(rs, kind):
    """Group a cell's rows into independent units: dict unit -> list of rows."""
    g = defaultdict(list)
    for r in rs:
        g[r['seed'] if kind == 'sim' else (r['source'] if kind == 'mesh' else (r['source'], r['draw']))].append(r)
    return g


def cell_diff(rs, kind, a, b):
    """Per-unit mean paired difference a - b (dB)."""
    return {u: float(np.mean([r['e'][a] - r['e'][b] for r in v])) for u, v in units(rs, kind).items()}


def analyse(rows, kind, cells, family_n):
    alpha = ALPHA / family_n
    by = defaultdict(list)
    for r in rows:
        by[r['cell_key']].append(r)
    table = {}
    for c in cells:
        rs = by[c]
        methods = sorted(set.intersection(*[set(r['e']) for r in rs]))
        mean = {m: float(np.mean([r['e'][m] for r in rs])) for m in methods}
        bg = min(GENERIC, key=lambda m: mean[m])
        comp = {}
        for a, b in [(PRIMARY, bg), (PRIMARY, 'gsmf'), (PRIMARY, 'mo26_cap5'), (PRIMARY, 'sinc_ev'), (PRIMARY, 'v1'),
                     (PRIMARY, 'hybrid_v1'), (PRIMARY, 'j_cv'), (PRIMARY, 'ind_cv_ext'), (PRIMARY, 'j_sw_ext'), ('j_cv', 'j_K1'),
                     ('j_sw', 'j_K1'), (PRIMARY, 'ref_point'), (PRIMARY, 'ref_region')] + \
                [(PRIMARY, m) for m in ('two_stage_point', 'two_stage_region') if m in methods] + \
                ([(SECONDARY, bg), (SECONDARY, PRIMARY), (SECONDARY, 'gsmf')] if SECONDARY in methods else []):
            if a in methods and b in methods:
                d = cell_diff(rs, kind, a, b)
                m_, lo, hi = tint(list(d.values()), alpha)
                comp[f'{a}-vs-{b}'] = dict(mean=m_, low=lo, high=hi, units=len(d), cls=classify(m_, lo, hi), non_inferior=bool(hi < NI_MARGIN))
        Kc = [r['info']['j_K']['cv'] for r in rs]
        table[c] = dict(n=len(rs), best_generic=bg, mean_db=mean, comparisons=comp, K_cv_hist=np.bincount(Kc, minlength=6).tolist(),
                        cap_bound_fraction=float(np.mean([k == 5 for k in Kc])))
    return table


def pooled(rows, kind, cells, a, b):
    """Units shared by all cells: per unit, mean over cells of the per-unit cell differences; 95 % t interval."""
    by = defaultdict(list)
    for r in rows:
        by[r['cell_key']].append(r)
    per = defaultdict(list)
    for c in cells:
        for u, v in cell_diff(by[c], kind, a, b).items():
            per[u].append(v)
    vals = [np.mean(v) for v in per.values()]
    m, lo, hi = tint(vals, ALPHA)
    return dict(mean=m, low=lo, high=hi, units=len(vals))


def count(table, cells, key):
    cl = [table[c]['comparisons'][key]['cls'] for c in cells if key in table[c]['comparisons']]
    return dict(wins=cl.count('win'), ties=cl.count('tie'), losses=cl.count('loss'), cells=len(cl))


def main():
    study, out = sys.argv[1], Path(sys.argv[2])
    rows, errs = load(out)
    res = dict(study=study, n_rows=len(rows), n_errors=len(errs), delta_db=DELTA, ni_margin_db=NI_MARGIN, alpha=ALPHA)
    if study == 'TB-S2':
        for r in rows:
            r['cell_key'] = r['cell']
        cells = list(dict.fromkeys(r['cell'] for r in rows))
        prox = [c for c in cells if c != FAR]
        T = analyse(rows, 'sim', cells, len(prox))
        bgk = lambda c: f"{PRIMARY}-vs-{T[c]['best_generic']}"
        s1 = [T[c]['comparisons'][bgk(c)]['cls'] for c in prox]
        R = {}
        R['S1'] = dict(rule='primary beats the best source-free or sparse estimator in >= 90% of source-proximal cells and loses in none',
                       wins=s1.count('win'), ties=s1.count('tie'), losses=s1.count('loss'), cells=len(prox),
                       passed=bool(s1.count('win') >= 0.9 * len(prox) and s1.count('loss') == 0))
        s2 = {}
        for comp in ('gsmf', 'mo26_cap5'):
            k = f'{PRIMARY}-vs-{comp}'
            ni = sum(T[c]['comparisons'][k]['non_inferior'] for c in prox)
            pl = pooled(rows, 'sim', prox, PRIMARY, comp)
            s2[comp] = dict(non_inferior=ni, **count(T, prox, k), pooled=pl, passed=bool(ni == len(prox) and pl['high'] < 0))
        R['S2'] = dict(rule='primary non-inferior (upper bound < +0.2 dB) to the group-sparse and the cap-5 region kernel in every source-proximal cell, and pooled superior to each',
                       per_competitor=s2, passed=bool(all(v['passed'] for v in s2.values())))
        two = ('A6-second0.6', 'A6-second1.0')
        T2 = analyse(rows, 'sim', list(two), 2)
        R['S3'] = dict(rule='cross-validated order beats one point in both two-source cells (points, shared coordinates)',
                       per_cell={c: T2[c]['comparisons']['j_cv-vs-j_K1'] for c in two},
                       pooled=pooled(rows, 'sim', prox, 'j_cv', 'j_K1'),
                       passed=bool(all(T2[c]['comparisons']['j_cv-vs-j_K1']['cls'] == 'win' for c in two)))
        a4 = count(T, prox, f'{PRIMARY}-vs-j_cv'); p4 = pooled(rows, 'sim', prox, PRIMARY, 'j_cv')
        R['S4'] = dict(rule='extension loses in no source-proximal cell and is pooled superior to points', **a4, pooled=p4,
                       passed=bool(a4['losses'] == 0 and p4['high'] < 0))
        a5 = count(T, prox, f'{PRIMARY}-vs-ind_cv_ext'); p5 = pooled(rows, 'sim', prox, PRIMARY, 'ind_cv_ext')
        R['S5'] = dict(rule='shared coordinates win in >= half of the source-proximal cells, lose in none, and are pooled superior to per-bin fitting',
                       **a5, pooled=p5, passed=bool(a5['wins'] >= 0.5 * len(prox) and a5['losses'] == 0 and p5['high'] < 0))
        T6 = analyse(rows, 'sim', cells, len(cells))
        a6 = count(T6, cells, f'{PRIMARY}-vs-sinc_ev')
        R['S6'] = dict(rule='primary never loses to the source-free sinc kernel (all cells, far control included)', **a6, passed=bool(a6['losses'] == 0))
        p7 = pooled(rows, 'sim', prox, PRIMARY, 'v1')
        R['S7'] = dict(rule='primary pooled superior to version 1', **count(T, prox, f'{PRIMARY}-vs-v1'), pooled=p7, passed=bool(p7['high'] < 0))
        R['descriptive'] = dict(schwarz_vs_cv=count(T, prox, f'{PRIMARY}-vs-j_sw_ext'), hybrid=count(T, prox, f'{PRIMARY}-vs-hybrid_v1'),
                                pooled_vs_hybrid=pooled(rows, 'sim', prox, PRIMARY, 'hybrid_v1'),
                                cap_bound_fraction_median=float(np.median([T[c]['cap_bound_fraction'] for c in prox])))
        losses_named = max(s2[c]['losses'] for c in s2)
        R['changes_claims'] = (['S1'] if not R['S1']['passed'] else []) + \
            (['primary loses to a named competitor in more than half of the source-proximal cells'] if losses_named > len(prox) / 2 else []) + \
            [f'pooled loss to {c}' for c in s2 if s2[c]['pooled']['low'] > 0]
        res.update(rules=R, cells=T)
    else:
        mesh = [r for r in rows if r['kind'] == 'mesh']; held = [r for r in rows if r['kind'] == 'held']
        for r in rows:
            r['cell_key'] = f"{int(r['band'])}Hz-M{r['M']}"
        cells = sorted({r['cell_key'] for r in mesh}, key=lambda c: (int(c.split('Hz')[0]), int(c.split('M')[1])))
        T = analyse(mesh, 'mesh', cells, len(cells))
        bgk = lambda c: f"{PRIMARY}-vs-{T[c]['best_generic']}"
        m1 = [T[c]['comparisons'][bgk(c)]['cls'] for c in cells]
        R = {}
        R['M1'] = dict(rule='primary beats the best source-free or sparse estimator in >= 12 of the 16 MeshRIR cells and loses in none',
                       wins=m1.count('win'), ties=m1.count('tie'), losses=m1.count('loss'), passed=bool(m1.count('win') >= 12 and m1.count('loss') == 0))
        m2 = {}
        for comp in ('gsmf', 'mo26_cap5'):
            k = f'{PRIMARY}-vs-{comp}'
            ni = sum(T[c]['comparisons'][k]['non_inferior'] for c in cells)
            pl = pooled(mesh, 'mesh', cells, PRIMARY, comp)
            m2[comp] = dict(non_inferior=ni, **count(T, cells, k), pooled=pl, passed=bool(ni == len(cells) and pl['high'] < 0))
        R['M2'] = dict(rule='primary non-inferior to the group-sparse and the cap-5 region kernel in every MeshRIR cell, and pooled superior to each',
                       per_competitor=m2, passed=bool(all(v['passed'] for v in m2.values())))
        a3 = count(T, cells, f'{PRIMARY}-vs-sinc_ev')
        R['M3'] = dict(rule='primary never loses to the sinc kernel on MeshRIR', **a3, passed=bool(a3['losses'] == 0))
        low = [c for c in cells if c.startswith(('125', '250'))]
        m4 = [T[c]['comparisons'][f"{SECONDARY}-vs-{T[c]['best_generic']}"]['cls'] for c in low if f"{SECONDARY}-vs-{T[c]['best_generic']}" in T[c]['comparisons']]
        R['M4'] = dict(rule='secondary: remainder-family variant beats the best source-free or sparse estimator in every 125-250 Hz MeshRIR cell',
                       wins=m4.count('win'), ties=m4.count('tie'), losses=m4.count('loss'), cells=len(low), passed=bool(len(m4) == len(low) and m4.count('win') == len(low)))
        R['descriptive'] = dict(extension=count(T, cells, f'{PRIMARY}-vs-j_cv'), shared_coordinates=count(T, cells, f'{PRIMARY}-vs-ind_cv_ext'),
                                schwarz_vs_cv=count(T, cells, f'{PRIMARY}-vs-j_sw_ext'), v1=count(T, cells, f'{PRIMARY}-vs-v1'),
                                hybrid=count(T, cells, f'{PRIMARY}-vs-hybrid_v1'),
                                two_stage_point=count(T, cells, f'{PRIMARY}-vs-two_stage_point'),
                                pooled_vs_v1=pooled(mesh, 'mesh', cells, PRIMARY, 'v1'),
                                pooled_vs_best_generic_task=pooled(mesh, 'mesh', cells, PRIMARY, 'best_generic_task'))
        # stop rule: best comparator of a cell = lowest mean among the generic set and the two named competitors
        stop = []
        for c in cells:
            best = min(GENERIC + ('gsmf', 'mo26_cap5'), key=lambda m: T[c]['mean_db'][m])
            k = f'{PRIMARY}-vs-{best}'
            stop.append(T[c]['comparisons'][k]['cls'] if k in T[c]['comparisons'] else
                        classify(*tint(list(cell_diff([r for r in mesh if r['cell_key'] == c], 'mesh', PRIMARY, best).values()), ALPHA / len(cells))))
        pl = pooled(mesh, 'mesh', cells, PRIMARY, 'best_generic_task')
        R['stop_rule'] = dict(rule='ask the user if the primary loses to the best comparator in more than 4 of 16 cells or in the pooled comparison',
                              cell_losses=stop.count('loss'), triggered=bool(stop.count('loss') > 4 or pl['low'] > 0))
        H = {}
        if held:
            hcells = sorted({r['cell_key'] for r in held}, key=lambda c: (int(c.split('Hz')[0]), int(c.split('M')[1])))
            H = analyse(held, 'held', hcells, len(hcells))
            R['held_out_same_recordings'] = dict(note='descriptive only: the S4/S7/S8 recordings were seen in TB-M1 and in development',
                                                 vs_best_generic={c: H[c]['comparisons'][f"{PRIMARY}-vs-{H[c]['best_generic']}"]['cls'] for c in hcells},
                                                 vs_gsmf=count(H, hcells, f'{PRIMARY}-vs-gsmf'), vs_mo26_cap5=count(H, hcells, f'{PRIMARY}-vs-mo26_cap5'))
        res.update(rules=R, cells=T, held_out_cells=H)
    (out / 'summary.json').write_text(json.dumps(res, indent=1, default=float))
    with open(out / 'cell-table.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        meth = (PRIMARY, 'j_cv', 'ind_cv_ext', 'j_sw_ext', 'gsmf', 'mo26_cap5', 'hybrid_v1', 'v1', 'sinc_ev', 'ref_point')
        w.writerow(['cell', 'n', 'best_generic'] + [f'mean_db_{m}' for m in meth] + ['mean_db_best_generic'])
        for c, t in res['cells'].items():
            w.writerow([c, t['n'], t['best_generic']] + [round(t['mean_db'].get(m, float('nan')), 3) for m in meth] +
                       [round(t['mean_db'][t['best_generic']], 3)])
    for k, v in res['rules'].items():
        if isinstance(v, dict) and 'passed' in v:
            print(k, 'PASS' if v['passed'] else 'FAIL', '-', v['rule'])
    print('changes_claims' if study == 'TB-S2' else 'stop_rule', res['rules'].get('changes_claims', res['rules'].get('stop_rule')))


if __name__ == '__main__':
    main()
