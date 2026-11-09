"""Registered analysis for TB-S1, amendment 1 (PROTOCOLS/TB-S1-AMENDMENT-1.md): identical to analyze_sim.py except that
the strings True/False written by the csv module are parsed as booleans. Rules H1-H6 of PROTOCOLS/TB-S1.md.

python3 -B analyze_sim_r1.py studies/TB-S1/primary
Writes summary.json and cell-table.csv next to field-metrics.csv; every registered rule is reported pass or fail.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
import numpy as np

GENERIC = ('sinc_ev', 'sinc_loo', 'rbf', 'adk', 'sbl_mackay', 'sbl_em')
FAR = ('A1-beside1.2',)
BOOT, BOOT_SEED = 4000, 20261105
NONINF = 0.10


def load(path):
    rows = []
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            for k, v in list(r.items()):
                if v in ('', None):
                    r[k] = None
                    continue
                if v in ('True', 'False'):          # amendment 1: csv writes Python booleans as these strings
                    r[k] = v == 'True'
                    continue
                try:
                    r[k] = json.loads(v)
                except (ValueError, TypeError):
                    pass
            rows.append(r)
    return rows


def boot(d):
    d = np.asarray(d, float)
    rng = np.random.default_rng(BOOT_SEED)
    m = d[rng.integers(0, len(d), (BOOT, len(d)))].mean(axis=1)
    return dict(mean=float(d.mean()), low=float(np.quantile(m, 0.025)), high=float(np.quantile(m, 0.975)), n=len(d))


def cls(b):
    return 'win' if b['high'] < 0 else ('loss' if b['low'] > 0 else 'tie')


def paired(rows, a, b):
    return boot([r[f'nmse_{a}'] - r[f'nmse_{b}'] for r in rows])


def main():
    out_dir = Path(sys.argv[1])
    rows = load(out_dir / 'field-metrics.csv')
    cells = list(dict.fromkeys(r['cell'] for r in rows))
    by = {c: [r for r in rows if r['cell'] == c] for c in cells}
    prox = [c for c in cells if c not in FAR]
    methods = sorted(k[5:] for k in rows[0] if k.startswith('nmse_') and k != 'nmse_remainder_sinc')
    table, res = [], {}
    for c in cells:
        rr = by[c]
        mean = {m: float(np.mean([r[f'nmse_{m}'] for r in rr])) for m in methods}
        best_g = min(GENERIC, key=lambda m: mean[m])
        row = dict(cell=c, n=len(rr), f=rr[0]['f'], M=rr[0]['M'], best_generic=best_g,
                   **{f'mean_{m}': mean[m] for m in methods},
                   **{f'median_{m}': float(np.median([r[f'nmse_{m}'] for r in rr])) for m in ('prop_bic', 'prop_cv', 'mo26', best_g)},
                   K_bic=np.bincount([r['K_bic'] for r in rr], minlength=4).tolist(),
                   K_cv=np.bincount([r['K_cv'] for r in rr], minlength=4).tolist(),
                   k1_outside_region=float(np.mean([bool(r['k1_outside_region']) for r in rr])),
                   k1_outside_room=float(np.mean([bool(r['k1_outside_room']) for r in rr])),
                   sel_outside_room=float(np.mean([bool(r['sel_outside_room']) for r in rr])),
                   k1_loc_error_m_median=float(np.median([r['k1_loc_error_m'] for r in rr])))
        if rr[0].get('nmse_remainder_sinc') is not None:
            row['mean_remainder_sinc'] = float(np.mean([r['nmse_remainder_sinc'] for r in rr]))
        cmp = {}
        for a, b in (('prop_bic', best_g), ('prop_cv', best_g), ('prop_bic', 'mo26'), ('prop_cv', 'mo26'),
                     ('prop_bic', 'prop_k1'), ('prop_cv', 'prop_k1'), ('prop_k1', 'prop_k1_menu'),
                     ('prop_bic', 'prop_bic_room'), ('prop_bic', 'sinc_ev'), ('prop_bic', 'oracle'), ('mo26', best_g),
                     ('prop_bic', 'mo26_greedy'), ('mo26_greedy', 'mo26'), ('prop_bic', 'mo25')):
            bb = paired(rr, a, b); bb['class'] = cls(bb); cmp[f'{a}-vs-{b}'] = bb
        row['comparisons'] = cmp
        table.append(row)
    T = {r['cell']: r for r in table}
    # H1
    h1 = {c: T[c]['comparisons'][f"prop_bic-vs-{T[c]['best_generic']}"]['class'] for c in prox}
    h1cv = {c: T[c]['comparisons'][f"prop_cv-vs-{T[c]['best_generic']}"]['class'] for c in prox}
    wins = sum(v == 'win' for v in h1.values()); losses = sum(v == 'loss' for v in h1.values())
    res['H1'] = dict(rule='prop_bic beats the best source-free/sparse comparator in >= 90% of source-proximal cells, loses in none',
                     wins=wins, ties=sum(v == 'tie' for v in h1.values()), losses=losses, cells=len(prox),
                     passed=bool(wins >= 0.9 * len(prox) and losses == 0), per_cell=h1,
                     prop_cv=dict(wins=sum(v == 'win' for v in h1cv.values()), losses=sum(v == 'loss' for v in h1cv.values())))
    # H2
    h2 = {}
    for c in prox:
        b = T[c]['comparisons']['prop_bic-vs-mo26']
        h2[c] = dict(non_inferior=bool(b['high'] < NONINF * T[c]['mean_mo26']), cls=b['class'])
    res['H2'] = dict(rule='prop_bic non-inferior to the multiple-region kernel (upper bound < 0.10 x its mean) in every source-proximal cell',
                     non_inferior=sum(v['non_inferior'] for v in h2.values()), cells=len(prox),
                     wins=sum(v['cls'] == 'win' for v in h2.values()), ties=sum(v['cls'] == 'tie' for v in h2.values()),
                     losses=sum(v['cls'] == 'loss' for v in h2.values()),
                     passed=bool(all(v['non_inferior'] for v in h2.values())), per_cell=h2)
    # H3 (kill criterion for model-order selection)
    two = ('A6-second0.6', 'A6-second1.0')
    h3b = all(T[c]['comparisons']['prop_bic-vs-prop_k1']['class'] == 'win' for c in two)
    h3c = all(T[c]['comparisons']['prop_cv-vs-prop_k1']['class'] == 'win' for c in two)
    allk = [T[c]['comparisons']['prop_bic-vs-prop_k1']['class'] for c in prox]
    res['H3'] = dict(rule='K-selection beats K = 1 in both two-source cells (by evidence or by cross-validation); otherwise drop (a)',
                     bic_passed=bool(h3b), cv_passed=bool(h3c), passed=bool(h3b or h3c),
                     two_source={c: {k: T[c]['comparisons'][k] for k in ('prop_bic-vs-prop_k1', 'prop_cv-vs-prop_k1')} for c in two},
                     all_cells=dict(wins=allk.count('win'), ties=allk.count('tie'), losses=allk.count('loss')))
    # H4 (continuous tuning)
    allm = [T[c]['comparisons']['prop_k1-vs-prop_k1_menu']['class'] for c in cells]
    b4 = T['base']['comparisons']['prop_k1-vs-prop_k1_menu']
    res['H4'] = dict(rule='continuous tuning beats the finite menus at K = 1 in the base cell', base=b4, passed=bool(b4['class'] == 'win'),
                     all_cells=dict(wins=allm.count('win'), ties=allm.count('tie'), losses=allm.count('loss')))
    # H5 (unconstrained search and image absorption)
    o0, o180 = T['A3-phi0-h0.05']['k1_outside_room'], T['A3-phi180-h0.05']['k1_outside_room']
    h5c = {c: T[c]['comparisons']['prop_bic-vs-prop_bic_room']['class'] for c in ('base', 'A2-f1000-M96', 'A2-f1000-M192')}
    res['H5'] = dict(rule='(i) K = 1 outside-room fraction >= 0.5 for an in-phase floor at 0.05 wavelengths and <= 0.1 for a phase-inverting floor; '
                          '(ii) the unconstrained search beats the room-restricted one in at least one of base, 1 kHz x 96, 1 kHz x 192 and loses in none',
                     outside_room_phi0=o0, outside_room_phi180=o180, i_passed=bool(o0 >= 0.5 and o180 <= 0.1),
                     ii=h5c, ii_passed=bool(list(h5c.values()).count('win') >= 1 and 'loss' not in h5c.values()))
    res['H5']['passed'] = bool(res['H5']['ii_passed'])     # (c) is claimed on (ii); (i) is mechanism evidence, reported either way
    # H6 (safe default)
    h6 = {c: T[c]['comparisons']['prop_bic-vs-sinc_ev']['class'] for c in cells}
    res['H6'] = dict(rule='the source-aware pipeline never loses to the source-free kernel (every cell, including the far control)',
                     losses=[c for c, v in h6.items() if v == 'loss'], wins=list(h6.values()).count('win'),
                     passed=bool('loss' not in h6.values()))
    # fractional factorial main effects (descriptive)
    ff = [c for c in cells if c.startswith('FF-')]
    if ff:
        import sim  # noqa: E402  (factor signs)
        signs = {c: sim.cells()[c]['ff_signs'] for c in ff}
        y1 = {c: float(np.mean([np.log10(r['nmse_prop_bic']) for r in by[c]])) for c in ff}
        y2 = {c: float(np.mean([np.log10(r['nmse_prop_bic'] / r[f"nmse_{T[c]['best_generic']}"]) for r in by[c]])) for c in ff}
        y3 = {c: float(np.mean([np.log10(r['nmse_prop_bic'] / r['nmse_mo26']) for r in by[c]])) for c in ff}
        eff = {}
        for fac in 'ABCDEFGH':
            hi = [c for c in ff if signs[c][fac] > 0]; lo = [c for c in ff if signs[c][fac] < 0]
            eff[fac] = dict(log10_nmse=float(np.mean([y1[c] for c in hi]) - np.mean([y1[c] for c in lo])),
                            log10_ratio_to_best_generic=float(np.mean([y2[c] for c in hi]) - np.mean([y2[c] for c in lo])),
                            log10_ratio_to_mo26=float(np.mean([y3[c] for c in hi]) - np.mean([y3[c] for c in lo])))
        res['fractional_factorial_main_effects'] = eff
    res['changes_claims'] = (['H1'] if not res['H1']['passed'] else []) + \
        (['H2: prop_bic loses to mo26 in more than half of the source-proximal cells'] if res['H2']['losses'] > len(prox) / 2 else [])
    g2 = [T[c]['comparisons']['prop_bic-vs-mo26_greedy']['class'] for c in prox]
    res['H2']['against_mo26_greedy'] = dict(wins=g2.count('win'), ties=g2.count('tie'), losses=g2.count('loss'))
    (out_dir / 'summary.json').write_text(json.dumps(dict(rules=res, cells=table), indent=1, default=float))
    keys = ['cell', 'n', 'f', 'M', 'best_generic'] + [f'mean_{m}' for m in methods] + ['k1_outside_room', 'sel_outside_room', 'mean_remainder_sinc']
    with open(out_dir / 'cell-table.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
        w.writeheader(); [w.writerow(r) for r in table]
    for h in ('H1', 'H2', 'H3', 'H4', 'H5', 'H6'):
        print(h, 'PASS' if res[h]['passed'] else 'FAIL', '-', res[h]['rule'])


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent / 'common'))
    main()
