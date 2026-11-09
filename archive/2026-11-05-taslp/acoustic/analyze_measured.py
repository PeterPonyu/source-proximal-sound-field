"""Registered analysis for TB-M1 (rules M1-M4 and the stop rule of PROTOCOLS/TB-M1.md).

python3 -B analyze_measured.py studies/TB-M1/primary
Cells: octave band x microphone count (16), pooled over the three loudspeakers. Uncertainty: cluster bootstrap over
(loudspeaker, frequency bin) clusters, the four microphone draws averaged inside a cluster (4000 resamples).
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
import numpy as np

GENERIC = ('sinc_ev', 'sinc_loo', 'rbf', 'adk', 'sbl_mackay', 'sbl_em')
BOOT, BOOT_SEED, NONINF = 4000, 20261106, 0.10


def load(path):
    rows = []
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            for k, v in list(r.items()):
                if v in ('True', 'False'):          # csv writes Python booleans as these strings
                    r[k] = v == 'True'
                    continue
                try:
                    r[k] = json.loads(v)
                except (ValueError, TypeError):
                    pass
            rows.append(r)
    return rows


def cluster_boot(rows, a, b):
    keys = sorted({(r['source'], r['bin']) for r in rows})
    d = np.array([np.mean([r[f'nmse_{a}'] - r[f'nmse_{b}'] for r in rows if (r['source'], r['bin']) == k]) for k in keys])
    rng = np.random.default_rng(BOOT_SEED)
    m = d[rng.integers(0, len(d), (BOOT, len(d)))].mean(axis=1)
    return dict(mean=float(d.mean()), low=float(np.quantile(m, 0.025)), high=float(np.quantile(m, 0.975)), clusters=len(d))


def cls(b):
    return 'win' if b['high'] < 0 else ('loss' if b['low'] > 0 else 'tie')


def main():
    out = Path(sys.argv[1])
    rows = load(out / 'field-metrics.csv')
    methods = sorted(k[5:] for k in rows[0] if k.startswith('nmse_'))
    cells = sorted({(r['band'], r['M']) for r in rows})
    table, T = [], {}
    for band, M in cells:
        rr = [r for r in rows if r['band'] == band and r['M'] == M]
        mean = {m: float(np.mean([r[f'nmse_{m}'] for r in rr])) for m in methods}
        best_g = min(GENERIC, key=lambda m: mean[m])
        best_any = min(GENERIC + ('mo26',), key=lambda m: mean[m])
        cmp = {}
        for a, b in (('prop_bic', best_g), ('prop_cv', best_g), ('prop_bic', 'mo26'), ('prop_bic', best_any), ('prop_bic', 'sinc_ev'),
                     ('prop_bic', 'prop_k1'), ('prop_cv', 'prop_k1'), ('prop_k1', 'prop_k1_menu'), ('prop_bic', 'mo26_greedy'),
                     ('prop_bic', 'oracle'), ('prop_bic', 'mo25')):
            bb = cluster_boot(rr, a, b); bb['class'] = cls(bb); cmp[f'{a}-vs-{b}'] = bb
        per_source = {s: {m: float(np.mean([r[f'nmse_{m}'] for r in rr if r['source'] == s])) for m in methods} for s in ('S4', 'S7', 'S8')}
        row = dict(band=band, M=M, n=len(rr), best_generic=best_g, best_comparator=best_any, **{f'mean_{m}': mean[m] for m in methods},
                   K_bic=np.bincount([r['K_bic'] for r in rr], minlength=4).tolist(), K_cv=np.bincount([r['K_cv'] for r in rr], minlength=4).tolist(),
                   sel_below_floor_S4=float(np.mean([r['sel_below_floor'] for r in rr if r['source'] == 'S4'])),
                   k1_below_floor_S4=float(np.mean([r['k1_below_floor'] for r in rr if r['source'] == 'S4'])),
                   comparisons=cmp, per_source=per_source)
        table.append(row); T[(band, M)] = row
    res = {}
    m1 = [T[c]['comparisons'][f"prop_bic-vs-{T[c]['best_generic']}"]['class'] for c in cells]
    res['M1'] = dict(rule='prop_bic beats the best source-free/sparse comparator in >= 12 of the 16 band x count cells and loses in none',
                     wins=m1.count('win'), ties=m1.count('tie'), losses=m1.count('loss'), passed=bool(m1.count('win') >= 12 and m1.count('loss') == 0))
    m2 = [(T[c]['comparisons']['prop_bic-vs-mo26']['high'] < NONINF * T[c]['mean_mo26'], T[c]['comparisons']['prop_bic-vs-mo26']['class']) for c in cells]
    res['M2'] = dict(rule='prop_bic non-inferior to the multiple-region kernel (upper bound < 0.10 x its mean) in every cell',
                     non_inferior=sum(a for a, _ in m2), wins=sum(b == 'win' for _, b in m2), ties=sum(b == 'tie' for _, b in m2),
                     losses=sum(b == 'loss' for _, b in m2), passed=bool(all(a for a, _ in m2)))
    m3 = [T[c]['comparisons']['prop_bic-vs-sinc_ev']['class'] for c in cells]
    res['M3'] = dict(rule='prop_bic never loses to the source-free sinc kernel', losses=m3.count('loss'), wins=m3.count('win'), passed=bool('loss' not in m3))
    m4 = [T[c]['comparisons']['prop_bic-vs-prop_k1']['class'] for c in cells]
    res['M4'] = dict(rule='descriptive: model-order selection against one point', wins=m4.count('win'), ties=m4.count('tie'), losses=m4.count('loss'))
    stop = [T[c]['comparisons'][f"prop_bic-vs-{T[c]['best_comparator']}"]['class'] for c in cells]
    allpairs = []
    for r in rows:
        c = T[(r['band'], r['M'])]['best_comparator']
        allpairs.append(dict(source=r['source'], bin=(r['bin'], r['M']), nmse_a=r['nmse_prop_bic'], nmse_b=r[f'nmse_{c}']))
    pb = cluster_boot(allpairs, 'a', 'b')
    res['stop_rule'] = dict(rule='ask the user if prop_bic loses to the best comparator (sparse, source-free or multiple-region) in more than 4 of 16 cells or in the pooled comparison',
                            cell_losses=stop.count('loss'), cell_wins=stop.count('win'), pooled=pb, pooled_class=cls(pb),
                            triggered=bool(stop.count('loss') > 4 or cls(pb) == 'loss'))
    (out / 'summary.json').write_text(json.dumps(dict(rules=res, cells=[dict(r, cell=f"{r['band']:.0f}Hz-M{r['M']}") for r in table]), indent=1, default=float))
    for h in ('M1', 'M2', 'M3'):
        print(h, 'PASS' if res[h]['passed'] else 'FAIL', '-', res[h]['rule'])
    print('stop rule', 'TRIGGERED' if res['stop_rule']['triggered'] else 'not triggered')


if __name__ == '__main__':
    main()
