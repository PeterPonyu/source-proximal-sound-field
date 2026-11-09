"""Kill rules of V2-DESIGN.md on the development rows; development numbers only.

python3 -B analyze_dev.py DEV_DIR   ->  DEV_DIR/summary.json and a printed report
Per task, a method's error is the mean NMSE over the bins of the band, in dB. Cells: simulated cells; measured
(band, microphone count) pooled over loudspeakers and draws. Paired differences are per task, in dB; a cell value is
the median paired difference.
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

GENERIC = ('sinc_ev', 'rbf', 'adk_standin', 'sbl_em_common', 'sbl_mackay_common')
TWO_SOURCE = ('A6-second1.0', 'A5-delta0.4')


def db(v):
    return 10 * np.log10(np.mean(v))


def load(d):
    rows, errs = [], []
    for line in open(Path(d) / 'rows.jsonl'):
        r = json.loads(line)
        (errs if 'error' in r else rows).append(r)
    return rows, errs


def cell_of(r):
    return r['cell'] if r['kind'] == 'sim' else f"meas-{int(r['band'])}-M{r['M']}"


def main():
    d = Path(sys.argv[1])
    rows, errs = load(d)
    E = defaultdict(list)                     # cell -> list of {method: dB}
    for r in rows:
        e = {m: db(v) for m, v in r['nmse'].items()}
        e['best_generic'] = min(e[m] for m in GENERIC if m in e)
        E[cell_of(r)].append(dict(e=e, r=r))
    cells = sorted(E)
    sim_cells = [c for c in cells if not c.startswith('meas')]
    meas_cells = [c for c in cells if c.startswith('meas')]

    def med(c, a, b):
        x = [t['e'][a] - t['e'][b] for t in E[c] if a in t['e'] and b in t['e']]
        return float(np.median(x)) if x else None

    def absmed(c, a):
        x = [t['e'][a] for t in E[c] if a in t['e']]
        return float(np.median(x)) if x else None
    res = dict(n_rows=len(rows), n_errors=len(errs), cells={c: len(E[c]) for c in cells})
    # (b) selectors, on the joint sinc fits
    sel = {}
    for s in ('sw', 'la', 'cv'):
        m = f'j_{s}'
        hind = []
        for c in cells:
            x = [t['e'][m] - min(t['e'][f'j_K{K}'] for K in range(6) if f'j_K{K}' in t['e']) for t in E[c]]
            hind.append(float(np.median(x)))
        sel[s] = dict(mean_error_db=float(np.mean([absmed(c, m) for c in cells])),
                      two_source_vs_k1={c: med(c, m, 'j_K1') for c in TWO_SOURCE if c in E},
                      meas_cells_losing_to_k1=[c for c in meas_cells if med(c, m, 'j_K1') > 0],
                      median_loss_vs_hindsight_best_db=float(np.median(hind)))
        v = sel[s]
        v['passes'] = bool(all(x is not None and x < 0 for x in v['two_source_vs_k1'].values())
                           and len(v['meas_cells_losing_to_k1']) <= 2 and v['median_loss_vs_hindsight_best_db'] <= 0.3)
    ok = [s for s in sel if sel[s]['passes']]
    chosen = min(ok, key=lambda s: sel[s]['mean_error_db']) if ok else 'K1'
    res['b_order_selection'] = dict(selectors=sel, chosen=chosen,
                                    cap3_vs_cap5_schwarz={c: med(c, 'j_sw3', 'j_sw') for c in cells})
    tag = 'K1' if chosen == 'K1' else chosen
    pts = 'j_K1' if chosen == 'K1' else f'j_{chosen}'
    # (a) extension at the chosen selector
    if chosen != 'K1':
        a = {c: med(c, f'j_{chosen}_ext', pts) for c in cells}
        res['a_extension'] = dict(per_cell=a, worst_db=max(a.values()), best_db=min(a.values()),
                                  keep=bool(max(a.values()) <= 0.1 and min(a.values()) <= -0.2))
    # (c) shared coordinates
    loc_j = [t['r']['j_k1_loc_m'] for c in meas_cells for t in E[c]]
    loc_i = [float(np.median(t['r']['ind_k1_loc_m'])) for c in meas_cells for t in E[c]]
    groups = {'125 low M': [c for c in meas_cells if '-125-' in c and int(c.split('M')[-1]) <= 48],
              '125 high M': [c for c in meas_cells if '-125-' in c and int(c.split('M')[-1]) >= 96],
              '250 low M': [c for c in meas_cells if '-250-' in c and int(c.split('M')[-1]) <= 48],
              '250 high M': [c for c in meas_cells if '-250-' in c and int(c.split('M')[-1]) >= 96]}
    gmed = {g: float(np.median([t['e']['j_sw'] - t['e']['ind_sw'] for c in cs for t in E[c]])) for g, cs in groups.items() if cs}
    simw = {c: med(c, 'j_sw', 'ind_sw') for c in sim_cells}
    red = 1 - np.median(loc_j) / np.median(loc_i) if loc_i else None
    res['c_shared_coordinates'] = dict(meas_loc_median_joint_m=float(np.median(loc_j)) if loc_j else None,
                                       meas_loc_median_independent_m=float(np.median(loc_i)) if loc_i else None,
                                       reduction=None if red is None else float(red), low_band_groups=gmed, sim_cells=simw,
                                       keep=bool(red is not None and red >= 0.3 and sum(v < 0 for v in gmed.values()) >= 2
                                                 and (not simw or max(simw.values()) <= 0.1)))
    # (d) remainder family
    dsim = {c: med(c, 'fj_sw', 'j_sw') for c in sim_cells}
    d125 = {c: med(c, 'fj_sw', 'j_sw') for c in meas_cells if '-125-' in c}
    fams = defaultdict(int)
    for c in cells:
        for t in E[c]:
            fams[(c.startswith('meas'), t['r']['family'])] += 1
    res['d_family'] = dict(sim_cells=dsim, meas_125_cells=d125, chosen_counts={f'{"meas" if k[0] else "sim"}:{k[1]}': v for k, v in fams.items()},
                           keep=bool((not dsim or max(dsim.values()) <= 0.1) and d125 and np.median(list(d125.values())) < 0))
    # version 2 as assembled by the kill rules
    fam_on = res['d_family']['keep']
    base = ('fj' if fam_on else 'j')
    v2m = f'{base}_K1' if chosen == 'K1' else f'{base}_{chosen}' + ('_ext' if res.get('a_extension', {}).get('keep') else '')
    if not res['c_shared_coordinates']['keep']:
        v2m_note = 'shared coordinates killed: version 2 should use per-bin fits; the joint form is still reported here'
    else:
        v2m_note = ''
    res['version2_method'] = dict(column=v2m, note=v2m_note)
    comp = ('v1', 'mo26_cap5', 'hybrid_v1', 'gsmf', 'best_generic', 'sinc_ev', 'rbf', 'ref_point', 'ref_region', 'two_stage_point', 'two_stage_region')
    res['per_cell'] = {c: dict(n=len(E[c]), v2=absmed(c, v2m), **{f'v2_minus_{k}': med(c, v2m, k) for k in comp}) for c in cells}
    regimes = {'sparse (M <= 48)': [c for c in cells if (('M24' in c or 'M48' in c) or c in ('A2-f125-M24', 'A2-f1000-M24', 'A2-f250-M48'))],
               'rich (M >= 192 or 40 dB)': [c for c in cells if 'M192' in c or c == 'A9-snr40'],
               'low frequency (125-250 Hz)': [c for c in cells if '125' in c or '250' in c],
               'inside or beside (simulated)': [c for c in sim_cells if c.startswith('A1')],
               'all simulated': sim_cells, 'all measured': meas_cells}
    res['regimes'] = {g: {k: (float(np.median([res['per_cell'][c][f'v2_minus_{k}'] for c in cs if res['per_cell'][c][f'v2_minus_{k}'] is not None]))
                              if cs else None) for k in comp} for g, cs in regimes.items()}
    (d / 'summary.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ('per_cell',)}, indent=1, default=float)[:6000])


if __name__ == '__main__':
    main()
