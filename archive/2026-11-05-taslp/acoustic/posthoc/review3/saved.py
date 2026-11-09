"""Post hoc analyses from saved outputs only (PLAN.md, part A). No estimate is refitted.

A1 driver geometry of every TB-S2 condition; A2 margins by driver geometry; A3 the proposed estimator against every
generic estimator separately (intersection-union); A4 run time with only the cross-validated order's extension; A5 the
undefined measured comparison counted apart. The registered statistics of analyze_v2.py are imported unchanged.

    python3 -B posthoc/review3/saved.py posthoc/review3/saved.json
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
sys.path.insert(0, str(S / 'common')); sys.path.insert(0, str(S)); sys.path.insert(0, str(S / 'posthoc/geometry'))
import analyze_v2 as AV  # noqa: E402
import geometry as G  # noqa: E402
import sim  # noqa: E402

PRIMARY = AV.PRIMARY
GEN = AV.GENERIC
SEEDS = json.loads((S / 'SEEDS.json').read_text())['declared']['TB-S2_simulation_fields']


def driver_geometry():
    cells = sim.cells()
    out = {}
    for name, cell in cells.items():
        per = []
        for seed in range(SEEDS[0], SEEDS[1] + 1):
            g = G.geometry_only(cell, seed)
            lo, hi = g['box']
            o = 0.5 * (lo + hi); a = 0.5 * float(np.linalg.norm(hi - lo))
            labels = ['primary'] + (['coherent'] if cell['delta'] else []) + (['second'] if cell['second'] else [])
            assert len(labels) == len(g['sources'])
            drv = []
            for lab, s_ in zip(labels, g['sources']):
                inside = bool(np.all(s_ >= lo) and np.all(s_ <= hi))
                if inside:
                    sd = -float(np.min(np.r_[s_ - lo, hi - s_]))
                else:
                    sd = float(np.linalg.norm(np.maximum(np.maximum(lo - s_, s_ - hi), 0)))
                drv.append(dict(driver=lab, inside=inside, signed_distance_m=sd, d_over_a=float(np.linalg.norm(s_ - o)) / a))
            per.append(drv)
        labs = [d['driver'] for d in per[0]]
        summ = {}
        for i, lab in enumerate(labs):
            v = [p[i] for p in per]
            summ[lab] = dict(inside_fields=int(sum(d['inside'] for d in v)),
                             signed_distance_m=G.span([d['signed_distance_m'] for d in v]),
                             d_over_a=G.span([d['d_over_a'] for d in v]))
        out[name] = dict(drivers=summ, interior=bool(any(s['inside_fields'] for s in summ.values())),
                         interior_all_fields=bool(all(any(d['inside'] for d in p) for p in per)))
    return out


def cells_of(rows):
    by = defaultdict(list)
    for r in rows:
        by[r['cell_key']].append(r)
    return by


def finite_diff(rs, kind, a, b):
    d = AV.cell_diff(rs, kind, a, b)
    return d, bool(all(np.isfinite(v) for v in d.values()))


def pooled_bg(by, kind, cells, best):
    per = defaultdict(list)
    for c in cells:
        for u, v in AV.cell_diff(by[c], kind, PRIMARY, best[c]).items():
            per[u].append(v)
    m, lo, hi = AV.tint([np.mean(v) for v in per.values()], AV.ALPHA)
    return dict(mean=m, low=lo, high=hi, units=len(per), cells=len(cells))


def pooled(by, kind, cells, b):
    return pooled_bg(by, kind, cells, {c: b for c in cells})


def classes(table, cells, key):
    cl = [table[c]['comparisons'][key]['cls'] for c in cells]
    return dict(wins=cl.count('win'), ties=cl.count('tie'), losses=cl.count('loss'), cells=len(cl))


def every_member(by, kind, cells, family_n, members):
    """Per condition, the primary against each member separately; a win against every member is an intersection-union
    test at the level of each component (Bonferroni over the conditions of the rule only)."""
    alpha = AV.ALPHA / family_n
    per_cell = {}
    for c in cells:
        res = {}
        for m in members:
            d, ok = finite_diff(by[c], kind, PRIMARY, m)
            if not ok:
                res[m] = dict(cls='undefined'); continue
            mu, lo, hi = AV.tint(list(d.values()), alpha)
            res[m] = dict(mean=mu, low=lo, high=hi, cls=AV.classify(mu, lo, hi))
        per_cell[c] = dict(per_member=res, win_all=bool(all(v['cls'] == 'win' for v in res.values())),
                           loss_any=bool(any(v['cls'] == 'loss' for v in res.values())),
                           undefined_any=bool(any(v['cls'] == 'undefined' for v in res.values())))
    return dict(win_against_every_member=sum(v['win_all'] for v in per_cell.values()),
                loss_to_some_member=sum(v['loss_any'] for v in per_cell.values()),
                undefined_some_member=sum(v['undefined_any'] for v in per_cell.values()), cells=len(cells),
                per_member_classes={m: {k: sum(per_cell[c]['per_member'][m]['cls'] == k for c in cells)
                                        for k in ('win', 'tie', 'loss', 'undefined')} for m in members},
                per_cell=per_cell)


def main():
    out = dict(status='post hoc; saved outputs only; no estimate refitted')
    # A1
    geo = driver_geometry()
    interior = [c for c, v in geo.items() if v['interior']]
    out['A1_driver_geometry'] = dict(interior_conditions=interior, per_condition=geo)
    # A2 and A3, simulation
    rows, errs = AV.load(S / 'studies/TB-S2/primary')
    for r in rows:
        r['cell_key'] = r['cell']
    cells = list(dict.fromkeys(r['cell'] for r in rows))
    prox = [c for c in cells if c != AV.FAR]
    T = AV.analyse(rows, 'sim', cells, len(prox))
    best = {c: T[c]['best_generic'] for c in cells}
    by = cells_of(rows)
    ext = [c for c in prox if c not in interior]
    inn = [c for c in prox if c in interior]
    inn5 = [c for c in inn if c != 'A1-in']
    a2 = {}
    for lab, cs in (('all_63', prox), ('no_interior_driver', ext), ('interior_driver', inn), ('interior_driver_without_A1_in', inn5),
                    ('without_A1_in_only', [c for c in prox if c != 'A1-in'])):
        a2[lab] = dict(cells=len(cs), best_generic=pooled_bg(by, 'sim', cs, best),
                       best_generic_classes={k: sum(T[c]['comparisons'][f'{PRIMARY}-vs-{best[c]}']['cls'] == k for c in cs) for k in ('win', 'tie', 'loss')},
                       gsmf=dict(pooled=pooled(by, 'sim', cs, 'gsmf'), **classes(T, cs, f'{PRIMARY}-vs-gsmf'),
                                 non_inferior=int(sum(T[c]['comparisons'][f'{PRIMARY}-vs-gsmf']['non_inferior'] for c in cs))),
                       mo26_cap5=dict(pooled=pooled(by, 'sim', cs, 'mo26_cap5'), **classes(T, cs, f'{PRIMARY}-vs-mo26_cap5'),
                                      non_inferior=int(sum(T[c]['comparisons'][f'{PRIMARY}-vs-mo26_cap5']['non_inferior'] for c in cs))))
    a2['per_interior_condition'] = {c: dict(best_generic=best[c], **{k: T[c]['comparisons'][f'{PRIMARY}-vs-{best[c]}'][k]
                                                                       for k in ('mean', 'low', 'high', 'cls')}) for c in inn}
    a2['note'] = 'per-condition classes keep the registered Bonferroni correction over 63 conditions'
    out['A2_margins_by_driver_geometry'] = a2
    out['A3_every_generic_member'] = dict(simulation=every_member(by, 'sim', prox, len(prox), GEN))
    out['A3_every_generic_member']['simulation_no_interior_driver'] = {
        'win_against_every_member': sum(out['A3_every_generic_member']['simulation']['per_cell'][c]['win_all'] for c in ext), 'cells': len(ext)}
    # A3, measured: registered and target-guarded sparse Bayesian comparators
    mrows, _ = AV.load(S / 'studies/TB-M2/primary')
    mesh = [r for r in mrows if r['kind'] == 'mesh']
    for r in mesh:
        r['cell_key'] = f"{int(r['band'])}Hz-M{r['M']}"
    fix = {(r['source'], r['band'], r['M'], r['draw']): r for r in map(json.loads, open(S / 'studies/TB-M2-SBLFIX/primary/rows.jsonl'))
           if 'error' not in r}
    for r in mesh:
        for m, v in fix[(r['source'], r['band'], r['M'], r['draw'])]['nmse'].items():
            v = np.asarray(v, float)
            r['e'][m] = AV.db(v) if np.all(np.isfinite(v)) and np.all(v < 1e6) else np.nan
    for r in mesh:                       # the registered threshold: an error that is non-finite or above 1e6 is undefined
        for m in ('sbl_em_common', 'sbl_mackay_common'):
            v = np.asarray(r['nmse'][m], float)
            if not (np.all(np.isfinite(v)) and np.all(v < 1e6)):
                r['e'][m] = np.nan
    mcells = sorted({r['cell_key'] for r in mesh}, key=lambda c: (int(c.split('Hz')[0]), int(c.split('M')[1])))
    mby = cells_of(mesh)
    out['A3_every_generic_member']['measured_registered'] = every_member(mby, 'mesh', mcells, len(mcells), GEN)
    out['A3_every_generic_member']['measured_guarded'] = every_member(
        mby, 'mesh', mcells, len(mcells), ('sinc_ev', 'rbf', 'adk_standin', 'sbl_em_guard', 'sbl_mackay_guard'))
    out['A3_every_generic_member']['measured_finite_members'] = every_member(mby, 'mesh', mcells, len(mcells), ('sinc_ev', 'rbf', 'adk_standin'))
    # A4: run time with only the cross-validated order's extension (registered timers; one core of the rented server)
    def t_prop(r, ext=True):
        t = r['timing']; k = r['info']['j_K']['cv']
        return t['j_fit'] + t['j_cv'] + ((t.get(f'j_ext_K{k}', 0.0) if k > 0 else 0.0) if ext else 0.0)
    def t_reg(r):
        return sum(v for k, v in r['timing'].items() if k in ('j_fit', 'j_cv') or k.startswith('j_ext_'))
    rt = {}
    for M in (24, 48, 96, 192):
        rs = [r for r in mesh if r['M'] == M]
        rt[str(M)] = dict(proposed_cv_extension_only=float(np.median([t_prop(r) for r in rs])),
                          proposed_as_registered_table=float(np.median([t_reg(r) for r in rs])),
                          points_only=float(np.median([t_prop(r, False) for r in rs])),
                          cross_validation_share=float(np.median([r['timing']['j_cv'] / t_prop(r) for r in rs])),
                          schwarz_extension_counted=int(sum(t_reg(r) != t_prop(r) for r in rs)), problems=len(rs))
    ex = [json.loads(l) for l in open(S / 'studies/TB-M2-EXACT/primary/rows.jsonl')]
    hi = [json.loads(l) for l in open(S / 'studies/TB-M2-EXACT-HI/primary/rows.jsonl')]
    exact = {}
    for nm, rr in (('M24_48', ex), ('M96_192', hi)):
        rr = [r for r in rr if 'error' not in r]
        for M in sorted({r['M'] for r in rr}):
            q = [r for r in rr if r['M'] == M]
            exact[str(M)] = {k: float(np.median([r['seconds'][k] for r in q])) for k in q[0]['seconds']}
    out['A4_runtime'] = dict(unit='median process seconds per band of six bins, one core',
                             registered_machine='rented server, AMD EPYC 7B13 (registered TB-M2)',
                             exact_machine='laptop, Intel Core Ultra 9 275HX (TB-M2-EXACT, TB-M2-EXACT-HI), with in-run cubature twins',
                             proposed=rt, exact_followups=exact)
    # A5: the measured per-bin five-ball comparison, undefined apart
    MT = AV.analyse(mesh, 'mesh', mcells, len(mcells))
    cnt = dict(win=0, tie=0, loss=0, undefined=0)
    for c in mcells:
        d = MT[c]['comparisons'][f'{PRIMARY}-vs-mo26_cap5']
        cnt['undefined' if not all(np.isfinite([d['mean'], d['low'], d['high']])) else d['cls']] += 1
    out['A5_measured_per_bin_variant'] = cnt
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k not in ('A1_driver_geometry', 'A3_every_generic_member')}, indent=1, default=float)[:6000])
    print('interior', interior)
    for k, v in out['A3_every_generic_member'].items():
        print(k, {kk: vv for kk, vv in v.items() if kk not in ('per_cell',)})


if __name__ == '__main__':
    main()
