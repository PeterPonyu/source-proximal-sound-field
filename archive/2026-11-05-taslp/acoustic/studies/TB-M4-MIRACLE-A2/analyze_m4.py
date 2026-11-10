"""TB-M4-MIRACLE-A2 registered analysis (PROTOCOL.md), with the statistics of analyze_v2.py and analyze_m3.py imported.

usage:
  python3 -B analyze_m4.py RUN_DIR [--a1-check DRY_DIR]
      -> RUN_DIR/summary.json. A2 rows from RUN_DIR; A1 and R2 rows from TB-M3 (SHA-256 checked against its receipt).
      DRY_DIR: output of `run_m4.py dry` on the A2 box; its A1 reproduction is reported beside rule A2-2.
  python3 -B analyze_m4.py --dry OUT_DIR
      -> OUT_DIR/summary-dry.json: code-path check only. TB-M3's R2 rows stand in for A2 and copies of the guarded SBL
      errors stand in for the comparator. No A2 data.
"""
from __future__ import annotations
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
sys.path.insert(0, str(S)); sys.path.insert(0, str(S / 'studies' / 'TB-M3-MIRACLE'))
import analyze_v2 as AV  # noqa: E402
import analyze_m3 as AM3  # noqa: E402

P = AV.PRIMARY
MF = ('mfsbl_mackay', 'mfsbl_em')      # order matters: min() keeps the first on a tie, so ties go to mfsbl_mackay
MF_EXCLUDE_MAX = 0.05
M3_DIR = S / 'studies/TB-M3-MIRACLE/primary'
BANDS = ('250', '500', '1000')


def cells_of(rs):
    for r in rs:
        r['cell_key'] = f"{int(r['band'])}Hz-M{r['M']}"
    return sorted({r['cell_key'] for r in rs}, key=lambda c: (int(c.split('Hz')[0]), int(c.split('M')[1])))


def by_cell(rs):
    by = defaultdict(list)
    for r in rs:
        by[r['cell_key']].append(r)
    return by


def classes(by, cells, other):
    """Per condition: primary minus other(c), Bonferroni over the conditions, the registered classification."""
    out = {}
    for c in cells:
        d = AV.cell_diff(by[c], 'mesh', P, other(c))
        m, lo, hi = AV.tint(list(d.values()), AV.ALPHA / len(cells))
        out[c] = dict(against=other(c), n_rows=len(by[c]), mean=m, low=lo, high=hi, cls=AV.classify(m, lo, hi), non_inferior=bool(hi < AV.NI_MARGIN))
    cl = [v['cls'] for v in out.values()]
    return dict(wins=cl.count('win'), ties=cl.count('tie'), losses=cl.count('loss'), by_cell=out)


def unit_margin(by, cells, best):
    """Per source: mean over conditions of primary - best(c) (dB). Sources missing a condition are dropped and listed."""
    per = defaultdict(list)
    for c in cells:
        for u, v in AV.cell_diff(by[c], 'mesh', P, best(c)).items():
            per[u].append(v)
    keep = {u: float(np.mean(v)) for u, v in per.items() if len(v) == len(cells)}
    return keep, sorted(u for u in per if u not in keep)


def per_band(T, cells, key):
    return {b: AV.count(T, [c for c in cells if c.startswith(b + 'Hz')], key) for b in BANDS}


def comparator(rs, cells):
    """Multi-frequency SBL summaries on the rows where both variants' errors are finite (PROTOCOL.md, rule A2-4)."""
    ok = [r for r in rs if all(m in r['e'] and np.isfinite(r['e'][m]) for m in MF)]
    excluded = len(rs) - len(ok)
    res = dict(rows=len(rs), rows_used=len(ok), rows_excluded=excluded, excluded_fraction=excluded / max(len(rs), 1),
               comparator_errors=sum('mfsbl_error' in r for r in rs),
               non_finite={m: int(sum(m in r['e'] and not np.isfinite(r['e'][m]) for r in rs)) for m in MF},
               missing={m: int(sum(m not in r['e'] for r in rs)) for m in MF})
    res['evaluable'] = bool(res['excluded_fraction'] <= MF_EXCLUDE_MAX)
    with_info = [r for r in rs if all(m in r['info'] for m in MF)]
    res['fits'] = {m: dict(stopped_by_iteration_cap=float(np.mean([not r['info'][m]['stopped_by_tolerance'] for r in with_info])) if with_info else None,
                           iterations=float(np.mean([r['info'][m]['iterations'] for r in with_info])) if with_info else None,
                           active=float(np.mean([r['info'][m]['active'] for r in with_info])) if with_info else None) for m in MF}
    if not ok or {r['cell_key'] for r in ok} != set(cells):
        res['evaluable'] = False
        return res
    by = by_cell(ok)
    mean = {c: {m: float(np.mean([r['e'][m] for r in by[c]])) for m in AV.GENERIC + MF} for c in cells}
    better = {c: min(MF, key=lambda m: mean[c][m]) for c in cells}
    ext = {c: min(AV.GENERIC + MF, key=lambda m: mean[c][m]) for c in cells}
    bv = classes(by, cells, lambda c: better[c])
    res.update(better_variant=dict(**bv, member_by_cell=better, comparator_wins_in=[c for c, v in bv['by_cell'].items() if v['cls'] == 'loss']),
               pooled_vs_better_variant=AM3.pooled_bg(by, cells, better),
               fixed_point_only_descriptive=dict(**classes(by, cells, lambda c: 'mfsbl_mackay'), pooled=AV.pooled(ok, 'mesh', cells, P, 'mfsbl_mackay')),
               em_only_descriptive=dict(**classes(by, cells, lambda c: 'mfsbl_em'), pooled=AV.pooled(ok, 'mesh', cells, P, 'mfsbl_em')),
               extended_best_generic=dict(**classes(by, cells, lambda c: ext[c]), member_by_cell=ext, pooled=AM3.pooled_bg(by, cells, ext)),
               guarded_sbl_minus_mfsbl={f'{a}-{b}': AV.pooled(ok, 'mesh', cells, a, b) for a, b in
                                        (('sbl_em_common', 'mfsbl_em'), ('sbl_mackay_common', 'mfsbl_mackay'))})
    return res


def scenario(rs, with_comparator):
    cells = cells_of(rs)
    T = AV.analyse(rs, 'mesh', cells, len(cells))
    by = by_cell(rs)
    best = {c: T[c]['best_generic'] for c in cells}
    every = 0                                                       # as analyze_m3: wins against every generic member
    for c in cells:
        every += all(AV.classify(*AV.tint(list(AV.cell_diff(by[c], 'mesh', P, m).values()), AV.ALPHA / len(cells))) == 'win'
                     for m in AV.GENERIC)
    cnt = lambda k: AV.count(T, cells, k)                            # noqa: E731
    pl = lambda m: AV.pooled(rs, 'mesh', cells, P, m)                # noqa: E731
    s = dict(problems=len(rs), cells=len(cells), units=len({r['source'] for r in rs}),
             best_generic=classes(by, cells, lambda c: best[c]), best_generic_by_cell=best, every_member_wins=every,
             sinc=dict(**cnt(f'{P}-vs-sinc_ev'), by_band=per_band(T, cells, f'{P}-vs-sinc_ev')),
             gsmf=dict(**cnt(f'{P}-vs-gsmf'), pooled=pl('gsmf'), by_band=per_band(T, cells, f'{P}-vs-gsmf')),
             mo26_cap5=dict(**cnt(f'{P}-vs-mo26_cap5'), pooled=pl('mo26_cap5')),
             v1=dict(**cnt(f'{P}-vs-v1'), pooled=pl('v1')),
             two_stage_point=dict(**cnt(f'{P}-vs-two_stage_point'), pooled=pl('two_stage_point')),
             two_stage_region=dict(**cnt(f'{P}-vs-two_stage_region'), pooled=pl('two_stage_region')),
             ref_point=cnt(f'{P}-vs-ref_point'), ref_region=cnt(f'{P}-vs-ref_region'),
             pooled_vs_best_generic_task=pl('best_generic_task'),
             pooled_vs_best_generic_cell=AM3.pooled_bg(by, cells, best),
             by_band={b: AM3.pooled_bg(by, [c for c in cells if c.startswith(b + 'Hz')], best) for b in BANDS},
             K_cv_hist=np.bincount([r['info']['j_K']['cv'] for r in rs], minlength=6).tolist(),
             median_error_db=dict(proposed=float(np.median([r['e'][P] for r in rs])),
                                  best_generic_of_condition=float(np.median([r['e'][best[r['cell_key']]] for r in rs])),
                                  arrival_time_point=float(np.median([r['e']['two_stage_point'] for r in rs]))),
             cells_table=T)
    if with_comparator:
        s['mfsbl'] = comparator(rs, cells)
    margins = {'best_generic_cell': unit_margin(by, cells, lambda c: best[c]),
               'best_generic_task': unit_margin(by, cells, lambda c: 'best_generic_task')}
    margins.update({m: unit_margin(by, cells, lambda c, m=m: m) for m in AV.GENERIC})
    return s, margins


def contrast(a, b, label):
    """Per shared source: margin(a) - margin(b), 95 % Student-t interval over the sources."""
    u = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in u]
    m, lo, hi = AV.tint(d, AV.ALPHA)
    return dict(contrast=label, mean=m, low=lo, high=hi, units=len(u),
                interval='above zero' if lo > 0 else ('below zero' if hi < 0 else 'straddles zero'))


def analyse(a2_rows, m3_rows, a1_check=None):
    res = dict(n_rows_a2=len(a2_rows), note='sbl_* are the target-guarded comparators; adk_standin has the corrected gradient; '
                                            'mfsbl_* is the multi-frequency shared-support comparator')
    margins = {}
    for name, rs in (('A2', a2_rows), ('A1', [r for r in m3_rows if r['scenario'] == 'A1']), ('R2', [r for r in m3_rows if r['scenario'] == 'R2'])):
        res[name], margins[name] = scenario(rs, name == 'A2')
    res['dropped_sources'] = {sc: m['best_generic_cell'][1] for sc, m in margins.items()}
    a2 = res['A2']
    dist = contrast(margins['A2']['best_generic_cell'][0], margins['A1']['best_generic_cell'][0], 'A2 - A1 (distance)')
    refl = contrast(margins['R2']['best_generic_cell'][0], margins['A2']['best_generic_cell'][0], 'R2 - A2 (reflector)')
    extra = {k: contrast(margins['A2'][k][0], margins['A1'][k][0], f'A2 - A1 against {k}') for k in margins['A2'] if k != 'best_generic_cell'}
    mf = a2['mfsbl']
    rules = dict(
        A2_1=dict(rule='A2: the primary beats the best generic estimator in >= 9 of 12 conditions and loses in none',
                  wins=a2['best_generic']['wins'], ties=a2['best_generic']['ties'], losses=a2['best_generic']['losses'],
                  passed=bool(a2['best_generic']['wins'] >= 9 and a2['best_generic']['losses'] == 0)),
        A2_2=dict(rule='distance: per-source pooled margin on A2 minus on A1, 95 % interval; the proximity explanation predicts '
                       'an interval above zero', **dist, descriptive_against_each_member=extra, a1_reproduction_on_a2_box=a1_check),
        A2_3=dict(rule='reflector: R2 minus A2, descriptive', **refl),
    )
    r4 = 'A2: the primary beats the better multi-frequency SBL variant in >= 9 of 12 conditions and loses in none (rows with both comparator errors finite)'
    if mf['evaluable']:
        b = mf['better_variant']
        rules['A2_4'] = dict(rule=r4, wins=b['wins'], ties=b['ties'], losses=b['losses'], comparator_wins_in=b['comparator_wins_in'],
                             rows_excluded=mf['rows_excluded'], passed=bool(b['wins'] >= 9 and b['losses'] == 0))
    else:
        rules['A2_4'] = dict(rule=r4, passed=None, status='not evaluable', rows_excluded=mf['rows_excluded'],
                             excluded_fraction=mf['excluded_fraction'])
    res['rules'] = rules
    res['stop_rule'] = dict(rule='state in the abstract if the primary is pooled worse than the best generic estimator on A2',
                            triggered=bool(a2['pooled_vs_best_generic_cell']['low'] > 0))
    return res


def load_m3():
    want = json.loads((M3_DIR / 'run-receipt.json').read_text())['outputs']['rows.jsonl']
    got = hashlib.sha256((M3_DIR / 'rows.jsonl').read_bytes()).hexdigest()
    assert got == want, f'TB-M3 rows.jsonl SHA-256 {got} does not match its run receipt {want}'
    return AV.load(M3_DIR)


def stand_in(m3_rows, nan_rows=0):
    a2 = [dict(r, scenario='A2') for r in json.loads(json.dumps([r for r in m3_rows if r['scenario'] == 'R2']))]
    for i, r in enumerate(a2):
        for m, src in zip(MF, ('sbl_mackay_common', 'sbl_em_common')):
            r['e'][m] = r['e'][src]; r['info'][m] = dict(stopped_by_tolerance=True, iterations=1, active=1)
        if i < nan_rows:
            r['e']['mfsbl_em'] = float('nan')
    return a2


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        raise SystemExit(__doc__)
    m3_rows, m3_err = load_m3()
    if sys.argv[1] == '--dry':
        out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
        ref = json.loads((M3_DIR / 'summary.json').read_text())['R2']
        res = analyse(stand_in(m3_rows), m3_rows)
        a2 = res['A2']; bv = a2['mfsbl']['better_variant']['by_cell']; bg = a2['best_generic']['by_cell']
        few = analyse(stand_in(m3_rows, 3), m3_rows)['A2']['mfsbl']
        many = analyse(stand_in(m3_rows, 40), m3_rows)['A2']['mfsbl']
        res['dry'] = dict(note='code-path check: TB-M3 R2 rows relabelled as A2, guarded SBL copies as the comparator; not a result',
                          a2_1_counts_equal_tb_m3_g4=bool(all(a2['best_generic'][k] == ref['best_generic'][k] for k in ('wins', 'ties', 'losses'))),
                          pooled_equal=bool(abs(a2['pooled_vs_best_generic_cell']['mean'] - ref['pooled_vs_best_generic_cell']['mean']) < 1e-12),
                          every_member_wins_equal=bool(a2['every_member_wins'] == ref['every_member_wins']),
                          reflector_contrast_is_zero=bool(abs(res['rules']['A2_3']['mean']) < 1e-12),
                          stand_in_minus_guarded_sbl_is_zero=bool(all(abs(v['mean']) < 1e-12 for v in a2['mfsbl']['guarded_sbl_minus_mfsbl'].values())),
                          a2_4_equals_a2_1_where_both_pick_the_fixed_point=bool(all(
                              bv[c]['cls'] == bg[c]['cls'] for c in bg if a2['best_generic_by_cell'][c] == 'sbl_mackay_common'
                              and bv[c]['against'] == 'mfsbl_mackay')),
                          three_nan_rows=dict(excluded=few['rows_excluded'], evaluable=few['evaluable']),
                          forty_nan_rows=dict(excluded=many['rows_excluded'], evaluable=many['evaluable']))
        res['dry']['passed'] = bool(all(v for k, v in res['dry'].items() if isinstance(v, bool))
                                    and few['rows_excluded'] == 3 and few['evaluable'] and many['rows_excluded'] == 40 and not many['evaluable'])
        (out / 'summary-dry.json').write_text(json.dumps(res, indent=1, default=float))
        print(json.dumps(res['dry']))
        return
    run = Path(sys.argv[1])
    a1_check = None
    if '--a1-check' in sys.argv:
        d = json.loads((Path(sys.argv[sys.argv.index('--a1-check') + 1]) / 'dry-compare.json').read_text())
        a1_check = dict(problems=d['problems'], reproduced_1e6=d['reproduced_1e6'], worst_rel=d['worst_rel'])
    rows, errs = AV.load(run)
    res = analyse([r for r in rows if r['scenario'] == 'A2'], m3_rows, a1_check)
    res.update(n_errors_a2=len(errs), n_errors_m3=len(m3_err))
    (run / 'summary.json').write_text(json.dumps(res, indent=1, default=float))
    for k, v in res['rules'].items():
        print(k, v.get('passed', v.get('interval')), '-', v['rule'])


if __name__ == '__main__':
    main()
