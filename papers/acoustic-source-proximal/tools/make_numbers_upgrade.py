"""Numbers of the 2026-11-09 revision analyses (post hoc), read from their saved outputs.

python3 -B tools/make_numbers_upgrade.py ../../archive/2026-11-05-taslp/acoustic generated

Inputs (all post hoc, under posthoc/upgrade-*):
  upgrade-wp1/wp1-k1.jsonl, wp1-maxiter.jsonl  spherical configuration, K = 1 diagnosis of the fitting defect
  upgrade-wp1/sphere-pooled.jsonl              spherical configuration, full pipeline, registered vs converged fitting
  upgrade-wp1/wp1-prev-roomgrid.jsonl          convergence of the registered greedy fit on 64 room-grid problems
  upgrade-wp7/wp7-results.json                 per-comparator tests, practical superiority, linear scale, grouping
  upgrade-wp2/checks.json, stress.jsonl, refit-exact.jsonl   exact region predictor (when present)
  upgrade-timing/timing.jsonl                  run time of the arrival-time point and the single fitted point
Only counts, medians and sums of saved per-problem values are formed here; intervals are read from the saved analyses.
Writes generated/upgrade-numbers.tex (\\vdef{key}{value}).
"""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
import numpy as np

S = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
P = S / 'posthoc'
V = {}


def put(k, v):
    if k in V:
        raise SystemExit(f'duplicate key {k}')
    V[k] = v


def num(x, d=2, sign=True):
    s = f'{x:+.{d}f}' if sign else f'{x:.{d}f}'
    if s.lstrip('+-').strip('0.') == '':
        s = ('+' if sign else '') + s.lstrip('+-')
    return '$' + s + '$'


def iv(m, lo, hi, d=2, unit='~dB'):
    return f'{num(m, d)}{unit} [{num(lo, d)}, {num(hi, d)}]'


def sci(x):
    m, e = f'{x:.0e}'.split('e')
    return f'${m}\\times10^{{{int(e)}}}$'


def jl(p):
    return [json.loads(l) for l in open(p)]


def words(w, t, l):
    f = lambda n, s, p: 'no ' + p if n == 0 else f'{n} {s if n == 1 else p}'
    return f"{f(w, 'win', 'wins')}, {f(t, 'tie', 'ties')} and {f(l, 'loss', 'losses')}"


# ------------------------------------------------------------------ WP1: spherical configuration, K = 1 diagnosis
k1 = jl(P / 'upgrade-wp1/wp1-k1.jsonl')
assert len(k1) == 20 and all(r['replicated'] for r in k1)
fail = [(r, i) for r in k1 for i in range(15) if r['base']['db'][i] > -10]
put('up-sph-trials', str(len(k1))); put('up-sph-bins', str(15 * len(k1)))
put('up-sph-k1-fail', str(len(fail)))
put('up-sph-k1-bound', str(sum(r['init_at_bound'][i] for r, i in fail)))
put('up-sph-k1-cap', str(sum(r['base']['nit'] >= 200 for r in k1)))
sp = sorted({round(r['spacing_m'], 6) for r in k1}); assert len(sp) == 1
put('up-sph-spacing', f'{sp[0]:.2f}')
put('up-sph-spacing-wl', f'{sp[0] / (343.0 / 1500.0):.1f}')
put('up-sph-node-dist', f"{np.median([r['best_node_dist_m'] for r in k1]):.2f}")
for v in ('reset', 'zoom', 'fix_a'):
    vk = v.replace('_', '')
    put(f'up-sph-{vk}-fixed', str(sum(r[v]['db'][i] <= -10 for r, i in fail)))
    put(f'up-sph-{vk}-broken', str(sum(r[v]['db'][i] > -10 for r in k1 for i in range(15) if r['base']['db'][i] <= -10)))
mx = jl(P / 'upgrade-wp1/wp1-maxiter.jsonl')
put('up-sph-maxiter-fail', str(sum(d > -10 for r in mx for d in r['db'])))
put('up-sph-maxiter-nit', f"{int(np.median([r['nit'] for r in mx])):,}")

# ------------------------------------------------------------------ WP1: spherical configuration, full pipeline (pooled as Table III)
sp_path = P / 'upgrade-wp1/sphere-pooled.jsonl'
sph = jl(sp_path) if sp_path.exists() else []
if len(sph) != 20:
    print('sphere-pooled incomplete:', len(sph)); sph = []
for tag, key in ((('registered', 'reg'), ('converged', 'conv')) if sph else ()):
    e = sum(a for r in sph for a, b in r[tag]['sq']); t = sum(b for r in sph for a, b in r[tag]['sq'])
    put(f'up-sph-pool-{key}', num(10 * math.log10(e / t), 1))
    put(f'up-sph-fail-{key}', str(sum(10 * math.log10(a / b) > -10 for r in sph for a, b in r[tag]['sq'])))
    put(f'up-sph-K1-{key}', str(sum(r[tag]['K'] == 1 for r in sph)))
    pc = sum(r[tag]['point_caps'] for r in sph); pn = sum(r[tag]['point_refinements'] for r in sph)
    put(f'up-sph-pcap-{key}', f'{100 * pc / pn:.0f}')
# gate: the registered fitting reproduces the stored reproduction rows (posthoc/review3/reproduction)
rep = {r['task'][1]: r for r in jl(P / 'review3/reproduction/rows.jsonl') if r['task'][0] == 'band'}
for r in sph:
    st = rep[r['seed']]['proposed_shared']
    assert all(abs(a / b - c / d) <= 1e-9 * (c / d) for (a, b), (c, d) in zip(r['registered']['sq'], st)), r['seed']

# ------------------------------------------------------------------ WP1: convergence of the registered greedy fit, room grid sample
pr = jl(P / 'upgrade-wp1/wp1-prev-roomgrid.jsonl')
assert len(pr) == 64 and not any('error' in r for r in pr)
put('up-rg-n', str(len(pr)))
for K in '12345':
    put(f'up-rg-cap-K{K}', str(sum(r['base'][K]['cap'] for r in pr)))
dm = [abs(np.mean([np.mean(r['reset'][K]['db']) - np.mean(r['base'][K]['db']) for r in pr])) for K in '12345']
put('up-rg-reset-maxmean', num(max(dm), 3, sign=False))
put('up-rg-reset-maxabs', num(max(abs(np.mean(r['reset'][K]['db']) - np.mean(r['base'][K]['db'])) for r in pr for K in '12345'), 1, sign=False))

# ------------------------------------------------------------------ WP7: statistics re-analysis
W = json.loads((P / 'upgrade-wp7/wp7-results.json').read_text())
assert W['step1_gate']['passed']
for study, key in (('room_grid', 'rg'), ('miracle_A1', 'A1'), ('miracle_R2', 'R2'), ('simulation', 'sim')):
    for d in ('current', 'practical'):
        c = W['a_best_generic_two_definitions'][study][d]
        put(f'up-{d[:4]}-{key}', f"{c['win']}/{c['tie']}/{c['loss']}"); put(f'up-{d[:4]}-{key}-w', words(c['win'], c['tie'], c['loss']))
    b = W['b_linear'][study]['best_generic']['pooled_pct_change_geometric']
    put(f'up-pct-{key}', iv(b['mean'], b['low'], b['high'], 0, unit='\\%') if 'low' in b else num(b['mean'], 0) + '\\%')
for study, key in (('room_grid', 'rg'), ('miracle_A1', 'A1'), ('miracle_R2', 'R2')):
    c = W['c_cluster_bootstrap'][study]
    lo, hi = c['cluster_bootstrap_fixed_selection']['ci']
    put(f'up-clu-{key}', f"[{num(lo)}, {num(hi)}]")
    put(f'up-clu-{key}-ratio', f"{c['width_ratio_cluster_over_t']:.2f}")
# per-comparator Holm tests: which generic estimators change classification from the registered-style Bonferroni one
GEN = ('sinc_ev', 'rbf', 'adk_standin', 'sbl_em_common', 'sbl_mackay_common', 'sbl_em_guard', 'sbl_mackay_guard')
wtl = lambda d: (d['win'], d['tie'], d['loss'])
changed = [(c, s) for st in ('simulation', 'room_grid', 'miracle') for c, s in W['a_per_comparator'][st]['summary'].items()
           if c.split(':')[1] in GEN and wtl(s['registered_style_bonferroni_over_conditions']) != wtl(s['holm_current'])]
assert [c for c, _ in changed] == ['miracle_R2:sbl_mackay_common'], changed
_t = W['a_per_comparator']['room_grid']['tests']
_sets = {c: frozenset(x['cond'] for x in _t if x['comparator'] == c and x[k] == 'win') for c in GEN for k in ('reg_cls', 'holm_current')
         if any(y['comparator'] == c for y in _t)}
assert len(set(_sets.values())) == 1, 'room grid: every generic estimator is beaten in the same conditions'
put('up-rg-same-wins', str(len(next(iter(_sets.values())))))
put('up-holm-R2-reg', words(*wtl(changed[0][1]['registered_style_bonferroni_over_conditions'])))
put('up-holm-R2-holm', words(*wtl(changed[0][1]['holm_current'])))
put('up-sel-opt-rg',num(W['e_selection']['room_grid']['by_draw']['optimism_db'], 2, sign=False))
put('up-sel-differs-rg', str(W['e_selection']['room_grid']['by_draw']['differs_between_draws']))

# ------------------------------------------------------------------ WP2: exact region predictor (when its outputs exist)
ck = P / 'upgrade-wp2/checks.json'
if ck.exists():
    C = json.loads(ck.read_text())
    put('up-x-e1', sci(C['E1_max_rel_err']))
    R2 = json.loads((P / 'upgrade-wp2/checks-e2-refine.json').read_text())
    put('up-x-e2', sci(max(r['rel_by_directions']['80000'] for r in R2)))
st = P / 'upgrade-wp2/stress.jsonl'
if st.exists() and sum(1 for _ in open(st)) == 3168:
    # registered room-grid ('mesh') and simulated ('sim') problems only; the 96 'held' rows are not a registered study
    rows = [r for r in jl(st) if 'error' not in r and 'skipped' not in r and r['key']['kind'] in ('mesh', 'sim')]
    assert not any('error' in r for r in jl(st))
    assert max(r['gate_cubature'] for r in rows) < 1e-9
    put('up-x-n', f'{len(rows):,}')
    put('up-x-inside', f"{sum(r['targets_inside'] > 0 for r in rows):,}")
    put('up-x-near', f"{sum(r['targets_within_2cm'] > 0 for r in rows):,}")
    d = [10 * math.log10(np.mean(r['nmse']['exact'])) - 10 * math.log10(np.mean(r['nmse']['cubature'])) for r in rows]
    put('up-x-max-shift', num(max(abs(x) for x in d), 2, sign=False))
    put('up-x-gt01', str(sum(abs(x) > 0.1 for x in d)))
    assert max(r['gate_series'] for r in rows) < 1e-9
    ds = [10 * math.log10(np.mean(r['nmse']['exact'])) - 10 * math.log10(np.mean(r['nmse']['series'])) for r in rows]
    put('up-x-series-max', num(max(abs(x) for x in ds), 2, sign=False))
    ratio_c = [r['node_stress']['cubature']['max_on_node_over_typical'] / r['node_stress']['cubature']['max_near_over_typical'] for r in rows]
    ratio_x = [r['node_stress']['exact']['max_on_node_over_typical'] / r['node_stress']['exact']['max_near_over_typical'] for r in rows]
    put('up-x-node-cub', f'{np.median(ratio_c):,.0f}'); put('up-x-node-exact', f'{np.median(ratio_x):.2f}')
    put('up-x-node-exact-max', f'{max(ratio_x):.2f}')
rf = P / 'upgrade-wp2/refit-exact.jsonl'
if rf.exists() and sum(1 for _ in open(rf)) == 128:
    rr = [r for r in jl(rf) if 'error' not in r and not r.get('skipped')]
    assert not any('error' in r for r in jl(rf))
    for kind, key in (('mesh', 'rg'), ('sim', 'sim')):
        x = [10 * math.log10(np.mean(r['nmse_exact_refit'])) - 10 * math.log10(np.mean(r['nmse_registered'])) for r in rr if r['key']['kind'] == kind]
        m = float(np.mean(x)); se = float(np.std(x, ddof=1) / math.sqrt(len(x)))
        from scipy import stats
        h = stats.t.ppf(0.975, len(x) - 1) * se
        put(f'up-rf-{key}', iv(m, m - h, m + h)); put(f'up-rf-{key}-n', str(len(x)))
        put(f'up-rf-{key}-gt01', str(sum(abs(v) > 0.1 for v in x))); put(f'up-rf-{key}-max', num(max(abs(v) for v in x), 2, sign=False))
    a_, b_ = V['up-rf-rg-gt01'], V['up-rf-sim-gt01']
    put('up-rf-gt01-phrase', 'one problem each' if a_ == b_ == '1' else f'{a_} and {b_} problems')
    put('up-rf-conv', f"{100 * np.mean([r['converged'] for r in rr]):.0f}")

# ------------------------------------------------------------------ run time of the two point baselines (rows of the run-time table)
tm = jl(P / 'upgrade-timing/timing.jsonl')
assert len(tm) == 64 and all(r['at_ok'] and r['gate_k1'] <= 1e-6 and r['gate_at'] <= 1e-6 for r in tm)
put('up-rt-n', str(len(tm)))
_rr = [r['jfit5_seconds'] / r['jfit5_recorded'] for r in tm]
put('up-rt-ratio-lo', f'{min(_rr):.2f}'); put('up-rt-ratio-hi', f'{max(_rr):.2f}')
for M in (24, 48, 96, 192):
    R = [r for r in tm if r['M'] == M]
    assert len(R) == 16
    # seconds on this machine divided by the problem's calibration ratio (re-timed over recorded five-point fit)
    for f, key in (('at_seconds', 'at'), ('k1_seconds', 'k1')):
        put(f'up-rt-{key}-{M}', f"{np.median([r[f] * r['jfit5_recorded'] / r['jfit5_seconds'] for r in R]):.1f}")

# ------------------------------------------------------------------ revised fitting on every data set (posthoc/upgrade-revfit)
RV = {f: json.loads((P / f'upgrade-revfit/summary-{f}.json').read_text()) for f in ('mesh', 'miracle', 'sim')}
assert all(v['complete'] and v['self_check']['passed'] for v in RV.values())
cls = lambda c: f"{c.get('win', 0)}/{c.get('tie', 0)}/{c.get('loss', 0)}"
_n = 0
for f, lab, key, pool in (('mesh', 'mesh', 'rg', 'all_cells'), ('miracle', 'A1', 'A1', 'all_cells'), ('miracle', 'R2', 'R2', 'all_cells'),
                          ('sim', 'sim', 'sim', 'source_proximal')):
    x = RV[f]['sets'][lab]; _n += x['problems']
    d = x['paired_diff_db']; q = d['pooled95'][pool]
    _d = 3 if abs(q['mean']) < 0.05 else 2
    put(f'up-rv-{key}', iv(q['mean'], q['low'], q['high'], _d)); put(f'up-rv-{key}-mean', num(q['mean'], _d))
    put(f'up-rv-{key}-cls-reg', cls(x['classes']['registered'])); put(f'up-rv-{key}-cls-rev', cls(x['classes']['revised']))
    put(f'up-rv-{key}-nchg', str(len(x['classes']['changed_in_family'])))
    put(f'up-rv-{key}-ord', f"{100 * x['orders']['changed_fraction']:.0f}")
    put(f'up-rv-{key}-min', num(d['problems']['min'], 1)); put(f'up-rv-{key}-max', num(d['problems']['max'], 1))
    put(f'up-rv-{key}-pcap', f"{100 * x['convergence_revised']['point_cap_fraction']:.1f}")
    put(f'up-rv-{key}-rcap', f"{100 * x['convergence_revised']['region_cap_fraction']:.1f}")
    assert x['stable']['stable'] == (key in ('rg', 'sim'))
    if f == 'miracle':
        for g, v in d['per_group'].items():
            put(f"up-rv-{key}-b{g.replace('Hz', '')}", num(v['pooled95']['mean']))
put('up-rv-n', f'{_n:,}')

# ------------------------------------------------------------------ distance-control study on the far anechoic grid (studies/TB-M4-MIRACLE-A2)
M4 = json.loads((S / 'studies/TB-M4-MIRACLE-A2/primary/summary.json').read_text())
assert M4['n_errors_a2'] == 0 and M4['n_rows_a2'] == 768 and not M4['stop_rule']['triggered']
assert M4['rules']['A2_2']['a1_reproduction_on_a2_box']['reproduced_1e6']
_q = json.loads((S / 'studies/TB-M4-MIRACLE-A2/extract/miracle-A2-receipt.json').read_text())['qualification']
assert _q['passed']
put('a2-da', f"{_q['d_over_a'][0]:.2f}--{_q['d_over_a'][1]:.2f}"); put('a2-dist', f"{_q['plane_to_array_m']:.2f}")
put('a2-n', str(M4['n_rows_a2']))
for sc in ('A1', 'A2', 'R2'):
    x = M4[sc]; g = x['best_generic']
    put(f'a2-{sc}-gen-cls', f"{g['wins']}/{g['ties']}/{g['losses']}"); put(f'a2-{sc}-gen-w', words(g['wins'], g['ties'], g['losses']))
    q = x['pooled_vs_best_generic_cell']; put(f'a2-{sc}-gen', iv(q['mean'], q['low'], q['high'])); put(f'a2-{sc}-gen-mean', num(q['mean']))
    for b, v in x['by_band'].items():
        put(f'a2-{sc}-gen-b{b}', iv(v['mean'], v['low'], v['high'])); put(f'a2-{sc}-gen-b{b}-mean', num(v['mean']))
    for k, kk in (('proposed', 'prop'), ('best_generic_of_condition', 'gen'), ('arrival_time_point', 'at')):
        put(f'a2-{sc}-med-{kk}', num(x['median_error_db'][k], 1))
    for k, kk in (('two_stage_point', 'tsp'), ('two_stage_region', 'tsr'), ('mo26_cap5', 'cap5'), ('gsmf', 'gsmf')):
        put(f'a2-{sc}-{kk}-iv', iv(x[k]['pooled']['mean'], x[k]['pooled']['low'], x[k]['pooled']['high']))
        put(f'a2-{sc}-{kk}', num(x[k]['pooled']['mean'])); put(f'a2-{sc}-{kk}-cls', f"{x[k]['wins']}/{x[k]['ties']}/{x[k]['losses']}")
_l = [c for c, v in M4['A2']['best_generic']['by_cell'].items() if v['cls'] == 'loss']
assert _l == ['500Hz-M48']
for k, kk in (('A2_2', 'dist-contrast'), ('A2_3', 'refl-contrast')):
    r = M4['rules'][k]; put(f'a2-{kk}', iv(r['mean'], r['low'], r['high'])); assert r['interval'] == 'above zero'
_h = M4['A2']['K_cv_hist']; put('a2-K5', f'{100 * _h[5] / sum(_h):.0f}')

# ------------------------------------------------------------------ gridded multi-frequency sparse model (posthoc/upgrade-mfsbl; A2 from the study)
MF = json.loads((P / 'upgrade-mfsbl/summary-seen.json').read_text())
assert all(MF['integrity'][k]['complete'] for k in ('room', 'A1', 'R2', 'sim'))
_caps_em, _caps_fp, _sf = [], [], []
def _mf(key, x, far=False):
    b = x['better_variant']
    put(f'mf-{key}-cls', f"{b['wins']}/{b['ties']}/{b['losses']}"); put(f'mf-{key}-w', words(b['wins'], b['ties'], b['losses']))
    q = x['pooled_vs_better_variant']; put(f'mf-{key}', iv(q['mean'], q['low'], q['high'])); put(f'mf-{key}-mean', num(q['mean']))
    _caps_em.append(x['fits']['mfsbl_em']['stopped_by_iteration_cap']); _caps_fp.append(x['fits']['mfsbl_mackay']['stopped_by_iteration_cap'])
    e = x['em_only_descriptive']; put(f'mf-{key}-em-cls', f"{e['wins']}/{e['ties']}/{e['losses']}"); put(f'mf-{key}-em', num(e['pooled']['mean']))
    f_ = x['fixed_point_only_descriptive']; put(f'mf-{key}-fp-cls', f"{f_['wins']}/{f_['ties']}/{f_['losses']}"); put(f'mf-{key}-fp', num(f_['pooled']['mean']))
    ls = [v['mean'] for v in b['by_cell'].values() if v['cls'] == 'loss']
    if ls:
        put(f'mf-{key}-loss-lo', num(min(ls), 1, sign=False)); put(f'mf-{key}-loss-hi', num(max(ls), 1, sign=False))
for key, s_ in (('rg', 'room'), ('A1', 'A1'), ('R2', 'R2'), ('sim', 'sim')):
    x = MF['sets'][s_]; _mf(key, x)
    for b, v in x.get('pooled_vs_better_variant_by_band', {}).items():
        put(f'mf-{key}-b{b}', num(v['mean']))
    r = x['revised_fit_vs_better_variant']
    put(f'mf-{key}-rev', iv(r['pooled']['mean'], r['pooled']['low'], r['pooled']['high'])); put(f'mf-{key}-rev-cls', f"{r['wins']}/{r['ties']}/{r['losses']}")
    g = x['guarded_sbl_minus_mfsbl']['better_minus_better']; put(f'mf-{key}-sf', num(g['mean'])); _sf.append(g['mean'])
    for m, mm in (('j_cv_ext', 'prop'), ('mfsbl_mackay', 'fp'), ('mfsbl_em', 'em')):
        put(f'mf-{key}-med-{mm}', num(x['error_db']['all'][m]['median'], 1))
_a2 = MF['far_grid_A2_registered']; assert _a2['rule_A2_4']['wins'] == M4['rules']['A2_4']['wins']
_mf('A2', _a2)
for b, v in M4['A2']['mfsbl'].get('pooled_vs_better_variant_by_band', {}).items():
    put(f'mf-A2-b{b}', num(v['mean']))
_sm = MF['sets']['sim']
assert _sm['better_variant']['comparator_wins_in'] == ['FF-13']
put('mf-sim-all-cls', f"{_sm['all_cells']['wins']}/{_sm['all_cells']['ties']}/{_sm['all_cells']['losses']}")
_fc = _sm['far_control_cell']['A1-beside1.2']; assert _fc['cls'] == 'tie'; put('mf-sim-far', num(_fc['mean']))
put('mf-sim-loss', num(_sm['better_variant']['by_cell']['FF-13']['mean']))
put('mf-sim-best', num(min(v['mean'] for v in _sm['better_variant']['by_cell'].values()), 1))
put('mf-sim-ncell', str(_sm['source_proximal_conditions']))
put('mf-em-cap-lo', f'{100 * min(_caps_em):.0f}'); put('mf-em-cap-hi', f'{100 * max(_caps_em):.0f}'); put('mf-fp-cap-hi', f'{100 * max(_caps_fp):.0f}')
put('mf-sf-lo', num(min(_sf), 1, sign=False)); put('mf-sf-hi', num(max(_sf), 1, sign=False))
_dg = MF['single_vs_multi_frequency_model']['dictionary_evidence']
assert _dg['problems_with_all_six_bin_node_counts_equal_to_band_dictionary'] == 0
TM = MF['same_machine_timing_A2']; assert TM['rows'] == 768
_r = []
for M in ('24', '48', '96', '192', 'all'):
    t = TM['by_microphones'][M]; assert t['zero_second_fixed_point_rows'] == 0 if 'zero_second_fixed_point_rows' in t else True
    put(f'tm-prop-{M}', f"{t['primary']:.0f}"); put(f'tm-fp-{M}', f"{t['mfsbl_mackay']:.1f}"); put(f'tm-em-{M}', f"{t['mfsbl_em']:.0f}")
    put(f'tm-ratio-{M}', f"{t['ratio_primary_over_fixed_point']['median']:.0f}")
    if M != 'all':
        _r.append(t['ratio_primary_over_fixed_point']['median'])
put('tm-ratio-lo', f'{min(_r):.0f}'); put('tm-ratio-hi', f'{max(_r):.0f}')
put('tm-ratio-em', f"{TM['by_microphones']['all']['ratio_primary_over_em']['median']:.1f}")

_xc = json.loads((S / 'studies/TB-M4-MIRACLE-A2/xcheck/xcheck-compare.json').read_text())
assert _xc['passed'] and _xc['identical'] == _xc['problems'] and _xc['worst_rel'] == 0.0
put('a2-xc-n', str(_xc['problems']))
_dc = json.loads((S / 'studies/TB-M4-MIRACLE-A2/primary/a1-check/dry-compare.json').read_text())
assert _dc['reproduced_1e6'] and _dc['task_list']['passed']; put('a2-a1check-n', str(_dc['problems']))
_g = M4['A2']['mfsbl']['guarded_sbl_minus_mfsbl']['sbl_mackay_common-mfsbl_mackay']; put('mf-A2-sf-fp', num(_g['mean']))
for key, s_ in (('rg', 'room'), ('A1', 'A1'), ('R2', 'R2'), ('sim', 'sim')):
    put(f'mf-{key}-emcap', f"{100 * MF['sets'][s_]['fits']['mfsbl_em']['stopped_by_iteration_cap']:.0f}")
put('mf-A2-emcap', f"{100 * MF['far_grid_A2_registered']['fits']['mfsbl_em']['stopped_by_iteration_cap']:.0f}")

# means of two pooled margins already printed with intervals by make_numbers_v2 (for the configuration table)
import re
_v2 = dict(re.findall(r'\\vdef\{([^}]+)\}\{(.*)\}\s*$', (OUT / 'v2-numbers.tex').read_text(), flags=re.M))
for key, src in (('tb-rg-gen', 'meas-M1c-pooled-cell'), ('tb-sim-gen', 'sm-all')):
    put(key, re.match(r'\$[^$]+\$', _v2[src]).group(0))
assert V['a2-A1-gen-mean'] == re.match(r'\$[^$]+\$', _v2['m3-A1-bg-pooled']).group(0)
assert V['a2-R2-gen-mean'] == re.match(r'\$[^$]+\$', _v2['m3-R2-bg-pooled']).group(0)

# unsigned magnitudes for the abstract and conclusion (the sign is carried by the sentence); each is checked for its sign
def _mag(key, src, sign, d=2):
    x = float(src.replace('$', '').replace('+', ''))
    assert (x < 0) == (sign < 0), (key, src)
    put(key, f'{abs(x):.{d}f}')
for key, src, sign in (('ab-sim', V['tb-sim-gen'], -1), ('ab-rg', V['tb-rg-gen'], -1), ('ab-A1', V['a2-A1-gen-mean'], -1),
                       ('ab-A2', V['a2-A2-gen-mean'], -1), ('ab-R2', V['a2-R2-gen-mean'], -1),
                       ('ab-mf-sim', V['mf-sim-mean'], -1), ('ab-mf-rg', V['mf-rg-mean'], -1), ('ab-mf-A1', V['mf-A1-mean'], -1),
                       ('ab-mf-A2', V['mf-A2-mean'], +1), ('ab-mf-R2', V['mf-R2-mean'], +1),
                       ('ab-rv-A1', V['up-rv-A1-mean'], -1), ('ab-rv-R2', V['up-rv-R2-mean'], -1)):
    _mag(key, src, sign)
put('ab-rv-small', f"{max(abs(RV[f]['sets'][l]['paired_diff_db']['pooled95'][q][e]) for f, l, q in (('mesh', 'mesh', 'all_cells'), ('sim', 'sim', 'source_proximal')) for e in ('low', 'high')):.2f}")
put('ab-dist', f"{M4['rules']['A2_2']['mean']:.2f}"); put('ab-refl', f"{M4['rules']['A2_3']['mean']:.2f}")
assert abs(float(V['ab-A1']) - float(_v2['abs-m3-A1'])) < 1e-9

with open(OUT / 'upgrade-numbers.tex', 'w') as fh:
    fh.write('% Generated by tools/make_numbers_upgrade.py from the saved post hoc revision outputs; do not edit.\n')
    for k in sorted(V):
        fh.write(f'\\vdef{{{k}}}{{{V[k]}}}\n')
print(len(V), 'numbers')
