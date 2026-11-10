"""Post hoc analysis of the multi-frequency SBL comparator on the seen sets (PLAN.md, "Later"): MIRACLE A1 and R2, the room
grid (TB-M2 problems) and, once its rows are complete, the simulation (TB-S2 problems). Nothing here is a registered result.

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 -B posthoc/upgrade-mfsbl/analyze_mfsbl.py [ROWS] [--a2-summary FILE]
    ROWS: rows.jsonl of `run_mfsbl.py --sets ...` (default: rows.jsonl beside this file).
    -> summary-seen.json beside this file; the README tables are printed to stdout.

The statistics are the studies' own, imported: analyze_v2 (interval, classes, pooled comparison), analyze_m3 (pooled
comparison against a per-condition member) and analyze_m4 (`comparator()`: the A2-4 computation, with its rule for
non-finite rows and "ties go to mfsbl_mackay"). The comparator errors are attached to the registered rows, so every paired
difference is taken against the registered error of the same problem. A2 is not recomputed: its registered summary is copied.
"""
from __future__ import annotations
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
sys.path.insert(0, str(S)); sys.path.insert(0, str(S / 'studies' / 'TB-M3-MIRACLE')); sys.path.insert(0, str(S / 'studies' / 'TB-M4-MIRACLE-A2'))
import analyze_v2 as AV  # noqa: E402
import analyze_m3 as AM3  # noqa: E402
import analyze_m4 as AM4  # noqa: E402

P, MF = AV.PRIMARY, AM4.MF
EXPECTED = dict(room=1024, A1=768, R2=768, sim=2048)
GUARD = dict(room=('sbl_mackay_guard', 'sbl_em_guard'), A1=('sbl_mackay_common', 'sbl_em_common'), R2=('sbl_mackay_common', 'sbl_em_common'),
             sim=('sbl_mackay_common', 'sbl_em_common'))            # order: ties go to the fixed-point rule, as for MF
GUARD_NOTE = dict(room='target-guarded SBL of TB-M2-SBLFIX (sbl_*_guard)', A1='target-guarded SBL of TB-M3 (stored as sbl_*_common)',
                  R2='target-guarded SBL of TB-M3 (stored as sbl_*_common)',
                  sim='UNGUARDED SBL of TB-S2 (sbl_*_common), the registered comparator there; no guarded rows exist for the simulation')
REV = 'j_cv_ext_revised'
REVFIT = dict(room=S / 'posthoc/upgrade-revfit/revfit-mesh.jsonl', A1=S / 'posthoc/upgrade-revfit/revfit-miracle.jsonl',
              R2=S / 'posthoc/upgrade-revfit/revfit-miracle.jsonl', sim=S / 'posthoc/upgrade-revfit/revfit-sim.jsonl')
PRIMARY_TIME = 'j_fit + j_cv + j_ext_K{k}, k = info.j_K.cv of the registered row (no extension term when k = 0); j_laplace not included'


def key(set_, r):                                                   # as run_mfsbl.key
    return (set_, r['cell'], r['seed']) if set_ == 'sim' else (set_, r['source'], float(r['band']), r['M'], r['draw'], r['seed'])


def registered():
    """Set -> {problem key: registered row with r['e'] in dB}; the room rows also get the TB-M2-SBLFIX guarded errors."""
    m3, _ = AV.load(S / 'studies/TB-M3-MIRACLE/primary')
    m2, _ = AV.load(S / 'studies/TB-M2/primary')
    s2, _ = AV.load(S / 'studies/TB-S2/primary')
    out = dict(A1={key('A1', r): r for r in m3 if r['scenario'] == 'A1'}, R2={key('R2', r): r for r in m3 if r['scenario'] == 'R2'},
               room={key('room', r): r for r in m2 if r['kind'] == 'mesh'}, sim={key('sim', r): r for r in s2 if r['kind'] == 'sim'})
    for l in open(S / 'studies/TB-M2-SBLFIX/primary/rows.jsonl'):
        r = json.loads(l)
        if 'error' not in r:
            out['room'][key('room', r)]['e'].update({m: AV.db(v) for m, v in r['nmse'].items()})
    return out


def add_revised(set_, reg):
    """Adds the revised fit's error (dB) to the registered rows; returns the check of its stored registered error."""
    if set_ not in REVFIT or not REVFIT[set_].exists():
        return dict(available=False)
    n, worst = 0, 0.0
    for l in open(REVFIT[set_]):
        r = json.loads(l)
        k = r['key']
        if 'error' in r or r.get('fitting') != 'revised' or (set_ in ('A1', 'R2') and k.get('scenario') != set_):
            continue
        g = reg[key(set_, k)]
        g['e'][REV] = AV.db(r['nmse'])
        worst = max(worst, abs(AV.db(r['nmse_reg']) - g['e'][P]) / abs(g['e'][P])); n += 1
    return dict(available=True, file=str(REVFIT[set_].relative_to(S)), rows=n, complete=bool(n == len(reg)),
                nmse_reg_worst_relative_difference_to_registered=f'{worst:.3e}')


def same_machine_timing(path):
    """Far-grid (A2) rows: the primary and both comparator variants timed in the same run on one machine."""
    if not path.exists():
        return dict(available=False, path=str(path))
    rs = [r for r in map(json.loads, open(path)) if 'error' not in r and all(m in r['timing'] for m in MF)]
    out = dict(available=True, rows=len(rs), set='A2 (TB-M4-MIRACLE-A2 run, one machine, same run)', primary_seconds_definition=PRIMARY_TIME, by_microphones={})
    for M in sorted({r['M'] for r in rs}) + ['all']:
        v = [r for r in rs if M == 'all' or r['M'] == M]
        pr = np.array([primary_seconds(r) for r in v]); fp = np.array([r['timing']['mfsbl_mackay'] for r in v]); em = np.array([r['timing']['mfsbl_em'] for r in v])
        q = lambda x: dict(median=float(np.median(x)), p05=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95)))   # noqa: E731
        out['by_microphones'][str(M)] = dict(n=len(v), primary=float(np.median(pr)), mfsbl_mackay=float(np.median(fp)), mfsbl_em=float(np.median(em)),
                                             ratio_primary_over_fixed_point=q(pr / fp), ratio_primary_over_em=q(pr / em),
                                             zero_second_fixed_point_rows=int((fp <= 0).sum()))
    return out


def dictionary_evidence(mf_room):
    """Room grid: node counts of the stored guarded single-frequency fits (one dictionary per bin, TB-M2-SBLFIX rows) beside the band dictionary."""
    mfn = {tuple(r['key'][1:]): r['info']['mfsbl_mackay']['nodes'] for r in mf_room if 'error' not in r}
    n = same = 0
    ex = {}
    for l in open(S / 'studies/TB-M2-SBLFIX/primary/rows.jsonl'):
        r = json.loads(l)
        if 'error' in r:
            continue
        k = (r['source'], float(r['band']), r['M'], r['draw'], r['seed'])
        per = [d['nodes'] for d in r['dictionary']]
        n += 1; same += all(p == mfn[k] for p in per)
        ex.setdefault(f"{int(r['band'])}Hz", dict(single_frequency_nodes_per_bin=per, multi_frequency_nodes=mfn[k]))
    return dict(set='room grid (the only seen set whose single-frequency rows store their dictionary sizes)', problems=n,
                problems_with_all_six_bin_node_counts_equal_to_band_dictionary=same, example_first_row_per_band=ex)


SHARED = dict(
    identical=[
        'atom types: free-field monopoles on a regular grid plus plane waves on a Fibonacci sphere; monopole columns divided by their RMS over the microphones',
        'node spacing rule: a quarter wavelength, coarsened by E.grid_nodes (spacing x 1.1 until at most v2.SBL_NODES = 6,000 nodes)',
        'guard on A1, R2 and the room grid: nodes within E.GUARD = 5 cm of any microphone or any target removed (sbl_guard)',
        'noise model: complex Gaussian, one variance per bin, floored at 1e-12 x the bin mean power; same update per rule',
        'gamma update per rule (EM: |mu|^2 + posterior variance; fixed point: |mu|^2 / (1 - variance/gamma)), here summed over bins',
        'initialisation: gamma = 10 x mean power / N, noise = 0.01 x mean power',
        'pruning at gamma <= 1e-8 max gamma; stop at max|d gamma| / max gamma < 1e-4; caps 200 (EM) and 1,000 (fixed point) iterations',
        'prediction: posterior mean at the targets',
        'check 1 (checks.json): with ONE bin the multi-frequency code equals sbl_guarded (relative difference <= 4.2e-12, same iterations and active sets; 6 cases)'],
    different_besides_gamma_coupling=[
        'wavelength of the node grid: single-frequency uses each bin\'s own wavelength (lambda_f / 4), so six different grids per band; multi-frequency uses one grid at sqrt(lambda_min lambda_max) / 4 for all six bins (coarser than lambda_f/4 in the upper bins, finer in the lower bins, before the 6,000 cap)',
        'search box: single-frequency enlarges the region box by max(lambda_f, 1.5 m) per bin; multi-frequency by max(lambda at the lowest bin, 1.5 m) for all bins (differs when lambda_f > 1.5 m, i.e. below about 229 Hz)',
        'plane waves: single-frequency takes the count from each bin\'s own k R; multi-frequency from the highest bin\'s k for all bins (more directions in the lower bins)',
        'guard removal is applied once to the band grid instead of once per bin grid, so the removed nodes differ',
        'pruning and stopping are joint: an atom is pruned for all bins at once and one iteration count and one stopping test serve the band; single-frequency prunes and stops each bin separately (each bin has its own 200 / 1,000 cap)',
        'each bin is divided by the root of its mean microphone power before the fit (no effect with one bin)',
        'simulation only: the stored single-frequency SBL is UNGUARDED (microphone guard only), the multi-frequency dictionary also removes nodes within 5 cm of targets'],
    isolates_cross_frequency_coupling=False,
    statement='No. Check 1 shows the code reduces to the guarded single-frequency fit when it is given one bin, but the full-set single-frequency rows were '
              'fitted bin by bin with a per-bin dictionary (own wavelength, own box, own plane-wave count), not with the band dictionary of the '
              'multi-frequency runs. The single-minus-multi difference therefore mixes the shared gamma with the change of dictionary, and does not isolate the coupling.')


def geometry():
    p = S / 'posthoc/geometry/geometry.json'
    if not p.exists():
        return dict(available=False)
    g = json.loads(p.read_text())
    return dict(available=True, file='posthoc/geometry/geometry.json (read only)',
                definitions='a_m: region radius; d_over_a: source distance from the region centre over a; clearance: source to the nearest region point (microphone or target)',
                room_grid=g.get('measured'), simulation_cells=len(g.get('simulation', {})),
                A1='absent from geometry.json', A2='absent from geometry.json', R2='absent from geometry.json',
                miracle_note='geometry.json holds the simulation and the room grid (MeshRIR) only. TB-M4 PROTOCOL.md states the grid plane is 0.734 m '
                             'from the array on A1 and 1.467 m on A2 and R2 (R2 prep: 1.474 m); no extent ratio or wavelength figure is stored for them there.')


def primary_seconds(r):
    k = r['info']['j_K']['cv']
    t = r['timing']
    return t['j_fit'] + t['j_cv'] + (t[f'j_ext_K{k}'] if k > 0 else 0.0)


def integrity(set_, mf, reg):
    keys = [tuple(r['key']) for r in mf]
    good = [r for r in mf if 'error' not in r]
    worst, missing_key, mism = 0.0, 0, 0
    for r in good:
        ref = reg.get(tuple(r['key']))
        if ref is None:
            missing_key += 1; continue
        for m, v in r['stored_e'].items():
            rel = abs(v - ref['e'][m]) / max(abs(ref['e'][m]), 1e-300)
            worst = max(worst, rel)
            mism += rel > 1e-9 and m == P
    return dict(rows=len(mf), expected=EXPECTED[set_], complete=bool(len(set(keys)) == EXPECTED[set_] and set(keys) == set(reg)),
                registered_problems=len(reg), duplicate_keys=len(keys) - len(set(keys)), error_rows=len(mf) - len(good),
                keys_not_in_registered_rows=missing_key,
                non_finite={m: int(sum(not np.isfinite(r['e'].get(m, np.nan)) for r in good)) for m in MF},
                stored_primary_mismatches_1e9=int(mism), stored_e_worst_relative_difference=f'{worst:.3e}',
                stored_primary_equals_registered=bool(mism == 0 and missing_key == 0),
                all_rows_flagged_post_hoc=bool(all(r.get('post_hoc') is True for r in good)))


def attach(mf, reg):
    """Registered rows with the comparator's errors, info and timing added (the row format analyze_m4.comparator expects)."""
    rs = []
    for r in mf:
        if 'error' in r or tuple(r['key']) not in reg:
            continue
        g = reg[tuple(r['key'])]
        g['e'].update(r['e']); g['info'].update(r['info']); g['mf_timing'] = r['timing']; g['M_mics_mf'] = r['M_mics']
        rs.append(g)
    return rs


def pooled_pair(by, cells, a_of, b_of, kind):
    """analyze_m3.pooled_bg with a per-condition member on both sides: per unit, mean over conditions; 95 % t interval."""
    per = defaultdict(list)
    for c in cells:
        for u, v in AV.cell_diff(by[c], kind, a_of(c), b_of(c)).items():
            per[u].append(v)
    m, lo, hi = AV.tint([np.mean(v) for v in per.values()], AV.ALPHA)
    return dict(mean=m, low=lo, high=hi, units=len(per), cells=len(cells))


def classes_k(by, cells, other, kind, family_n, a=P):
    """analyze_m4.classes for any unit kind and Bonferroni family size (used for the simulation, whose units are seeds)."""
    out = {}
    for c in cells:
        m, lo, hi = AV.tint(list(AV.cell_diff(by[c], kind, a, other(c)).values()), AV.ALPHA / family_n)
        out[c] = dict(against=other(c), n_rows=len(by[c]), mean=m, low=lo, high=hi, cls=AV.classify(m, lo, hi), non_inferior=bool(hi < AV.NI_MARGIN))
    cl = [v['cls'] for v in out.values()]
    return dict(wins=cl.count('win'), ties=cl.count('tie'), losses=cl.count('loss'), by_cell=out)


def med_mean(x):
    return dict(median=float(np.median(x)), mean=float(np.mean(x)), n=len(x))


def describe(rs, set_, groups):
    """Absolute errors, process seconds and fit diagnostics (requirements e, f, g). groups: label -> rows."""
    methods = (P,) + MF + GUARD[set_]
    err = {g: {m: med_mean([r['e'][m] for r in v]) for m in methods} for g, v in groups.items()}
    ms = sorted({r['M_mics_mf'] for r in rs})
    sec = {}
    for M in ms + ['all']:
        v = [r for r in rs if M == 'all' or r['M_mics_mf'] == M]
        sec[str(M)] = dict(n=len(v), primary=float(np.median([primary_seconds(r) for r in v])),
                           **{m: float(np.median([r['mf_timing'][m] for r in v])) for m in MF})
    fits = {m: dict(stopped_by_iteration_cap=float(np.mean([not r['info'][m]['stopped_by_tolerance'] for r in rs])),
                    iterations_median=float(np.median([r['info'][m]['iterations'] for r in rs])),
                    iterations_mean=float(np.mean([r['info'][m]['iterations'] for r in rs])),
                    active_median=float(np.median([r['info'][m]['active'] for r in rs])),
                    active_mean=float(np.mean([r['info'][m]['active'] for r in rs]))) for m in MF}
    return dict(error_db=err, median_process_seconds_by_microphones=sec, primary_seconds_definition=PRIMARY_TIME, fits=fits,
                timing_note='process seconds; comparator timed on the 2026-11-09 box, the primary in the registered run of its study '
                            '(different machines and loads), so the two columns are not a controlled timing comparison')


def measured(rs, set_):
    """A1, R2 and the room grid: units are sources/loudspeakers; the computation of rule A2-4 (analyze_m4.comparator)."""
    cells = AM4.cells_of(rs)
    bands = sorted({c.split('Hz')[0] for c in cells}, key=int)
    res = AM4.comparator(rs, cells)
    res.pop('extended_best_generic', None); res.pop('guarded_sbl_minus_mfsbl', None)   # recomputed below with this set's guarded keys
    res.update(conditions=len(cells), units=len({r['source'] for r in rs}), unit='source' if set_ != 'room' else 'loudspeaker (two draws averaged)',
               bonferroni_family=len(cells))
    if 'better_variant' not in res:
        return res
    ok = [r for r in rs if all(np.isfinite(r['e'][m]) for m in MF)]
    by = AM4.by_cell(ok)
    better = res['better_variant']['member_by_cell']
    bc = lambda b: [c for c in cells if c.startswith(b + 'Hz')]                        # noqa: E731
    g = GUARD[set_]
    gbest = {c: min(g, key=lambda m: float(np.mean([r['e'][m] for r in by[c]]))) for c in cells}
    res['pooled_vs_better_variant_by_band'] = {b: AM3.pooled_bg(by, bc(b), better) for b in bands}
    for name, m in (('fixed_point_only_descriptive', 'mfsbl_mackay'), ('em_only_descriptive', 'mfsbl_em')):
        res[name]['pooled_by_band'] = {b: AV.pooled(ok, 'mesh', bc(b), P, m) for b in bands}
    res['guarded_sbl_minus_mfsbl'] = dict(
        single_frequency_sbl=GUARD_NOTE[set_], better_single_frequency_by_cell=gbest,
        better_minus_better=pooled_pair(by, cells, lambda c: gbest[c], lambda c: better[c], 'mesh'),
        better_minus_better_by_band={b: pooled_pair(by, bc(b), lambda c: gbest[c], lambda c: better[c], 'mesh') for b in bands},
        same_rule={f'{a}-{b}': AV.pooled(ok, 'mesh', cells, a, b) for a, b in zip(g, MF)})
    if all(REV in r['e'] for r in ok):                                                 # revised fitting (posthoc/upgrade-revfit), same paired statistics
        rv = classes_k(by, cells, lambda c: better[c], 'mesh', len(cells), a=REV)
        res['revised_fit_vs_better_variant'] = dict(
            note='revised fit (posthoc/upgrade-revfit, post hoc) minus the better comparator variant; the better variant is the one chosen above',
            **rv, comparator_wins_in=[c for c, v in rv['by_cell'].items() if v['cls'] == 'loss'],
            pooled=pooled_pair(by, cells, lambda c: REV, lambda c: better[c], 'mesh'),
            pooled_by_band={b: pooled_pair(by, bc(b), lambda c: REV, lambda c: better[c], 'mesh') for b in bands},
            median_error_db={k: float(np.median([r['e'][REV] for r in v])) for k, v in
                             {'all': ok, **{b: [r for r in ok if r['cell_key'].startswith(b + 'Hz')] for b in bands}}.items()})
    res.update(describe(ok, set_, {'all': ok, **{b: [r for r in ok if r['cell_key'].startswith(b + 'Hz')] for b in bands}}))
    return res


def simulation(rs):
    """TB-S2 as analyze_v2 does it: units are the seeds of a cell, the family is the source-proximal cells, the far cell apart."""
    for r in rs:
        r['cell_key'] = r['cell']
    cells = list(dict.fromkeys(r['cell'] for r in rs))
    prox = [c for c in cells if c != AV.FAR]
    ok = [r for r in rs if all(np.isfinite(r['e'][m]) for m in MF)]
    res = dict(rows=len(rs), rows_used=len(ok), rows_excluded=len(rs) - len(ok), excluded_fraction=(len(rs) - len(ok)) / len(rs),
               conditions=len(cells), source_proximal_conditions=len(prox), unit='seed (field)', bonferroni_family=len(prox))
    res['evaluable'] = bool(res['excluded_fraction'] <= AM4.MF_EXCLUDE_MAX and {r['cell_key'] for r in ok} == set(cells))
    if not res['evaluable']:
        return res
    by = AM4.by_cell(ok)
    mean = {c: {m: float(np.mean([r['e'][m] for r in by[c]])) for m in MF + GUARD['sim']} for c in cells}
    better = {c: min(MF, key=lambda m: mean[c][m]) for c in cells}
    gbest = {c: min(GUARD['sim'], key=lambda m: mean[c][m]) for c in cells}
    fam = sorted({c.split('-')[0] for c in prox})
    fc = lambda a: [c for c in prox if c.split('-')[0] == a]                           # noqa: E731
    bv = classes_k(by, prox, lambda c: better[c], 'sim', len(prox))
    res.update(better_variant=dict(**bv, member_by_cell=better, comparator_wins_in=[c for c, v in bv['by_cell'].items() if v['cls'] == 'loss']),
               far_control_cell=classes_k(by, [AV.FAR], lambda c: better[c], 'sim', len(cells))['by_cell'] if AV.FAR in cells else None,
               pooled_vs_better_variant=pooled_pair(by, prox, lambda c: P, lambda c: better[c], 'sim'),
               pooled_vs_better_variant_by_family={a: pooled_pair(by, fc(a), lambda c: P, lambda c: better[c], 'sim') for a in fam},
               fixed_point_only_descriptive=dict(**classes_k(by, prox, lambda c: 'mfsbl_mackay', 'sim', len(prox)), pooled=AV.pooled(ok, 'sim', prox, P, 'mfsbl_mackay')),
               em_only_descriptive=dict(**classes_k(by, prox, lambda c: 'mfsbl_em', 'sim', len(prox)), pooled=AV.pooled(ok, 'sim', prox, P, 'mfsbl_em')),
               guarded_sbl_minus_mfsbl=dict(
                   single_frequency_sbl=GUARD_NOTE['sim'], better_single_frequency_by_cell=gbest,
                   better_minus_better=pooled_pair(by, prox, lambda c: gbest[c], lambda c: better[c], 'sim'),
                   same_rule={f'{a}-{b}': AV.pooled(ok, 'sim', prox, a, b) for a, b in zip(GUARD['sim'], MF)}))
    al = classes_k(by, cells, lambda c: better[c], 'sim', len(cells))
    res['all_cells'] = dict(note='all cells, far control included; Bonferroni over all cells (as TB-S2 rule S6)', conditions=len(cells),
                            wins=al['wins'], ties=al['ties'], losses=al['losses'], comparator_wins_in=[c for c, v in al['by_cell'].items() if v['cls'] == 'loss'],
                            ties_in=[c for c, v in al['by_cell'].items() if v['cls'] == 'tie'], by_cell=al['by_cell'],
                            pooled=pooled_pair(by, cells, lambda c: P, lambda c: better[c], 'sim'),
                            single_frequency_minus_mfsbl=pooled_pair(by, cells, lambda c: gbest[c], lambda c: better[c], 'sim'))
    res['better_variant']['ties_in'] = [c for c, v in bv['by_cell'].items() if v['cls'] == 'tie']
    if all(REV in r['e'] for r in ok):                                                 # revised fitting (posthoc/upgrade-revfit), same paired statistics
        rv = classes_k(by, prox, lambda c: better[c], 'sim', len(prox), a=REV)
        res['revised_fit_vs_better_variant'] = dict(
            note='revised fit (posthoc/upgrade-revfit, post hoc) minus the better comparator variant, source-proximal cells; the better variant is the one chosen above',
            wins=rv['wins'], ties=rv['ties'], losses=rv['losses'],
            pooled=pooled_pair(by, prox, lambda c: REV, lambda c: better[c], 'sim'))
    res['units'] = len({r['seed'] for r in ok})
    res['units_note'] = 'the 32 seeds are shared by the cells; pooled: per seed, mean over the pooled cells, as analyze_v2.pooled'
    res.update(describe(ok, 'sim', {'all': ok, **{a: [r for r in ok if r['cell_key'].split('-')[0] == a] for a in fam}}))
    return res


def far_grid(path):
    """The registered A2 result, copied from its summary (not recomputed)."""
    if not path.exists():
        return dict(available=False, path=str(path))
    s = json.loads(path.read_text())
    m = s['A2']['mfsbl']
    keep = ('rows', 'rows_used', 'rows_excluded', 'fits', 'better_variant', 'pooled_vs_better_variant', 'fixed_point_only_descriptive',
            'em_only_descriptive', 'guarded_sbl_minus_mfsbl')
    return dict(available=True, standing='registered (TB-M4-MIRACLE-A2, rule A2-4); copied from its summary.json, not recomputed',
                rule_A2_4=s['rules']['A2_4'], **{k: m[k] for k in keep if k in m})


def rnd(o):
    if isinstance(o, dict):
        return {str(k): rnd(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [rnd(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return round(float(o), 4) if np.isfinite(o) else None
    return o


def iv(d):
    return f"{d['mean']:+.2f} [{d['low']:+.2f}, {d['high']:+.2f}]"


def markdown(res):
    L = []
    for s in ('room', 'A1', 'R2', 'sim'):
        r = res['sets'].get(s)
        if not r or 'better_variant' not in r:
            continue
        b = r['better_variant']
        L += [f"### {s} (post hoc; {r['rows_used']} problems, {r['units'] if 'units' in r else ''} units: {r['unit']})", '',
              f"Primary against the better variant: **{b['wins']} wins, {b['ties']} ties, {b['losses']} losses** in "
              f"{len(b['by_cell'])} conditions; pooled {iv(r['pooled_vs_better_variant'])} dB. "
              f"Fixed point alone: {r['fixed_point_only_descriptive']['wins']}/{r['fixed_point_only_descriptive']['ties']}/"
              f"{r['fixed_point_only_descriptive']['losses']}, pooled {iv(r['fixed_point_only_descriptive']['pooled'])} dB. "
              f"EM alone: {r['em_only_descriptive']['wins']}/{r['em_only_descriptive']['ties']}/{r['em_only_descriptive']['losses']}, "
              f"pooled {iv(r['em_only_descriptive']['pooled'])} dB. Single-frequency SBL minus multi-frequency (better of each per "
              f"condition): {iv(r['guarded_sbl_minus_mfsbl']['better_minus_better'])} dB.", '',
              '| condition | better variant | primary − better variant (dB), interval | class for the primary |', '|---|---|---|---|']
        L += [f"| {c} | `{v['against']}` | {iv(v)} | {v['cls']} |" for c, v in b['by_cell'].items() if s != 'sim' or v['cls'] != 'win']
        if s == 'sim':
            L += [f"| the other {b['wins']} source-proximal cells | | (in `summary-seen.json`) | win |"]
        if 'pooled_vs_better_variant_by_band' in r:
            L += ['', '| band (Hz) | pooled primary − better variant (dB) | single-frequency − multi-frequency (dB) | median error (dB): primary / fixed point / EM |',
                  '|---|---|---|---|']
            for k, v in list(r['pooled_vs_better_variant_by_band'].items()) + [('all', r['pooled_vs_better_variant'])]:
                g = r['guarded_sbl_minus_mfsbl']['better_minus_better_by_band'][k] if k != 'all' else r['guarded_sbl_minus_mfsbl']['better_minus_better']
                e = r['error_db'][k]
                L.append(f"| {k} | {iv(v)} | {iv(g)} | {e[P]['median']:.2f} / {e['mfsbl_mackay']['median']:.2f} / {e['mfsbl_em']['median']:.2f} |")
        if 'revised_fit_vs_better_variant' in r:
            v = r['revised_fit_vs_better_variant']
            L += ['', f"Revised fit (post hoc) against the better variant: **{v['wins']} wins, {v['ties']} ties, {v['losses']} losses**; pooled "
                      f"{iv(v['pooled'])} dB (registered fit: {iv(r['pooled_vs_better_variant'])}); by band " +
                      '; '.join(f"{b} Hz {iv(x)}" for b, x in v.get('pooled_by_band', {}).items()) + '. Comparator better in: ' +
                      (', '.join(v.get('comparator_wins_in', [])) or 'none listed') + '.']
        if 'all_cells' in r:
            a = r['all_cells']
            L += ['', f"All {a['conditions']} cells (far control included): **{a['wins']} wins, {a['ties']} ties, {a['losses']} losses**; pooled {iv(a['pooled'])} dB. "
                      f"Comparator better in: {', '.join(a['comparator_wins_in']) or 'none'}. Ties: {', '.join(a['ties_in']) or 'none'}.",
                  '', '| cell family | pooled primary − better variant (dB) | median error (dB): primary / fixed point / EM |', '|---|---|---|']
            L += [f"| {k} | {iv(x)} | {r['error_db'][k][P]['median']:.2f} / {r['error_db'][k]['mfsbl_mackay']['median']:.2f} / {r['error_db'][k]['mfsbl_em']['median']:.2f} |"
                  for k, x in r['pooled_vs_better_variant_by_family'].items()]
        L += ['', '| microphones | median process seconds (different machines, no ratio): primary / fixed point / EM |', '|---|---|']
        L += [f"| {k} | {v['primary']:.1f} / {v['mfsbl_mackay']:.1f} / {v['mfsbl_em']:.1f} |" for k, v in r['median_process_seconds_by_microphones'].items()]
        f = r['fits']
        L += ['', 'Fits: ' + '; '.join(f"`{m}` stopped by the iteration cap in {100 * f[m]['stopped_by_iteration_cap']:.1f} %, median "
                                        f"{f[m]['iterations_median']:.0f} iterations, median {f[m]['active_median']:.0f} active atoms" for m in MF) + '.', '']
    return '\n'.join(L)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    if '--a2-summary' in sys.argv:
        a2 = Path(sys.argv[sys.argv.index('--a2-summary') + 1]); args.remove(str(a2))
    rows_path = Path(args[0]).expanduser() if args else HERE / 'rows.jsonl'
    if '--a2-summary' not in sys.argv:
        a2 = rows_path.resolve().parent.parent / 'm4' / 'summary.json'
    mf = defaultdict(list)
    for l in open(rows_path):
        r = json.loads(l)
        mf[r['key'][0]].append(r)
    reg = registered()
    res = dict(standing='POST HOC on seen data: the comparator was added after the registered analyses of these sets. Not registered results.',
               rows_file=str(rows_path), statistics='analyze_v2 / analyze_m3 / analyze_m4 functions, imported unchanged',
               conventions=dict(difference='first minus second in dB; negative favours the first', delta_db=AV.DELTA, alpha=AV.ALPHA,
                                per_condition='Student-t interval over units, Bonferroni over the conditions of the set',
                                pooled='per unit, mean over conditions; 95 % Student-t interval over units',
                                better_variant='lower mean error in the condition; ties go to mfsbl_mackay',
                                exclusion=f'rows with a non-finite comparator error are dropped; not evaluable above {AM4.MF_EXCLUDE_MAX:.0%}'),
               integrity={}, sets={}, unknown_sets={k: len(v) for k, v in mf.items() if k not in EXPECTED})
    for s in ('room', 'A1', 'R2', 'sim'):
        chk = integrity(s, mf.get(s, []), reg[s])
        res['integrity'][s] = chk
        if not chk['complete'] or chk['duplicate_keys'] or chk['error_rows']:
            res['sets'][s] = dict(status='not analysed: the set is incomplete or has duplicate or error rows', rows=chk['rows'], expected=chk['expected'],
                                  rows_per_condition=dict(Counter(str(r['key'][1]) for r in mf.get(s, []))) if s == 'sim' else None)
            continue
        res['integrity'][s]['revised_fit_rows'] = add_revised(s, reg[s])
        rs = attach(mf[s], reg[s])
        res['sets'][s] = dict(status='analysed (post hoc)', **(simulation(rs) if s == 'sim' else measured(rs, s)))
    res['far_grid_A2_registered'] = far_grid(a2)
    res['same_machine_timing_A2'] = same_machine_timing(a2.parent / 'rows.jsonl')
    res['single_vs_multi_frequency_model'] = dict(**SHARED, dictionary_evidence=dictionary_evidence(mf.get('room', [])))
    res['geometry'] = geometry()
    (HERE / 'summary-seen.json').write_text(json.dumps(rnd(res), indent=1))
    for s, c in res['integrity'].items():
        print('integrity', s, json.dumps(rnd(c)))
    print(markdown(res))
    t = res['same_machine_timing_A2']
    if t['available']:
        print('| microphones (A2, same machine) | median s: primary / fixed point / EM | ratio primary / fixed point: median [5 %, 95 %] |\n|---|---|---|')
        for k, v in t['by_microphones'].items():
            q = v['ratio_primary_over_fixed_point']
            print(f"| {k} | {v['primary']:.1f} / {v['mfsbl_mackay']:.1f} / {v['mfsbl_em']:.1f} | {q['median']:.1f} [{q['p05']:.1f}, {q['p95']:.1f}] |")
    print(json.dumps(rnd(res['single_vs_multi_frequency_model']['dictionary_evidence'])))
    f = res['far_grid_A2_registered']
    if f['available']:
        print(f"A2 (registered, copied): {f['rule_A2_4']['wins']}/{f['rule_A2_4']['ties']}/{f['rule_A2_4']['losses']} "
              f"pooled {iv(f['pooled_vs_better_variant'])} dB")


if __name__ == '__main__':
    main()
