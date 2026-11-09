"""Analysis of the review-3 refits (PLAN.md, part B) and of the reproduction (PLAN-C-REPRODUCTION.md). Written and
committed before either output was read. The registered statistics of analyze_v2.py and analyze_followups.py are
imported unchanged.

    python3 -B posthoc/review3/analyze_review3.py refit       ->  posthoc/review3/refit-summary.json
    python3 -B posthoc/review3/analyze_review3.py reproduction ->  posthoc/review3/reproduction-summary.json
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
sys.path.insert(0, str(S / 'common')); sys.path.insert(0, str(S))
import analyze_v2 as AV  # noqa: E402
import analyze_followups as AF  # noqa: E402

GATE = 1e-12
GUARD = 0.05


def db(v):
    v = np.asarray(v, float)
    return float(10 * np.log10(np.mean(v))) if np.all(np.isfinite(v)) and np.all(v < 1e6) else np.nan


def span(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return None
    return dict(n=int(len(v)), median=float(np.median(v)), p95=float(np.percentile(v, 95)), max=float(v.max()), min=float(v.min()))


def rkey(study, r):
    k = r['key'] if 'key' in r else r
    if study == 'TB-S2' or k.get('kind') == 'sim':
        return ('sim', k['cell'], k['seed'])
    return (k['kind'], k['source'], float(k['band']), k['M'], k['draw'])


def refit():
    new = [json.loads(l) for l in open(HERE / 'refit' / 'rows.jsonl')]
    errs = [r for r in new if 'error' in r]
    new = {rkey(r['study'], r): r for r in new if 'error' not in r}
    out = dict(status='post hoc', rows=len(new), errors=len(errs))
    # gates
    g_adk = [k for k, r in new.items() if r['adk']['gate_rel'] > GATE]
    g_prop = [k for k, r in new.items() if max(r['gate_rel'].values()) > GATE]
    out['gate'] = dict(tolerance=GATE, adk_failed=len(g_adk), proposed_failed=len(g_prop),
                       adk_max=max(r['adk']['gate_rel'] for r in new.values()),
                       proposed_max=max(max(r['gate_rel'].values()) for r in new.values()))
    ok_adk = {k: r for k, r in new.items() if k not in set(g_adk)}
    ok_prop = {k: r for k, r in new.items() if k not in set(g_prop)}
    # B1 exposure
    expo = {}
    for study in ('TB-M2', 'TB-S2'):
        rs = [r for r in ok_adk.values() if r['study'] == study]
        bins = [(r, f) for r in rs for f in range(len(r['adk']['trace']))]
        clipped_any = [(r, f) for r, f in bins if r['adk']['trace'][f]['n_clipped_evals'] > 0]
        same_when_unclipped = all(r['adk']['registered'][f] == r['adk']['corrected'][f] for r, f in bins
                                  if r['adk']['trace'][f]['n_clipped_evals'] == 0)
        by = defaultdict(lambda: [0, 0])
        for r, f in bins:
            lab = (f"{int(r['key']['band'])}Hz-M{r['key']['M']}" if study == 'TB-M2' else r['key']['cell'])
            by[lab][1] += 1
            by[lab][0] += r['adk']['trace'][f]['n_clipped_evals'] > 0
        dchg = [db(r['adk']['corrected']) - db(r['adk']['registered']) for r in rs]
        expo[study] = dict(problems=len(rs), bins=len(bins), bins_clipped_at_some_evaluation=len(clipped_any),
                           bins_clipped_at_final_state=int(sum(r['adk']['final_clipped'][f] for r, f in bins)),
                           bins_clipped_at_final_state_corrected=int(sum(r['adk']['final_clipped_corrected'][f] for r, f in bins)),
                           problems_with_a_clipped_bin=int(sum(any(t['n_clipped_evals'] > 0 for t in r['adk']['trace']) for r in rs)),
                           unclipped_bins_identical=bool(same_when_unclipped),
                           change_db_corrected_minus_registered=span(dchg),
                           change_db_abs_over_0p1=int(sum(abs(x) > 0.1 for x in dchg if np.isfinite(x))),
                           by_condition={k: dict(clipped=v[0], bins=v[1]) for k, v in by.items() if v[0]})
    out['B1_exposure'] = expo
    # B1 rules recomputed with the corrected stand-in
    out['B1_rules'] = rules_with_corrected_adk(ok_adk)
    # B2 fitted state
    st = {}
    for study in ('TB-M2', 'TB-S2'):
        rs = [r for r in ok_prop.values() if r['study'] == study and (study == 'TB-S2' or r['key']['kind'] == 'mesh')]
        pc_cv, pc_all, bc, ker, tnode, frozen = [], [], [], defaultdict(list), [], []
        n_inside = n_invalid = 0
        for r in rs:
            p = r['proposed']
            K = p['K_cv']
            if K > 0:
                pc_cv.append(p['point_clearance'][str(K)])
            pc_all += [v for k, v in p['point_clearance'].items() if k != '0' and v is not None]
            if p['balls']:
                bc += p['balls']['clearance']
                n_inside += any(c <= 0 for c in p['balls']['clearance'])
                for kb in p['kernel']:
                    tnode.append(kb['target_to_node_min'])
                    if kb.get('series_invalid'):
                        continue
                    for b in kb['bins']:
                        band = int(r['key']['band']) if study == 'TB-M2' else None
                        lab = f'{band}Hz' if band else 'sim'
                        ker[lab].append((b['frob'], b['diag_max'], b['kR'], kb['radius']))
                        ker['all'].append((b['frob'], b['diag_max'], b['kR'], kb['radius']))
                if p['frozen_exact'] is None:
                    n_invalid += 1
                else:
                    frozen.append((r, db(p['frozen_exact']) - db(p['j_cv_ext'])))
        kerd = {lab: dict(ball_bins=len(v), frob=span([a for a, _, _, _ in v]), diag_max=span([b for _, b, _, _ in v]),
                          kR=span([c for _, _, c, _ in v]), at_radius_floor=int(sum(d <= 0.00501 for _, _, _, d in v)),
                          frob_over_1pct=int(sum(a > 0.01 for a, _, _, _ in v)), frob_over_10pct=int(sum(a > 0.1 for a, _, _, _ in v)))
                for lab, v in ker.items()}
        fz = [d for _, d in frozen]
        st[study] = dict(problems=len(rs),
                         point_clearance_cv_order=dict(span=span(pc_cv), below_guard=int(sum(c < GUARD for c in pc_cv))),
                         point_clearance_every_order=dict(span=span(pc_all), below_guard=int(sum(c < GUARD for c in pc_all))),
                         ball_clearance=dict(span=span(bc), below_guard=int(sum(c < GUARD for c in bc)), balls=len(bc),
                                             problems_with_mic_inside_a_ball=int(n_inside)),
                         kernel=kerd, target_to_node_min=dict(span=span(tnode), below_1cm=int(sum(t < 0.01 for t in tnode))),
                         frozen_exact_swap=dict(problems=len(fz), skipped_series_invalid=n_invalid, change_db=span(fz),
                                                abs_over_0p1=int(sum(abs(x) > 0.1 for x in fz)), abs_over_1=int(sum(abs(x) > 1 for x in fz)),
                                                pooled=pooled_change(study, frozen)))
    out['B2_fitted_state'] = st
    (HERE / 'refit-summary.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:8000])


def pooled_change(study, frozen):
    """Mean of the per-problem change over a study's registered conditions, the units as registered (seeds; loudspeakers
    with draws averaged), with a 95 % Student-t interval over the units."""
    cell = defaultdict(lambda: defaultdict(list))
    for r, d in frozen:
        k = r['key']
        if study == 'TB-S2':
            if k['cell'] == AV.FAR:
                continue
            cell[k['cell']][k['seed']].append(d)
        else:
            cell[f"{int(k['band'])}Hz-M{k['M']}"][k['source']].append(d)
    per = defaultdict(list)
    for c, units in cell.items():
        for u, v in units.items():
            per[u].append(float(np.mean(v)))
    m, lo, hi = AV.tint([np.mean(v) for v in per.values()], AV.ALPHA)
    return dict(mean=m, low=lo, high=hi, units=len(per), cells=len(cell))


def rules_with_corrected_adk(ok):
    res = {}
    # simulation S1
    rows, _ = AV.load(S / 'studies/TB-S2/primary')
    for r in rows:
        r['cell_key'] = r['cell']
        k = ('sim', r['cell'], r['seed'])
        if k in ok:
            r['e']['adk_standin'] = db(ok[k]['adk']['corrected'])
            r['e']['best_generic_task'] = min(r['e'][m] for m in AV.GENERIC)
    cells = list(dict.fromkeys(r['cell'] for r in rows))
    prox = [c for c in cells if c != AV.FAR]
    missing = sum(('sim', r['cell'], r['seed']) not in ok for r in rows)
    T = AV.analyse(rows, 'sim', cells, len(prox))
    cl = [T[c]['comparisons'][f"{AV.PRIMARY}-vs-{T[c]['best_generic']}"]['cls'] for c in prox]
    res['S1_corrected'] = dict(wins=cl.count('win'), ties=cl.count('tie'), losses=cl.count('loss'), cells=len(prox), missing_problems=missing,
                               adk_best_generic_cells=int(sum(T[c]['best_generic'] == 'adk_standin' for c in prox)),
                               pooled_vs_best_generic_cell=pooled_bg(rows, 'sim', prox, {c: T[c]['best_generic'] for c in prox}),
                               pooled_vs_per_problem_oracle=AV.pooled(rows, 'sim', prox, AV.PRIMARY, 'best_generic_task'))
    res['S1_corrected']['every_member'] = every_member_count(rows, 'sim', prox, len(prox), AV.GENERIC)
    # measured M1, registered sparse Bayesian comparators
    mrows, _ = AV.load(S / 'studies/TB-M2/primary')
    mesh = [r for r in mrows if r['kind'] == 'mesh']
    missing = 0
    for r in mesh:
        r['cell_key'] = f"{int(r['band'])}Hz-M{r['M']}"
        k = ('mesh', r['source'], float(r['band']), r['M'], r['draw'])
        if k in ok:
            r['nmse']['adk_standin'] = ok[k]['adk']['corrected']
            r['e']['adk_standin'] = AV.db(ok[k]['adk']['corrected'])
            r['e']['best_generic_task'] = min(r['e'][m] for m in AV.GENERIC)
        else:
            missing += 1
    mcells = sorted({r['cell_key'] for r in mesh}, key=lambda c: (int(c.split('Hz')[0]), int(c.split('M')[1])))
    MT = AV.analyse(mesh, 'mesh', mcells, len(mcells))
    cl = [MT[c]['comparisons'][f"{AV.PRIMARY}-vs-{MT[c]['best_generic']}"]['cls'] for c in mcells]
    res['M1_corrected_adk'] = dict(wins=cl.count('win'), ties=cl.count('tie'), losses=cl.count('loss'), cells=len(mcells), missing_problems=missing,
                                   adk_best_generic_cells=int(sum(MT[c]['best_generic'] == 'adk_standin' for c in mcells)))
    # measured M1 with the target-guarded sparse Bayesian comparators (as TB-M2-SBLFIX) and the corrected stand-in
    fix = {AF.key(r): r for r in map(json.loads, open(S / 'studies/TB-M2-SBLFIX/primary/rows.jsonl')) if 'error' not in r}
    G = ('sinc_ev', 'rbf', 'adk_standin', 'sbl_em_guard', 'sbl_mackay_guard')
    cells = defaultdict(list)
    for r in mesh:
        e = {m: AF.err(v) for m, v in r['nmse'].items()}
        e.update({m: AF.err(v) for m, v in fix[AF.key(r)]['nmse'].items()})
        cells[r['cell_key']].append(dict(source=r['source'], e=e))
    best = {}
    for c, rs in cells.items():
        with np.errstate(all='ignore'):
            mean = {m: np.nanmean([t['e'][m] for t in rs]) for m in G}
        best[c] = min((m for m in G if np.isfinite(mean[m])), key=lambda m: mean[m])
        for t in rs:
            t['e']['best_generic_cell'] = t['e'][best[c]]
            t['e']['best_generic_task'] = np.nanmin([t['e'][m] for m in G])
    c1 = AF.compare(cells, AV.PRIMARY, 'best_generic_cell', len(cells))
    res['M1_corrected_adk_guarded_sbl'] = dict(wins=c1['wins'], ties=c1['ties'], losses=c1['losses'],
                                               adk_best_generic_cells=int(sum(v == 'adk_standin' for v in best.values())),
                                               pooled_vs_per_problem_oracle=AF.compare(cells, AV.PRIMARY, 'best_generic_task', len(cells))['pooled'])
    ev = {}
    for m in G:
        cm = AF.compare(cells, AV.PRIMARY, m, len(cells))
        ev[m] = {c: v['cls'] for c, v in cm['per_cell'].items()}
    res['M1_corrected_adk_guarded_sbl']['every_member_wins'] = int(sum(all(ev[m][c] == 'win' for m in G) for c in cells))
    # the official-code subset (TB-M2-RIB, R2): the stand-in at the scored bin, corrected
    reg = {AF.key(r): r for r in mesh}
    rib = defaultdict(list)
    for l in open(S / 'studies/TB-M2-RIB/primary/rows.jsonl'):
        r = json.loads(l)
        if 'error' in r:
            continue
        base = reg[AF.key(r)]; b = r['bin']
        e = {'adk_standin': AF.err(base['nmse']['adk_standin'], [b]), 'ribeiro_official': AF.err(r['nmse']['ribeiro_official']),
             AV.PRIMARY: AF.err(base['nmse'][AV.PRIMARY], [b])}
        rib[f"{int(r['band'])}Hz-M{r['M']}"].append(dict(source=r['source'], e=e))
    res['RIB_R2_corrected'] = {k: v for k, v in AF.compare(rib, 'adk_standin', 'ribeiro_official', len(rib)).items() if k != 'per_cell'}
    return res


def pooled_bg(rows, kind, cells, best):
    by = defaultdict(list)
    for r in rows:
        by[r['cell_key']].append(r)
    per = defaultdict(list)
    for c in cells:
        for u, v in AV.cell_diff(by[c], kind, AV.PRIMARY, best[c]).items():
            per[u].append(v)
    m, lo, hi = AV.tint([np.mean(v) for v in per.values()], AV.ALPHA)
    return dict(mean=m, low=lo, high=hi, units=len(per))


def every_member_count(rows, kind, cells, family_n, members):
    by = defaultdict(list)
    for r in rows:
        by[r['cell_key']].append(r)
    n = 0
    for c in cells:
        ok = True
        for m in members:
            d = AV.cell_diff(by[c], kind, AV.PRIMARY, m)
            mu, lo, hi = AV.tint(list(d.values()), AV.ALPHA / family_n)
            ok &= AV.classify(mu, lo, hi) == 'win'
        n += ok
    return n


def reproduction():
    rows = [json.loads(l) for l in open(HERE / 'reproduction' / 'rows.jsonl')]
    errs = [r for r in rows if 'error' in r]
    rows = [r for r in rows if 'error' not in r]
    bins = [r for r in rows if r['task'][0] == 'bin']
    bands = [r for r in rows if r['task'][0] == 'band']
    acc = defaultdict(lambda: np.zeros(2))
    per_trial = defaultdict(lambda: defaultdict(lambda: np.zeros(2)))
    for r in bins:
        s = r['task'][1]
        items = [('sinc', r['sinc']), ('proposed_bin', r['proposed_bin'])]
        for d, v in r['adaptation'].items():
            items += [(f'adaptation_opt_{d}', v['opt']), (f'adaptation_stage1_{d}', v['stage1'])]
        for name, (e, p) in items:
            acc[name] += (e, p); per_trial[name][s] += (e, p)
    for r in bands:
        s = r['task'][1]
        for e, p in r['proposed_shared']:
            acc['proposed_shared'] += (e, p); per_trial['proposed_shared'][s] += (e, p)
    nm = {k: float(10 * np.log10(v[0] / v[1])) for k, v in acc.items()}
    trials = sorted(per_trial['proposed_bin'])
    tdb = {k: {s: float(10 * np.log10(v[s][0] / v[s][1])) for s in trials if s in v} for k, v in per_trial.items()}

    def paired(a, b):
        d = [tdb[a][s] - tdb[b][s] for s in trials if s in tdb[a] and s in tdb[b]]
        m, lo, hi = AV.tint(d, AV.ALPHA)
        return dict(mean=m, low=lo, high=hi, trials=len(d))
    pub = {'0.15': -20.8, '0.2': -19.3, '0.25': -5.9}
    pub1 = {'0.15': -20.1, '0.2': -17.9, '0.25': -3.4}
    crit = dict(within_3dB_at_0p2=bool(abs(nm['adaptation_opt_0.2'] - pub['0.2']) <= 3.0),
                collapse_at_0p25_ge_6dB=bool(nm['adaptation_opt_0.25'] - nm['adaptation_opt_0.2'] >= 6.0))
    crit['reproduced'] = bool(crit['within_3dB_at_0p2'] and crit['collapse_at_0p25_ge_6dB'])
    by_f = defaultdict(lambda: defaultdict(lambda: np.zeros(2)))
    for r in bins:
        f = r['task'][2]
        by_f['proposed_bin'][f] += r['proposed_bin']; by_f['adaptation_opt_0.2'][f] += r['adaptation']['0.2']['opt']
        by_f['sinc'][f] += r['sinc']
    out = dict(status='post hoc', bins=len(bins), bands=len(bands), errors=len(errs), pooled_nmse_db=nm, published_fig6=dict(opt=pub, stage1=pub1),
               criterion=crit,
               paired_per_trial=dict(proposed_bin_minus_adaptation_opt_0p2=paired('proposed_bin', 'adaptation_opt_0.2'),
                                     proposed_shared_minus_adaptation_opt_0p2=paired('proposed_shared', 'adaptation_opt_0.2'),
                                     proposed_bin_minus_sinc=paired('proposed_bin', 'sinc')),
               dictionary_sizes={d: sorted({r['adaptation'][d]['S'] for r in bins}) for d in ('0.15', '0.2', '0.25')},
               stage1_cap_hits={d: int(sum(r['adaptation'][d]['iterations'][0] >= 2000 for r in bins)) for d in ('0.15', '0.2', '0.25')},
               regions_median={d: float(np.median([r['adaptation'][d]['regions'] for r in bins])) for d in ('0.15', '0.2', '0.25')},
               proposed_bin_K=dict(zip(*np.unique([r['proposed_bin_K'] for r in bins], return_counts=True))),
               proposed_shared_K=[r['proposed_shared_K'] for r in bands],
               by_frequency_db={k: {str(f): float(10 * np.log10(v[0] / v[1])) for f, v in sorted(d.items())} for k, d in by_f.items()})
    out['proposed_bin_K'] = {str(k): int(v) for k, v in out['proposed_bin_K'].items()}
    (HERE / 'reproduction-summary.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:6000])


def gauss():
    """PLAN-D: the Gaussian-density extension against the registered extension, the best generic estimator and points."""
    new = [json.loads(l) for l in open(HERE / 'gauss' / 'rows.jsonl')]
    errs = [r for r in new if 'error' in r]
    new = {rkey(r['study'], r): r for r in new if 'error' not in r}
    bad = [k for k, r in new.items() if max(r['gate_rel'].values()) > GATE]
    out = dict(status='post hoc', rows=len(new), errors=len(errs), gate_failed=len(bad),
               gate_max=max(max(r['gate_rel'].values()) for r in new.values()))
    for study, kind, fam_all in (('TB-M2', 'mesh', None), ('TB-S2', 'sim', None)):
        rows, _ = AV.load(S / 'studies' / study / 'primary')
        rows = [r for r in rows if r.get('kind', 'sim') == kind or study == 'TB-S2']
        for r in rows:
            r['cell_key'] = r['cell'] if study == 'TB-S2' else f"{int(r['band'])}Hz-M{r['M']}"
        keep = []
        for r in rows:
            k = ('sim', r['cell'], r['seed']) if study == 'TB-S2' else ('mesh', r['source'], float(r['band']), r['M'], r['draw'])
            if k in new and k not in set(bad):
                r['e']['j_cv_ext_gauss'] = db(new[k]['j_cv_ext_gauss']); keep.append(r)
        cells = list(dict.fromkeys(r['cell_key'] for r in keep))
        if study == 'TB-S2':
            cells = [c for c in cells if c != AV.FAR]
        fam = len(cells)
        T = AV.analyse([r for r in keep if r['cell_key'] in cells], kind, cells, fam)
        by = defaultdict(list)
        for r in keep:
            by[r['cell_key']].append(r)
        def cls(a, b, best=None):
            res = dict(win=0, tie=0, loss=0)
            for c in cells:
                bb = T[c]['best_generic'] if best else b
                d = AV.cell_diff(by[c], kind, a, bb)
                m, lo, hi = AV.tint(list(d.values()), AV.ALPHA / fam)
                res[AV.classify(m, lo, hi)] += 1
            return res
        out[study] = dict(problems=len(keep), cells=len(cells),
                          gauss_minus_registered=dict(classes=cls('j_cv_ext_gauss', AV.PRIMARY),
                                                      pooled=AV.pooled(keep, kind, cells, 'j_cv_ext_gauss', AV.PRIMARY)),
                          gauss_vs_best_generic=dict(classes=cls('j_cv_ext_gauss', None, best=True),
                                                     pooled_vs_per_problem_oracle=AV.pooled(keep, kind, cells, 'j_cv_ext_gauss', 'best_generic_task')),
                          gauss_minus_points=AV.pooled(keep, kind, cells, 'j_cv_ext_gauss', 'j_cv'),
                          registered_minus_points=AV.pooled(keep, kind, cells, AV.PRIMARY, 'j_cv'),
                          radii=span([x for r in new.values() if r['study'] == study for x in r['radii']]))
    (HERE / 'gauss-summary.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:5000])


def gsmf_repro():
    """PLAN-E: the group-sparse adaptation in its published configuration, with the paper's zeta choice (3 kHz)."""
    import reproduce_gsmf as RG
    rows = [json.loads(l) for l in open(HERE / 'reproduction-gsmf' / 'rows.jsonl')]
    errs = [r for r in rows if 'error' in r]; rows = [r for r in rows if 'error' not in r]
    F = len(RG.FREQS); i3k = int(np.argmin(np.abs(RG.FREQS - 3000.0)))
    def trial_db(pairs):                         # per-frequency dB of one trial
        return [10 * np.log10(e / p) for e, p in pairs]
    per = defaultdict(dict)                      # method -> seed -> per-frequency dB (F,)
    for r in rows:
        kind, seed, arg = r['task']
        if kind == 'adapt':
            for z, v in r['zeta'].items():
                per[f'adapt_{arg}_{z}'][seed] = trial_db(v['sq'])
        elif kind == 'band':
            per['proposed_shared'][seed] = trial_db(r['proposed_shared'])
    for r in rows:
        kind, seed, arg = r['task']
        if kind == 'bin':
            for m in ('proposed_bin', 'sinc'):
                per[m].setdefault(seed, [np.nan] * F)[int(arg)] = 10 * np.log10(r[m][0] / r[m][1])
    curve = {m: np.nanmean(np.array(list(v.values())), axis=0) for m, v in per.items()}
    def bands(c):
        c = np.asarray(c)
        return dict(all=float(np.mean(c)), le_1k=float(np.mean(c[:8])), mid=float(np.mean(c[8:20])), high=float(np.mean(c[20:])))
    out = dict(status='post hoc', rows=len(rows), errors=len(errs), trials={m: len(v) for m, v in per.items()})
    for d in ('0.1', '0.05'):
        zs = [k for k in curve if k.startswith(f'adapt_{d}_')]
        chosen = min(zs, key=lambda k: curve[k][i3k])
        out[f'delta_{d}'] = dict(chosen_zeta=chosen.split('_')[-1], chosen=bands(curve[chosen]),
                                 registered_zeta_0p1=bands(curve[f'adapt_{d}_0.1']),
                                 per_zeta={k.split('_')[-1]: bands(curve[k]) for k in zs},
                                 published=bands(RG.PUBLISHED[d]), regions_note='see rows')
    c = out['delta_0.1']
    out['criterion'] = dict(all_within_3dB=bool(abs(c['chosen']['all'] - c['published']['all']) <= 3.0),
                            le_1k_within_3dB=bool(abs(c['chosen']['le_1k'] - c['published']['le_1k']) <= 3.0))
    out['criterion']['reproduced'] = bool(out['criterion']['all_within_3dB'] and out['criterion']['le_1k_within_3dB'])
    for m in ('proposed_bin', 'proposed_shared', 'sinc'):
        out[m] = bands(curve[m])
    chosen01 = f"adapt_0.1_{out['delta_0.1']['chosen_zeta']}"
    def paired(a, b):
        seeds = sorted(set(per[a]) & set(per[b]))
        dd = [float(np.mean(per[a][s_]) - np.mean(per[b][s_])) for s_ in seeds]
        m_, lo, hi = AV.tint(dd, AV.ALPHA)
        return dict(mean=m_, low=lo, high=hi, trials=len(dd))
    out['paired'] = dict(proposed_bin_minus_adaptation=paired('proposed_bin', chosen01),
                         proposed_shared_minus_adaptation=paired('proposed_shared', chosen01))
    out['curves'] = {m: [float(x) for x in v] for m, v in curve.items() if m in ('proposed_bin', 'proposed_shared', 'sinc', chosen01)}
    out['dictionary'] = sorted({(r['task'][2], r['S']) for r in rows if r['task'][0] == 'adapt'})
    (HERE / 'reproduction-gsmf-summary.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != 'curves'}, indent=1, default=float)[:5000])


def cap8():
    """PLAN-F: the cap raised to eight, against the registered estimator and the best generic estimator."""
    new = [json.loads(l) for l in open(HERE / 'cap8' / 'rows.jsonl')]
    errs = [r for r in new if 'error' in r]
    new = {rkey(r['study'], r): r for r in new if 'error' not in r}
    bad = [k for k, r in new.items() if max(r['gate_rel'].values()) > GATE or not r['K5_cv_check']]
    out = dict(status='post hoc', rows=len(new), errors=len(errs), gate_failed=len(bad))
    for study, kind in (('TB-M2', 'mesh'), ('TB-S2', 'sim')):
        rows, _ = AV.load(S / 'studies' / study / 'primary')
        rows = [r for r in rows if study == 'TB-S2' or r['kind'] == 'mesh']
        keep = []
        for r in rows:
            r['cell_key'] = r['cell'] if study == 'TB-S2' else f"{int(r['band'])}Hz-M{r['M']}"
            k = ('sim', r['cell'], r['seed']) if study == 'TB-S2' else ('mesh', r['source'], float(r['band']), r['M'], r['draw'])
            if k in new and k not in set(bad):
                r['e']['j_cv8_ext'] = db(new[k]['j_cv8_ext']); r['K8'] = new[k]['K8']; keep.append(r)
        cells = list(dict.fromkeys(r['cell_key'] for r in keep))
        if study == 'TB-S2':
            cells = [c for c in cells if c != AV.FAR]
        fam = len(cells)
        sub = [r for r in keep if r['cell_key'] in cells]
        T = AV.analyse(sub, kind, cells, fam)
        by = defaultdict(list)
        for r in sub:
            by[r['cell_key']].append(r)
        def cls(a, b=None):
            res = dict(win=0, tie=0, loss=0)
            for c in cells:
                d = AV.cell_diff(by[c], kind, a, b or T[c]['best_generic'])
                m, lo, hi = AV.tint(list(d.values()), AV.ALPHA / fam)
                res[AV.classify(m, lo, hi)] += 1
            return res
        K8 = [r['K8'] for r in sub]; K5 = [r['info']['j_K']['cv'] for r in sub]
        out[study] = dict(problems=len(sub), cells=len(cells),
                          K8_hist=np.bincount(K8, minlength=9).tolist(), share_at_8=float(np.mean([k == 8 for k in K8])),
                          share_changed=float(np.mean([a != b for a, b in zip(K8, K5)])),
                          cap8_minus_registered=dict(classes=cls('j_cv8_ext', AV.PRIMARY), pooled=AV.pooled(sub, kind, cells, 'j_cv8_ext', AV.PRIMARY)),
                          cap8_vs_best_generic=dict(classes=cls('j_cv8_ext'),
                                                    pooled_vs_per_problem_oracle=AV.pooled(sub, kind, cells, 'j_cv8_ext', 'best_generic_task')))
    (HERE / 'cap8-summary.json').write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float)[:4000])


if __name__ == '__main__':
    {'refit': refit, 'reproduction': reproduction, 'gauss': gauss, 'gsmf': gsmf_repro, 'cap8': cap8}[sys.argv[1]]()
