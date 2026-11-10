"""WP7 post hoc statistics re-analysis (not registered; no model fits). Reads the registered rows read-only.

    python3 -I -B posthoc/upgrade-wp7/wp7.py      # writes posthoc/upgrade-wp7/wp7-results.json

Metric, units and intervals follow analyze_v2.py / analyze_followups.py / analyze_m3.py:
  per-problem error = 10 log10(mean NMSE over the 6 bins of the band); paired differences in dB (negative favours the
  proposed estimator j_cv_ext); units = simulation seed (TB-S2), loudspeaker with its 2 draws averaged (room grid),
  array microphone with its 2 draws averaged (MIRACLE); per-condition Student-t intervals with Bonferroni over the
  conditions of the rule; pooled = per unit, mean over conditions, then a 95 % Student-t interval over units;
  margin 0.1 dB; best generic = generic estimator with the lowest observed mean error in the condition.
Room grid = TB-M2 mesh rows with the sparse Bayesian comparators replaced by the target-guarded ones of TB-M2-SBLFIX.
Step 1 is a reproduction gate: the script stops (exit 2) if any published number is not reproduced.
"""
from __future__ import annotations
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
DELTA, ALPHA, P = 0.1, 0.05, 'j_cv_ext'
SEED, NBOOT, NSPLIT, NBOOT_LIN = 20261109, 10000, 1000, 2000
FAR = 'A1-beside1.2'
INTERIOR = ('A1-in', 'A5-delta0.4', 'FF-02', 'FF-03', 'FF-06', 'FF-07')   # posthoc/review3/saved.json A1
G_REG = ('sinc_ev', 'rbf', 'adk_standin', 'sbl_em_common', 'sbl_mackay_common')
G_ROOM = ('sinc_ev', 'rbf', 'adk_standin', 'sbl_em_guard', 'sbl_mackay_guard')
OTHER = ('two_stage_point', 'two_stage_region', 'gsmf', 'mo26_cap5')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows_of(study):
    d = S / 'studies' / study / 'primary'
    assert sha(d / 'rows.jsonl') == json.loads((d / 'run-receipt.json').read_text())['outputs']['rows.jsonl'], study
    rs = [json.loads(l) for l in open(d / 'rows.jsonl')]
    assert not any('error' in r for r in rs), study
    return rs


def lin(v):
    v = np.asarray(v, float)
    return float(np.mean(v)) if np.all(np.isfinite(v)) and np.all(v < 1e6) else np.nan


class Data:
    """cells -> method -> arrays (U, R) of dB error and linear mean NMSE; units sorted, draws sorted."""

    def __init__(self, name, problems, cells, generic, methods):
        self.name, self.cells, self.generic = name, cells, generic
        self.units = sorted({p['unit'] for p in problems}); self.draws = sorted({p['draw'] for p in problems})
        ui = {u: i for i, u in enumerate(self.units)}; di = {d: i for i, d in enumerate(self.draws)}
        U, R = len(self.units), len(self.draws)
        self.L = {c: {m: np.full((U, R), np.nan) for m in methods} for c in cells}
        for p in problems:
            if p['cell'] in self.L:
                for m in methods:
                    self.L[p['cell']][m][ui[p['unit']], di[p['draw']]] = p['lin'][m]
        self.nonfinite = {m: int(sum(np.sum(~np.isfinite(self.L[c][m])) for c in cells)) for m in methods}
        for c in cells:   # only a named non-generic comparator may be undefined; it is then excluded pairwise
            for m in (P,) + tuple(generic):
                assert np.all(np.isfinite(self.L[c][m])), (name, c, m)
        self.E = {c: {m: 10 * np.log10(v) for m, v in self.L[c].items()} for c in cells}
        for c in cells:   # per-problem oracle over the generic set (min in dB = min in linear)
            self.E[c]['oracle'] = np.min([self.E[c][m] for m in generic], axis=0)
            self.L[c]['oracle'] = np.min([self.L[c][m] for m in generic], axis=0)
        self.best = {c: min(generic, key=lambda m: self.E[c][m].mean()) for c in cells}

    def udiff(self, c, b, a=P):
        """per-unit paired difference a - b, draws averaged (U,)"""
        b = self.best[c] if b == 'best_generic' else b
        return np.nanmean(self.E[c][a] - self.E[c][b], axis=1)

    def pooled_units(self, b, cells=None):
        cells = cells or self.cells
        return np.mean([self.udiff(c, b) for c in cells], axis=0)


def tint(x, alpha):
    x = np.asarray(x, float); n = len(x); m = float(x.mean())
    h = float(stats.t.ppf(1 - alpha / 2, n - 1) * x.std(ddof=1) / np.sqrt(n))
    return m, m - h, m + h


def classify(m, lo, hi):
    return 'win' if (hi < 0 and m < -DELTA) else ('loss' if (lo > 0 and m > DELTA) else 'tie')


def pooled(D, b, cells=None):
    m, lo, hi = tint(D.pooled_units(b, cells), ALPHA)
    return dict(mean=m, low=lo, high=hi, units=len(D.units), conditions=len(cells or D.cells))


def wtl(cls):
    return dict(win=cls.count('win'), tie=cls.count('tie'), loss=cls.count('loss'))


def registered_classes(D, b):
    out = []
    for c in D.cells:
        out.append(classify(*tint(D.udiff(c, b), ALPHA / len(D.cells))))
    return out


# ---------------------------------------------------------------- data
def load_all():
    s2 = rows_of('TB-S2')
    meth = (P,) + G_REG + ('gsmf', 'mo26_cap5')
    probs = [dict(cell=r['cell'], unit=r['seed'], draw=0, lin={m: lin(r['nmse'][m]) for m in meth}) for r in s2]
    allc = list(dict.fromkeys(r['cell'] for r in s2))
    prox = [c for c in allc if c != FAR]
    assert len(prox) == 63 and all(c in prox for c in INTERIOR)
    sim = Data('simulation', probs, prox, G_REG, meth)

    m2 = {(r['source'], r['band'], r['M'], r['draw']): r for r in rows_of('TB-M2') if r['kind'] == 'mesh'}
    fix = rows_of('TB-M2-SBLFIX')
    assert len(fix) == len(m2) == 1024
    meth = (P,) + G_ROOM + OTHER
    probs = []
    for r in fix:
        b = m2[(r['source'], r['band'], r['M'], r['draw'])]
        nm = dict(b['nmse']); nm.update(r['nmse'])
        probs.append(dict(cell=f"{int(r['band'])}Hz-M{r['M']}", unit=r['source'], draw=r['draw'], lin={m: lin(nm[m]) for m in meth}))
    cells = [f'{f}Hz-M{M}' for f in (125, 250, 500, 1000) for M in (24, 48, 96, 192)]
    room = Data('room_grid', probs, cells, G_ROOM, meth)

    m3 = rows_of('TB-M3-MIRACLE')
    meth = (P,) + G_REG + OTHER
    mir = {}
    for sc in ('A1', 'R2'):
        probs = [dict(cell=f"{int(r['band'])}Hz-M{r['M']}", unit=r['source'], draw=r['draw'], lin={m: lin(r['nmse'][m]) for m in meth})
                 for r in m3 if r['scenario'] == sc]
        cells = [f'{f}Hz-M{M}' for f in (250, 500, 1000) for M in (24, 48, 96, 192)]
        mir[sc] = Data(f'miracle_{sc}', probs, cells, G_REG, meth)
    return sim, room, mir


# ---------------------------------------------------------------- step 1: gate
def gate(sim, room, mir):
    def r2(x):
        return [round(x['mean'], 2), round(x['low'], 2), round(x['high'], 2)]
    chk = []

    def add(name, got, want):
        chk.append(dict(item=name, reproduced=got, published=want, ok=bool(got == want)))
    add('room grid pooled vs best generic', r2(pooled(room, 'best_generic')), [-0.72, -0.80, -0.64])
    add('room grid pooled vs per-problem oracle', r2(pooled(room, 'oracle')), [-0.60, -0.68, -0.52])
    c = wtl(registered_classes(room, 'best_generic')); add('room grid W/T/L vs best generic', [c['win'], c['tie'], c['loss']], [13, 3, 0])
    add('simulation pooled vs best generic (63)', r2(pooled(sim, 'best_generic')), [-5.03, -5.17, -4.88])
    for sc, w, p in (('A1', [8, 3, 1], [-2.25, -2.49, -2.02]), ('R2', [4, 4, 4], [-0.21, -0.34, -0.07])):
        c = wtl(registered_classes(mir[sc], 'best_generic'))
        add(f'MIRACLE {sc} W/T/L vs best generic', [c['win'], c['tie'], c['loss']], w)
        add(f'MIRACLE {sc} pooled vs best generic', r2(pooled(mir[sc], 'best_generic')), p)
    # exact agreement with the registered summaries (full precision)
    reg = json.loads((S / 'studies/TB-M2-SBLFIX/primary/summary.json').read_text())
    exact = dict(room_best=abs(pooled(room, 'best_generic')['mean'] - reg['M1_corrected']['pooled']['mean']),
                 room_oracle=abs(pooled(room, 'oracle')['mean'] - reg['pooled_vs_task_best_generic']['mean']),
                 room_best_selection_same=bool(room.best == reg['M1_corrected']['best_generic']))
    reg3 = json.loads((S / 'studies/TB-M3-MIRACLE/primary/summary.json').read_text())
    for sc in ('A1', 'R2'):
        exact[f'miracle_{sc}'] = abs(pooled(mir[sc], 'best_generic')['mean'] - reg3[sc]['pooled_vs_best_generic_cell']['mean'])
        exact[f'miracle_{sc}_selection_same'] = bool(mir[sc].best == reg3[sc]['best_generic_by_cell'])
    regs = json.loads((S / 'studies/TB-S2/primary/summary.json').read_text())
    exact['sim_selection_same'] = bool(all(sim.best[c] == regs['cells'][c]['best_generic'] for c in sim.cells))
    mg = json.loads((S / 'posthoc/margins/margins.json').read_text())['simulated_margin']['all_63']
    exact['sim_best'] = abs(pooled(sim, 'best_generic')['mean'] - mg['mean'])
    tol_ok = all(v < 1e-9 for k, v in exact.items() if not k.endswith('_same')) and all(v for k, v in exact.items() if k.endswith('_same'))
    return dict(checks=chk, exact_vs_registered_summaries=exact, passed=bool(all(x['ok'] for x in chk) and tol_ok))


# ---------------------------------------------------------------- (a) per-comparator inference with Holm
def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.empty(n); run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (n - i) * p[j])); adj[j] = run
    return adj


def per_comparator(datasets, comparators, label):
    """datasets: list of Data sharing one Holm family (all comparator x condition tests)."""
    tests = []
    for D in datasets:
        for b in comparators:
            for c in D.cells:
                d = D.udiff(c, b); n = len(d); m = float(d.mean()); se = float(d.std(ddof=1) / np.sqrt(n)); df = n - 1
                p0 = float(2 * stats.t.sf(abs(m) / se, df))
                pw = float(min(1.0, 2 * stats.t.cdf((m + DELTA) / se, df)))   # CI-equivalent: upper bound < -0.1
                pl = float(min(1.0, 2 * stats.t.sf((m - DELTA) / se, df)))    # lower bound > +0.1
                m_, lo, hi = tint(d, ALPHA / len(D.cells))
                tests.append(dict(data=D.name, comparator=b, cond=c, mean=m, se=se, p0=p0, p_pr=min(pw, pl),
                                  dir_pr='win' if pw <= pl else 'loss', reg_cls=classify(m_, lo, hi), bonf_lo=lo, bonf_hi=hi))
    N = len(tests)
    a0 = holm([t['p0'] for t in tests]); ap = holm([t['p_pr'] for t in tests])
    for t, x, y in zip(tests, a0, ap):
        t['p0_holm'], t['p_pr_holm'] = float(x), float(y)
        rej = x < ALPHA
        t['holm_current'] = 'win' if (rej and t['mean'] < -DELTA) else ('loss' if (rej and t['mean'] > DELTA) else 'tie')
        t['holm_practical'] = t['dir_pr'] if y < ALPHA else 'tie'
        t['bonf_family_practical'] = t['dir_pr'] if t['p_pr'] < ALPHA / N else 'tie'
    summ = {}
    for D in datasets:
        for b in comparators:
            ts = [t for t in tests if t['data'] == D.name and t['comparator'] == b]
            summ[f'{D.name}:{b}'] = dict(
                registered_style_bonferroni_over_conditions=wtl([t['reg_cls'] for t in ts]),
                holm_current=wtl([t['holm_current'] for t in ts]),
                holm_practical=wtl([t['holm_practical'] for t in ts]),
                pooled_db=pooled(D, b))
    return dict(family=label, n_tests=N, alpha_familywise=ALPHA, summary=summ,
                tests=[{k: (round(v, 6) if isinstance(v, float) else v) for k, v in t.items()} for t in tests])


# ---------------------------------------------------------------- (b) linear scale
def pct(d):
    return 100 * (10 ** (d / 10) - 1)


def linear(D, comparators, rng):
    out = {}
    U = len(D.units)
    idx = rng.integers(0, U, size=(NBOOT_LIN, U))
    for b in comparators:
        per = {}
        num, den = [], []
        for c in D.cells:
            bb = D.best[c] if b == 'best_generic' else b
            lc = D.L[c][bb]; lp = np.where(np.isfinite(lc), D.L[c][P], np.nan)   # (U, R), undefined problems dropped pairwise
            dd = float(D.udiff(c, b).mean())
            per[c] = dict(comparator=bb, mean_db=dd, geometric_ratio=10 ** (dd / 10),
                          ratio_of_means=float(np.nanmean(lp) / np.nanmean(lc)), mean_of_ratios=float(np.nanmean(lp / lc)))
            num.append(np.nanmean(lp, axis=1)); den.append(np.nanmean(lc, axis=1))   # per unit, draws averaged (linear)
        num, den = np.array(num), np.array(den)                              # (C, U)
        rom = float(num.sum() / den.sum())
        boot = num[:, idx].sum(axis=(0, 2)) / den[:, idx].sum(axis=(0, 2))
        pl = pooled(D, b)
        out[b] = dict(pooled_db=pl,
                      pooled_pct_change_geometric=dict(mean=pct(pl['mean']), low=pct(pl['low']), high=pct(pl['high'])),
                      pooled_ratio_of_means=dict(value=rom, boot95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]),
                      median_condition_ratio_of_means=float(np.median([v['ratio_of_means'] for v in per.values()])),
                      conditions_ratio_of_means_above_1=int(sum(v['ratio_of_means'] > 1 for v in per.values())),
                      per_condition=per)
    return out


# ---------------------------------------------------------------- (c) cluster bootstrap
def groups_room():
    z = np.load(S / 'B3-measured/meshrir-prep/meshrir-prep.npz')
    ps = z['pos_src']
    lab = {}
    for i, (x, y, _) in enumerate(ps):
        lab[i] = ('x+' if x > 0 else 'x-') if abs(x) >= abs(y) else ('y+' if y > 0 else 'y-')
    return lab, ps.tolist()


def groups_miracle():
    z = np.load(S / 'studies/TB-M3-MIRACLE/extract/miracle-A1.npz')
    z2 = np.load(S / 'studies/TB-M3-MIRACLE/extract/miracle-R2.npz')
    assert np.allclose(z['pos_src'], z2['pos_src'])
    ps = z['pos_src']; c = 0.5 * (ps.min(0) + ps.max(0))
    lab = {i: ('+' if x >= c[0] else '-') + ('+' if y >= c[1] else '-') for i, (x, y, _) in enumerate(ps)}
    return lab, c.tolist()


def cluster_boot(D, lab, rng):
    U = len(D.units)
    g = np.array([lab[u] for u in D.units]); names = sorted(set(g))
    members = [np.where(g == k)[0] for k in names]
    v = D.pooled_units('best_generic')
    # generic means per unit for the reselection variant: (C, nG, U)
    Gm = np.array([[D.E[c][m].mean(axis=1) for m in D.generic] for c in D.cells])
    Pm = np.array([D.E[c][P].mean(axis=1) for c in D.cells])
    stat_fixed, stat_resel, stat_iid = np.empty(NBOOT), np.empty(NBOOT), np.empty(NBOOT)
    for b in range(NBOOT):
        ks = rng.integers(0, len(names), len(names))
        idx = np.concatenate([members[k][rng.integers(0, len(members[k]), len(members[k]))] for k in ks])
        stat_fixed[b] = v[idx].mean()
        sel = Gm[:, :, idx].mean(axis=2).argmin(axis=1)                      # best generic re-chosen on the resample
        stat_resel[b] = np.mean([(Pm[ci, idx] - Gm[ci, sel[ci], idx]).mean() for ci in range(len(D.cells))])
        stat_iid[b] = v[rng.integers(0, U, U)].mean()
    m, lo, hi = tint(v, ALPHA)
    gm = np.array([v[mm].mean() for mm in members])
    gt = tint(gm, ALPHA)
    q = lambda s: [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]
    return dict(groups={k: int(len(mm)) for k, mm in zip(names, members)},
                group_means_db={k: float(x) for k, x in zip(names, gm)},
                student_t=dict(mean=m, low=lo, high=hi, width=hi - lo),
                cluster_bootstrap_fixed_selection=dict(ci=q(stat_fixed), width=float(np.diff(q(stat_fixed))[0]), sd=float(stat_fixed.std())),
                cluster_bootstrap_reselected=dict(mean=float(stat_resel.mean()), ci=q(stat_resel), width=float(np.diff(q(stat_resel))[0])),
                iid_unit_bootstrap=dict(ci=q(stat_iid), width=float(np.diff(q(stat_iid))[0])),
                cluster_level_t_df=dict(mean=gt[0], low=gt[1], high=gt[2], df=len(names) - 1),
                width_ratio_cluster_over_t=float(np.diff(q(stat_fixed))[0] / (hi - lo)))


# ---------------------------------------------------------------- (e) selection instability
def selection(D, rng):
    U, R = len(D.units), len(D.draws)
    out = dict(conditions=len(D.cells))
    insample = pooled(D, 'best_generic')
    # gap between best and runner-up generic, relative to the SE of their paired difference
    gaps = {}
    for c in D.cells:
        mu = sorted(D.generic, key=lambda m: D.E[c][m].mean())
        d = (D.E[c][mu[1]] - D.E[c][mu[0]]).mean(axis=1)
        gaps[c] = dict(best=mu[0], runner_up=mu[1], gap_db=float(d.mean()), gap_over_se=float(d.mean() / (d.std(ddof=1) / np.sqrt(U))))
    out['best_vs_runner_up'] = gaps
    out['conditions_gap_below_2se'] = int(sum(g['gap_over_se'] < 2 for g in gaps.values()))
    if R == 2:
        s0 = {c: min(D.generic, key=lambda m: D.E[c][m][:, 0].mean()) for c in D.cells}
        s1 = {c: min(D.generic, key=lambda m: D.E[c][m][:, 1].mean()) for c in D.cells}
        cf = np.mean([0.5 * ((D.E[c][P][:, 1] - D.E[c][s0[c]][:, 1]) + (D.E[c][P][:, 0] - D.E[c][s1[c]][:, 0])) for c in D.cells], axis=0)
        m, lo, hi = tint(cf, ALPHA)
        out['by_draw'] = dict(differs_between_draws=int(sum(s0[c] != s1[c] for c in D.cells)),
                              draw0_differs_from_full=int(sum(s0[c] != D.best[c] for c in D.cells)),
                              draw1_differs_from_full=int(sum(s1[c] != D.best[c] for c in D.cells)),
                              selected={c: [s0[c], s1[c]] for c in D.cells},
                              cross_fitted_pooled=dict(mean=m, low=lo, high=hi), in_sample_pooled=insample,
                              optimism_db=float(insample['mean'] - m))
    Um = np.array([D.E[c][P].mean(axis=1) for c in D.cells])                       # (C, U)
    Gm = np.array([[D.E[c][g].mean(axis=1) for g in D.generic] for c in D.cells])  # (C, G, U)
    full = Gm.mean(axis=2).argmin(axis=1)
    differ, cfm, differ_full = [], [], []
    for _ in range(NSPLIT):
        perm = rng.permutation(U); A, B = perm[:U // 2], perm[U // 2:]
        sa = Gm[:, :, A].mean(axis=2).argmin(axis=1); sb = Gm[:, :, B].mean(axis=2).argmin(axis=1)
        differ.append(float(np.mean(sa != sb))); differ_full.append(float(np.mean(np.r_[sa != full, sb != full])))
        ci = np.arange(len(D.cells))
        dB = Um[:, B] - Gm[ci, sa, :][:, B]                                     # half B judged with half A's choice
        dA = Um[:, A] - Gm[ci, sb, :][:, A]
        cfm.append(float(np.concatenate([dA, dB], axis=1).mean()))
    out['split_half'] = dict(splits=NSPLIT, mean_fraction_conditions_differing=float(np.mean(differ)),
                             fraction_splits_with_any_difference=float(np.mean(np.array(differ) > 0)),
                             mean_fraction_half_differs_from_full=float(np.mean(differ_full)),
                             cross_fitted_pooled_mean=dict(mean=float(np.mean(cfm)), p2_5=float(np.percentile(cfm, 2.5)), p97_5=float(np.percentile(cfm, 97.5))),
                             in_sample_pooled_mean=insample['mean'], optimism_db=float(insample['mean'] - np.mean(cfm)))
    return out


# ---------------------------------------------------------------- main
def main():
    sim, room, mir = load_all()
    res = dict(status='post hoc, not registered; no model fits; registered rows read-only',
               provenance=dict(seed=SEED, n_boot=NBOOT, n_split=NSPLIT, n_boot_linear=NBOOT_LIN,
                               room_generic=G_ROOM, generic=G_REG, primary=P, delta_db=DELTA, alpha=ALPHA))
    res['undefined_problems'] = {D.name: {m: k for m, k in D.nonfinite.items() if k} for D in (sim, room, mir['A1'], mir['R2'])}
    res['check_room_vs_cap5_without_undefined'] = dict(reproduced=pooled(room, 'mo26_cap5'), published='-0.81 [-0.90, -0.72] (studies/TB-M2/RESULTS.md)')
    g = gate(sim, room, mir); res['step1_gate'] = g
    print('GATE', 'PASS' if g['passed'] else 'FAIL')
    for x in g['checks']:
        print(' ', 'ok ' if x['ok'] else 'BAD', x['item'], x['reproduced'], 'published', x['published'])
    print('  exact', g['exact_vs_registered_summaries'])
    if not g['passed']:
        (HERE / 'wp7-results.json').write_text(json.dumps(res, indent=1, default=float))
        sys.exit(2)

    rng = np.random.default_rng(SEED)
    comp_sim = G_REG + ('gsmf', 'mo26_cap5')
    comp_room = G_ROOM + OTHER
    comp_mir = G_REG + OTHER
    res['a_per_comparator'] = dict(
        simulation=per_comparator([sim], comp_sim, 'TB-S2, 63 source-proximal conditions x 7 comparators'),
        room_grid=per_comparator([room], comp_room, 'room grid (TB-M2 + SBLFIX), 16 conditions x 9 comparators'),
        miracle=per_comparator([mir['A1'], mir['R2']], comp_mir, 'TB-M3-MIRACLE, both scenarios, 24 conditions x 9 comparators'),
        miracle_per_scenario_family={sc: per_comparator([mir[sc]], comp_mir, f'MIRACLE {sc} only, 12 x 9')['summary'] for sc in ('A1', 'R2')})
    bgp = {}
    for D in (sim, room, mir['A1'], mir['R2']):     # best generic of each condition, both definitions, Bonferroni over conditions
        cur, pr = [], []
        for c in D.cells:
            m, lo, hi = tint(D.udiff(c, 'best_generic'), ALPHA / len(D.cells))
            cur.append(classify(m, lo, hi)); pr.append('win' if hi < -DELTA else ('loss' if lo > DELTA else 'tie'))
        bgp[D.name] = dict(current=wtl(cur), practical=wtl(pr))
    res['a_best_generic_two_definitions'] = bgp
    res['b_linear'] = dict(simulation=linear(sim, ('best_generic', 'oracle') + comp_sim, rng),
                           room_grid=linear(room, ('best_generic', 'oracle') + comp_room, rng),
                           miracle_A1=linear(mir['A1'], ('best_generic', 'oracle') + comp_mir, rng),
                           miracle_R2=linear(mir['R2'], ('best_generic', 'oracle') + comp_mir, rng))
    lr, pos = groups_room(); lm, ctr = groups_miracle()
    res['c_cluster_bootstrap'] = dict(
        room_grid=dict(grouping='side of the square of loudspeakers (|x| or |y| = 1 m in the MeshRIR prep pos_src)',
                       labels={int(k): v for k, v in lr.items()}, **cluster_boot(room, lr, rng)),
        miracle_A1=dict(grouping=f'array quadrant about the bounding-box centre {np.round(ctr, 4).tolist()} of the 64 array microphones',
                        labels={int(u): lm[u] for u in mir['A1'].units}, **cluster_boot(mir['A1'], lm, rng)),
        miracle_R2=dict(grouping='as A1', **cluster_boot(mir['R2'], lm, rng)))
    no_int = [c for c in sim.cells if c not in INTERIOR]
    res['d_simulation_interior'] = dict(all_63=pooled(sim, 'best_generic'), no_interior_driver_57=pooled(sim, 'best_generic', no_int),
                                        interior_driver_6=pooled(sim, 'best_generic', list(INTERIOR)),
                                        interior_without_A1_in_5=pooled(sim, 'best_generic', [c for c in INTERIOR if c != 'A1-in']),
                                        reproduces_review3=bool(round(pooled(sim, 'best_generic', no_int)['mean'], 2) == -4.52 and
                                                                round(pooled(sim, 'best_generic', list(INTERIOR))['mean'], 2) == -9.89),
                                        pct_change=dict(no_interior=pct(pooled(sim, 'best_generic', no_int)['mean']),
                                                        interior=pct(pooled(sim, 'best_generic', list(INTERIOR))['mean'])))
    res['e_selection'] = dict(simulation=selection(sim, rng), room_grid=selection(room, rng),
                              miracle_A1=selection(mir['A1'], rng), miracle_R2=selection(mir['R2'], rng))
    (HERE / 'wp7-results.json').write_text(json.dumps(res, indent=1, default=float))
    print('wrote', HERE / 'wp7-results.json')


if __name__ == '__main__':
    main()
