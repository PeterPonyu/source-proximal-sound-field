"""Statistics of the post hoc identifiability check, exactly as fixed in PLAN.md. Not registered.

    python3 -B posthoc/crb/analyze_bounds.py        # reads bounds.jsonl and the registered rows, writes comparison.json
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
BAND = (0.5, 2.0)          # agreement band for R_obs / R_pred
SHARE = 0.75               # share of conditions that must lie in it


def med(v):
    return float(np.median(np.asarray(v, float)))


def rms(v):
    v = np.asarray(v, float)
    return float(np.sqrt(np.mean(v ** 2))) if np.all(np.isfinite(v)) else float('inf')


def ratio(a, b):
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 0 else None


def load():
    obs = {}
    for line in (S / 'studies/TB-S2/primary/rows.jsonl').read_text().splitlines():
        r = json.loads(line)
        obs[('TB-S2', r['cell'], r['seed'])] = (r['info']['j_k1_loc_m'], r['info']['ind_k1_loc_m'])
    for line in (S / 'studies/TB-M2/primary/rows.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['kind'] == 'mesh':
            obs[('TB-M2', r['source'], r['band'], r['M'], r['seed'])] = (r['info']['j_k1_loc_m'], r['info']['ind_k1_loc_m'])
    bnd = [json.loads(line) for line in (HERE / 'bounds.jsonl').read_text().splitlines()]
    return obs, bnd


def main():
    obs, bnd = load()
    assert len(bnd) == len(obs) == 2048 + 1024, (len(bnd), len(obs))
    worst = max(b['rebuild_rel_diff'] for b in bnd)
    assert worst < 1e-6, f'rebuilt problems differ from the registered ones (relative sinc error difference {worst})'
    groups = defaultdict(list)
    for b in bnd:
        if b['study'] == 'TB-S2':
            k, cond = ('TB-S2', b['cell'], b['seed']), b['cell']
        else:
            k, cond = ('TB-M2', b['source'], b['band'], b['M'], b['seed']), f"{int(b['band'])} Hz, M = {b['M']}"
        eJ, ef = obs.pop(k)
        groups[(b['study'], cond)].append((b, eJ, ef))
    assert not obs
    res = dict(plan='PLAN.md', status='post hoc, descriptive, not registered', band=BAND, share_required=SHARE,
               rebuild_max_rel_diff=worst, studies={})
    for study in ('TB-S2', 'TB-M2'):
        conds = []
        for (st, cond), g in sorted(groups.items()):
            if st != study:
                continue
            eJ = [x[1] for x in g]; ef = [e for x in g for e in x[2]]
            BJ = [x[0]['B_J'] for x in g]; Bf = [v for x in g for v in x[0]['B_f']]
            c = dict(condition=cond, problems=len(g),
                     R_obs=ratio(med(eJ), med(ef)), R_pred=ratio(med(BJ), med(Bf)),
                     A_J=ratio(med(eJ), med(BJ)), A_f=ratio(med(ef), med(Bf)),
                     R_obs_rms=ratio(rms(eJ), rms(ef)), R_pred_rms=ratio(rms(BJ), rms(Bf)),
                     R_pred_a=ratio(med([x[0]['B_J_a'] for x in g]), med([v for x in g for v in x[0]['B_f_a']])),
                     R_pred_b=ratio(med([x[0]['B_J_b'] for x in g]), med([v for x in g for v in x[0]['B_f_b']])),
                     median_e_J=med(eJ), median_e_f=med(ef), median_B_J=med(BJ), median_B_f=med(Bf))
            c['obs_over_pred'] = ratio(c['R_obs'], c['R_pred']) if c['R_obs'] and c['R_pred'] else None
            c['in_band'] = c['obs_over_pred'] is not None and BAND[0] <= c['obs_over_pred'] <= BAND[1]
            conds.append(c)
        q = [c['obs_over_pred'] for c in conds if c['obs_over_pred'] is not None]
        share = float(np.mean([c['in_band'] for c in conds]))
        pr = [(c['R_obs'], c['R_pred']) for c in conds if c['R_obs'] is not None and c['R_pred'] is not None]
        rho_s = spearmanr([a for a, _ in pr], [b for _, b in pr])
        pl = [p for x in groups.items() if x[0][0] == study for b, _, _ in x[1] for p in b['plugin']]
        res['studies'][study] = dict(
            conditions=len(conds), share_in_band=share, tracks=bool(share >= SHARE),
            obs_over_pred_median=med(q), obs_over_pred_min=float(min(q)), obs_over_pred_max=float(max(q)),
            spearman_R_obs_R_pred=float(rho_s.statistic), spearman_p=float(rho_s.pvalue),
            R_obs_median=med([c['R_obs'] for c in conds if c['R_obs'] is not None]),
            R_pred_median=med([c['R_pred'] for c in conds if c['R_pred'] is not None]),
            R_pred_a_median=med([c['R_pred_a'] for c in conds if c['R_pred_a'] is not None]),
            R_pred_b_median=med([c['R_pred_b'] for c in conds if c['R_pred_b'] is not None]),
            R_obs_rms_median=med([c['R_obs_rms'] for c in conds if c['R_obs_rms'] is not None]),
            R_pred_rms_median=med([c['R_pred_rms'] for c in conds if c['R_pred_rms'] is not None]),
            A_J_median=med([c['A_J'] for c in conds if c['A_J'] is not None]),
            A_f_median=med([c['A_f'] for c in conds if c['A_f'] is not None]),
            plugin_eta_median=med([p['eta'] for p in pl]), plugin_rho_median=med([p['rho'] for p in pl]),
            plugin_rho_at_bound_share=float(np.mean([p['at_bound'] for p in pl])),
            per_condition=conds)
    (HERE / 'comparison.json').write_text(json.dumps(res, indent=1) + '\n')
    for study, r in res['studies'].items():
        print(study, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != 'per_condition'})


if __name__ == '__main__':
    main()
