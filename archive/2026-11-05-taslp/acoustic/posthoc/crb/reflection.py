"""Post hoc addendum to the identifiability check (REFLECTION-PLAN.md): the reflection ambiguity of the planar measured
grid, and the per-problem attainment e_J / B_J. Descriptive, not registered. Reads committed outputs only.

    python3 -B posthoc/crb/reflection.py        # writes posthoc/crb/reflection.json
"""
from __future__ import annotations
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import chi2, norm

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
NEAR_PLANE_M = 1e-3


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def registered_rows(study):
    d = S / 'studies' / study / 'primary'
    receipt = json.loads((d / 'run-receipt.json').read_text())
    assert sha(d / 'rows.jsonl') == receipt['outputs']['rows.jsonl'], f'{study} rows do not match their receipt'
    rows = [json.loads(line) for line in (d / 'rows.jsonl').read_text().splitlines()]
    assert not any('error' in r for r in rows)
    return rows


def med(v):
    return float(np.median(np.asarray(v, float)))


def spread(v):
    return dict(median=med(v), min=float(np.min(v)), max=float(np.max(v)))


def main():
    s2 = registered_rows('TB-S2')
    m2 = [r for r in registered_rows('TB-M2') if r['kind'] == 'mesh']
    assert len(s2) == 2048 and len(m2) == 1024
    prep = np.load(S / 'B3-measured/meshrir-prep/meshrir-prep.npz')
    pos, src = prep['pos_mic'], prep['pos_src']
    z0 = float(np.mean(pos[:, 2]))

    # Part A: the reflection on the measured grid
    rec, by_band = [], defaultdict(list)
    for r in m2:
        fit = np.asarray(r['info']['j_k1_S'], float).reshape(3)
        s = src[r['source']]
        mirror = np.array([s[0], s[1], 2 * z0 - s[2]])
        d_cat, d_mir = float(np.linalg.norm(fit - s)), float(np.linalg.norm(fit - mirror))
        x = dict(d_cat=d_cat, d_mir=d_mir, d_inv=min(d_cat, d_mir), stored=float(r['info']['j_k1_loc_m']),
                 near_plane=bool(abs(fit[2] - z0) < NEAR_PLANE_M), band=int(r['band']), M=int(r['M']))
        rec.append(x)
        by_band[x['band']].append(x)
    stored_diff = max(abs(x['d_cat'] - x['stored']) for x in rec)
    assert stored_diff < 1e-9, stored_diff
    part_a = dict(plane_z_m=z0, plane_z_spread_m=float(np.ptp(pos[:, 2])),
                  source_heights_relative_to_plane_m=sorted({round(float(v - z0), 4) for v in src[:, 2]}),
                  problems=len(rec), max_abs_diff_from_registered_distance_m=stored_diff,
                  closer_to_mirror=int(sum(x['d_mir'] < x['d_cat'] for x in rec)),
                  equidistant=int(sum(x['d_mir'] == x['d_cat'] for x in rec)),
                  within_1mm_of_plane=int(sum(x['near_plane'] for x in rec)),
                  median_d_catalogue_m=med([x['d_cat'] for x in rec]),
                  median_d_reflection_invariant_m=med([x['d_inv'] for x in rec]),
                  by_band={str(b): dict(problems=len(v), closer_to_mirror=int(sum(x['d_mir'] < x['d_cat'] for x in v)),
                                        median_d_catalogue_m=med([x['d_cat'] for x in v]),
                                        median_d_reflection_invariant_m=med([x['d_inv'] for x in v]))
                           for b, v in sorted(by_band.items())})

    # Part B: per-problem attainment e_J / B_J, conditions as in analyze_bounds.py
    inv = {(r['source'], r['band'], r['M'], r['seed']): x['d_inv'] for r, x in zip(m2, rec)}
    obs = {('TB-S2', r['cell'], r['seed']): r['info']['j_k1_loc_m'] for r in s2}
    obs.update({('TB-M2', r['source'], r['band'], r['M'], r['seed']): r['info']['j_k1_loc_m'] for r in m2})
    bnd = [json.loads(line) for line in (HERE / 'bounds.jsonl').read_text().splitlines()]
    assert len(bnd) == len(obs) == 3072
    groups = defaultdict(list)
    for b in bnd:
        if b['study'] == 'TB-S2':
            k, cond = ('TB-S2', b['cell'], b['seed']), b['cell']
            e_inv = None
        else:
            k, cond = ('TB-M2', b['source'], b['band'], b['M'], b['seed']), f"{int(b['band'])} Hz, M = {b['M']}"
            e_inv = inv[k[1:]]
        assert np.isfinite(b['B_J']) and b['B_J'] > 0
        groups[(b['study'], cond)].append((obs.pop(k) / b['B_J'], None if e_inv is None else e_inv / b['B_J']))
    assert not obs
    cmp_ = json.loads((HERE / 'comparison.json').read_text())['studies']
    part_b = {}
    for study in ('TB-S2', 'TB-M2'):
        conds = {c: g for (st, c), g in groups.items() if st == study}
        per = {c: med([r for r, _ in g]) for c, g in sorted(conds.items())}
        out = dict(conditions=len(per), per_condition_median=per,
                   median_over_conditions=spread(list(per.values())),
                   all_problems_median=med([r for g in conds.values() for r, _ in g]),
                   ratio_of_medians_A_J_median_over_conditions=cmp_[study]['A_J_median'])
        if study == 'TB-M2':
            per_inv = {c: med([ri for _, ri in g]) for c, g in sorted(conds.items())}
            out['reflection_invariant'] = dict(per_condition_median=per_inv,
                                               median_over_conditions=spread(list(per_inv.values())),
                                               all_problems_median=med([ri for g in conds.values() for _, ri in g]))
        part_b[study] = out
    res = dict(status='post hoc, descriptive, not registered (REFLECTION-PLAN.md)',
               provenance={p: sha(S / p) for p in ('posthoc/crb/REFLECTION-PLAN.md', 'posthoc/crb/reflection.py',
                                                   'posthoc/crb/bounds.jsonl', 'posthoc/crb/comparison.json',
                                                   'studies/TB-S2/primary/rows.jsonl', 'studies/TB-M2/primary/rows.jsonl',
                                                   'B3-measured/meshrir-prep/meshrir-prep.npz')},
               efficient_median_range=[float(norm.ppf(0.75)), float(np.sqrt(chi2.ppf(0.5, 3) / 3.0))],
               reflection_measured=part_a, attainment=part_b)
    (HERE / 'reflection.json').write_text(json.dumps(res, indent=1) + '\n')
    a = part_a
    print(f"plane z {z0} (spread {a['plane_z_spread_m']}), source heights {a['source_heights_relative_to_plane_m']}")
    print(f"closer to mirror {a['closer_to_mirror']} of {a['problems']}, equidistant {a['equidistant']}, "
          f"within 1 mm of the plane {a['within_1mm_of_plane']}; registered-distance check {a['max_abs_diff_from_registered_distance_m']:.1e}")
    print(f"median distance: catalogue {a['median_d_catalogue_m']:.4f} m, reflection-invariant {a['median_d_reflection_invariant_m']:.4f} m")
    for b, v in a['by_band'].items():
        print(f"  {b} Hz: mirror {v['closer_to_mirror']}/{v['problems']}, medians {v['median_d_catalogue_m']:.3f} / {v['median_d_reflection_invariant_m']:.3f} m")
    for st, v in part_b.items():
        m = v['median_over_conditions']
        print(f"{st}: per-problem attainment, median over conditions {m['median']:.3f} (range {m['min']:.2f}-{m['max']:.2f}); "
              f"all problems {v['all_problems_median']:.3f}; ratio of medians {v['ratio_of_medians_A_J_median_over_conditions']:.3f}")
        if 'reflection_invariant' in v:
            m = v['reflection_invariant']['median_over_conditions']
            print(f"  reflection-invariant: median over conditions {m['median']:.3f} (range {m['min']:.2f}-{m['max']:.2f})")


if __name__ == '__main__':
    main()
