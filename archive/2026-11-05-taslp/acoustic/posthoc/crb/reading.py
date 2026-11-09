"""A reading of the identifiability comparison, WRITTEN AFTER comparison.json was read. Not in PLAN.md, not registered.

The plan's criterion failed in both studies. The per-condition table suggested two patterns, and this script counts them
so that the manuscript can state them from committed outputs. Both are descriptive and post hoc.

1. Simulation. Group the cells by source structure: a single driver (no vertical driver offset, no second driver) at
   least 0.6 m above the floor, against every other cell. The source height comes from the cell definition: the
   placement 'below' puts the driver at the cell's height; 'beside' and 'inside' put it at 0.85 m or more.
2. Conditions below the band (observed ratio less than half the predicted one): their attainment, joint and per bin.

Also recorded: the median distance over the root-mean-square bound that an efficient, unbiased fit would give with
Gaussian errors, between one dominant axis (the median of |N(0, 1)|) and three equal axes (the median of a chi
variable with three degrees of freedom over the square root of three).

    python3 -B posthoc/crb/reading.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import chi2, norm

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
sys.path.insert(0, str(S / 'common'))
import sim  # noqa: E402

FLOOR_CLEAR_M = 0.6


def source_height_min(cell):
    """Lowest possible driver height of a cell (m)."""
    if cell['place'] == 'below':
        return float(cell['height'])
    if cell['place'] == 'beside':
        return 0.70 + 0.15
    if cell['place'] == 'inside':
        return 0.70 + 0.25
    raise ValueError(cell['place'])


def summary(cs):
    return dict(conditions=len(cs), in_band=int(sum(c['in_band'] for c in cs)),
                R_obs_median=float(np.median([c['R_obs'] for c in cs])),
                R_pred_median=float(np.median([c['R_pred'] for c in cs])),
                A_J_median=float(np.median([c['A_J'] for c in cs])),
                names=[c['condition'] for c in cs])


def main():
    cmp_ = json.loads((HERE / 'comparison.json').read_text())
    lo_b, _ = cmp_['band']
    cells = sim.cells()
    s2 = cmp_['studies']['TB-S2']['per_condition']
    single = [c for c in s2 if source_height_min(cells[c['condition']]) >= FLOOR_CLEAR_M
              and not cells[c['condition']]['delta'] and not cells[c['condition']]['second']]
    names = {c['condition'] for c in single}
    other = [c for c in s2 if c['condition'] not in names]
    below = [dict(study=st, condition=c['condition'], obs_over_pred=c['obs_over_pred'], A_J=c['A_J'], A_f=c['A_f'])
             for st in ('TB-S2', 'TB-M2') for c in cmp_['studies'][st]['per_condition']
             if c['obs_over_pred'] is not None and c['obs_over_pred'] < lo_b]
    m2 = cmp_['studies']['TB-M2']['per_condition']
    by_M = {}
    for c in m2:
        M = int(c['condition'].split('M = ')[1])
        by_M.setdefault(M, []).append(c)
    eff_lo = float(norm.ppf(0.75))
    eff_hi = float(np.sqrt(chi2.ppf(0.5, 3) / 3.0))
    out = dict(status='post hoc reading, written after comparison.json was read; descriptive, not registered',
               floor_clearance_m=FLOOR_CLEAR_M,
               s2_single_driver_clear_of_floor=summary(single), s2_other=summary(other),
               below_band=below,
               below_band_A_J_range=[min(b['A_J'] for b in below), max(b['A_J'] for b in below)] if below else None,
               below_band_A_f_range=[min(b['A_f'] for b in below), max(b['A_f'] for b in below)] if below else None,
               m2_by_microphones={str(M): summary(v) for M, v in sorted(by_M.items())},
               efficient_median_over_rms=[eff_lo, eff_hi])
    (HERE / 'reading.json').write_text(json.dumps(out, indent=1) + '\n')
    for k, v in out.items():
        if isinstance(v, dict) and 'names' in v:
            v = {a: b for a, b in v.items() if a != 'names'}
        print(k, v)


if __name__ == '__main__':
    main()
