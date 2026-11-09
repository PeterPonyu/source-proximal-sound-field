"""Post hoc, descriptive margins (PLAN.md): the arrival-time baselines, the measured margin by band, the simulated margin
with the source-inside condition apart, and absolute errors. Not registered. Uses the registered statistics of
analyze_v2.py, imported unchanged, on the registered rows.

    python3 -B posthoc/margins/margins.py        # writes posthoc/margins/margins.json
"""
from __future__ import annotations
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
sys.path.insert(0, str(S))
import analyze_v2 as A  # noqa: E402

INSIDE = 'A1-in'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def finite(*v):
    return all(x is not None and math.isfinite(x) for x in v)


def load(study):
    d = S / 'studies' / study / 'primary'
    assert sha(d / 'rows.jsonl') == json.loads((d / 'run-receipt.json').read_text())['outputs']['rows.jsonl'], study
    rows, errs = A.load(d)
    assert not errs, (study, len(errs))
    if study == 'TB-S2':
        for r in rows:
            r['cell_key'] = r['cell']
    else:
        rows = [r for r in rows if r['kind'] == 'mesh']
        for r in rows:
            r['cell_key'] = f"{int(r['band'])}Hz-M{r['M']}"
    return rows, json.loads((d / 'summary.json').read_text())['cells']


def per_cell(rows, kind, cells, a, comparator, family_n):
    """Registered per-condition interval (Bonferroni over the family) against comparator(c), with per-unit values."""
    by = defaultdict(list)
    for r in rows:
        by[r['cell_key']].append(r)
    out = {}
    for c in cells:
        b = comparator(c)
        d = A.cell_diff(by[c], kind, a, b)
        m, lo, hi = A.tint(list(d.values()), A.ALPHA / family_n)
        out[c] = dict(comparator=b, mean=m, low=lo, high=hi,
                      cls=A.classify(m, lo, hi) if finite(m, lo, hi) else 'undefined', per_unit=d)
    return out


def counts(per, cells):
    cl = [per[c]['cls'] for c in cells]
    return dict(wins=cl.count('win'), ties=cl.count('tie'), losses=cl.count('loss'), undefined=cl.count('undefined'),
                cells=len(cl))


def pooled(per, cells):
    units = defaultdict(list)
    for c in cells:
        for u, v in per[c]['per_unit'].items():
            units[u].append(v)
    assert len({len(v) for v in units.values()}) == 1 and len(next(iter(units.values()))) == len(cells)
    m, lo, hi = A.tint([float(np.mean(v)) for v in units.values()], A.ALPHA)
    return dict(mean=m, low=lo, high=hi, units=len(units), conditions=len(cells))


def strip(per):
    return {c: {k: v for k, v in x.items() if k != 'per_unit'} for c, x in per.items()}


def main():
    s2, s2cells = load('TB-S2')
    m2, m2cells = load('TB-M2')
    mcells = [f'{f}Hz-M{M}' for f in (125, 250, 500, 1000) for M in (24, 48, 96, 192)]
    prox = [c for c in s2cells if c != A.FAR]
    assert len(prox) == 63 and set(mcells) == set(m2cells) and INSIDE in prox

    # A: arrival-time baselines, measured
    part_a = {}
    below = [c for c in mcells if not c.startswith('1000')]
    for b in ('two_stage_point', 'two_stage_region'):
        per = per_cell(m2, 'mesh', mcells, A.PRIMARY, lambda c, b=b: b, len(mcells))
        for c in mcells:                                   # the registered per-condition result is reproduced
            reg = m2cells[c]['comparisons'][f'{A.PRIMARY}-vs-{b}']
            assert abs(reg['mean'] - per[c]['mean']) < 1e-12 and reg['cls'] == per[c]['cls'], (b, c)
        part_a[b] = dict(all=dict(**counts(per, mcells), pooled=pooled(per, mcells)),
                         below_1kHz=dict(**counts(per, below), pooled=pooled(per, below)), per_condition=strip(per))

    # B: measured margin over the best generic estimator of each condition, by band
    bg_m = lambda c: m2cells[c]['best_generic']
    per_bg = per_cell(m2, 'mesh', mcells, A.PRIMARY, bg_m, len(mcells))
    for c in mcells:
        reg = m2cells[c]['comparisons'][f'{A.PRIMARY}-vs-{bg_m(c)}']
        assert abs(reg['mean'] - per_bg[c]['mean']) < 1e-12 and reg['cls'] == per_bg[c]['cls'], c
    part_b = {str(f): pooled(per_bg, [c for c in mcells if c.startswith(f'{f}Hz')]) for f in (125, 250, 500, 1000)}
    part_b['all'] = pooled(per_bg, mcells)

    # C: simulated margin over the best generic estimator of each condition
    bg_s = lambda c: s2cells[c]['best_generic']
    per_s = per_cell(s2, 'sim', prox, A.PRIMARY, bg_s, len(prox))
    for c in prox:
        reg = s2cells[c]['comparisons'][f'{A.PRIMARY}-vs-{bg_s(c)}']
        assert abs(reg['mean'] - per_s[c]['mean']) < 1e-12 and reg['cls'] == per_s[c]['cls'], c
    m, lo, hi = A.tint(list(per_s[INSIDE]['per_unit'].values()), A.ALPHA)
    part_c = dict(all_63=pooled(per_s, prox), without_inside_62=pooled(per_s, [c for c in prox if c != INSIDE]),
                  inside_alone=dict(mean=m, low=lo, high=hi, units=len(per_s[INSIDE]['per_unit']), comparator=bg_s(INSIDE)))

    # D: absolute errors, median over problems (dB)
    med = lambda v: float(np.median(np.asarray(v, float)))
    e_m = dict(proposed=[r['e'][A.PRIMARY] for r in m2], best_generic_of_condition=[r['e'][bg_m(r['cell_key'])] for r in m2],
               arrival_time_point=[r['e']['two_stage_point'] for r in m2])
    sp = [r for r in s2 if r['cell_key'] in prox]
    e_s = dict(proposed=[r['e'][A.PRIMARY] for r in sp], best_generic_of_condition=[r['e'][bg_s(r['cell_key'])] for r in sp])
    for v in list(e_m.values()) + list(e_s.values()):
        assert all(math.isfinite(x) for x in v)
    part_d = dict(measured=dict(problems=len(m2), **{k: med(v) for k, v in e_m.items()}),
                  simulated_source_proximal=dict(problems=len(sp), **{k: med(v) for k, v in e_s.items()}))

    res = dict(status='post hoc, descriptive, not registered (PLAN.md)', delta_db=A.DELTA, alpha=A.ALPHA,
               provenance={p: sha(S / p) for p in ('analyze_v2.py', 'posthoc/margins/PLAN.md', 'posthoc/margins/margins.py',
                                                   'studies/TB-S2/primary/rows.jsonl', 'studies/TB-S2/primary/summary.json',
                                                   'studies/TB-M2/primary/rows.jsonl', 'studies/TB-M2/primary/summary.json')},
               arrival_time_baselines=part_a, measured_margin_by_band=part_b, simulated_margin=part_c, median_error_db=part_d)
    (HERE / 'margins.json').write_text(json.dumps(res, indent=1, default=float) + '\n')
    f = lambda p: f"{p['mean']:+.3f} [{p['low']:+.3f}, {p['high']:+.3f}]"
    for b, v in part_a.items():
        for k in ('all', 'below_1kHz'):
            x = v[k]
            print(f"{b:17s} {k:10s} {x['wins']}/{x['ties']}/{x['losses']} undef {x['undefined']}  pooled {f(x['pooled'])}")
    print('measured margin by band:', {k: f(v) for k, v in part_b.items()})
    print('simulated:', {k: f(v) for k, v in part_c.items()})
    print('median errors:', part_d)


if __name__ == '__main__':
    main()
