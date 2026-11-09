"""Post hoc analyses from saved outputs only (posthoc/review4/PLAN.md, part A). No estimate is refitted.

A1 single-point decomposition on the room grid and both anechoic scenarios; A2 effect size by microphone count;
A3 the 125 Hz floor of the room grid; A4 the spherical-failure counts, copied; A5 the direction of the bias of the
best generic estimator selected after the run. The registered statistics of analyze_v2.py are imported unchanged.

    python3 -B posthoc/review4/saved.py posthoc/review4/saved.json
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
sys.path.insert(0, str(S))
import analyze_v2 as AV  # noqa: E402

P = AV.PRIMARY
INPUTS = {
    'room': S / 'studies/TB-M2/primary/rows.jsonl',
    'anechoic': S / 'studies/TB-M3-MIRACLE/primary/rows.jsonl',
    'meshrir_prep': S / 'B3-measured/meshrir-prep/meshrir-prep.npz',
    'sphere_diag': S / 'posthoc/review3/reproduction-diagnostic-summary.json',
    'analyze_v2': S / 'analyze_v2.py',
}
DECOMP = [(P, 'j_K1'), ('j_K1', 'two_stage_point'), ('j_K5', 'j_K1'), (P, 'j_K5'), (P, 'two_stage_point'), (P, 'ref_point')]
MEANS = (P, 'j_K1', 'j_K5', 'ref_point', 'two_stage_point')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def grids():
    """name -> (rows, cells); measured rows use the registered unit (loudspeaker or source, draws averaged)."""
    out = {}
    rows, _ = AV.load(INPUTS['room'].parent)
    rows = [r for r in rows if r['kind'] == 'mesh']
    out['room'] = rows
    rows, _ = AV.load(INPUTS['anechoic'].parent)
    for sc in ('A1', 'R2'):
        out[f'anechoic_{sc}'] = [r for r in rows if r['scenario'] == sc]
    res = {}
    for g, rs in out.items():
        for r in rs:
            r['cell_key'] = f"{int(r['band'])}Hz-M{r['M']}"
        cells = sorted({r['cell_key'] for r in rs}, key=lambda c: (int(c.split('Hz')[0]), int(c.split('M')[1])))
        res[g] = (rs, cells)
    return res


def by_cell(rs):
    b = defaultdict(list)
    for r in rs:
        b[r['cell_key']].append(r)
    return b


def pooled(by, cells, a, b):
    """Registered pooling: per unit, mean over cells of its per-cell difference; 95 % t interval over units.
    `b` is a method name or a dict cell -> method."""
    per = defaultdict(list)
    for c in cells:
        bb = b[c] if isinstance(b, dict) else b
        for u, v in AV.cell_diff(by[c], 'mesh', a, bb).items():
            per[u].append(v)
    vals = [np.mean(v) for v in per.values()]
    m, lo, hi = AV.tint(vals, AV.ALPHA)
    return dict(mean=m, low=lo, high=hi, units=len(vals), cells=len(cells), finite=bool(np.isfinite(m)))


def a1(G):
    out = {}
    for g, (rs, cells) in G.items():
        by = by_cell(rs)
        bands = sorted({int(c.split('Hz')[0]) for c in cells})
        per_cell = {}
        for c in cells:
            d = {}
            for a, b in DECOMP:
                vals = list(AV.cell_diff(by[c], 'mesh', a, b).values())
                m, lo, hi = AV.tint(vals, AV.ALPHA / len(cells))
                d[f'{a}-minus-{b}'] = dict(mean=m, low=lo, high=hi, cls=AV.classify(m, lo, hi))
            per_cell[c] = dict(mean_db={m: float(np.mean([r['e'][m] for r in by[c]])) for m in MEANS}, differences=d)
        per_band = {}
        for bd in bands:
            cs = [c for c in cells if c.startswith(f'{bd}Hz')]
            per_band[str(bd)] = dict(mean_db={m: float(np.mean([r['e'][m] for c in cs for r in by[c]])) for m in MEANS},
                                     pooled={f'{a}-minus-{b}': pooled(by, cs, a, b) for a, b in DECOMP})
        # share of the single-point-to-five-point gap closed by localization, per band (descriptive)
        for bd, v in per_band.items():
            gap = v['pooled']['j_K5-minus-j_K1']['mean'] + v['pooled']['j_K1-minus-two_stage_point']['mean']
            v['j_K5_minus_two_stage_point'] = gap
        out[g] = dict(per_cell=per_cell, per_band=per_band,
                      all_bands={f'{a}-minus-{b}': pooled(by, cells, a, b) for a, b in DECOMP})
    return out


def a2(G):
    out = {}
    for g, (rs, cells) in G.items():
        by = by_cell(rs)
        T = AV.analyse(rs, 'mesh', cells, len(cells))
        best = {c: T[c]['best_generic'] for c in cells}
        lo_m = [c for c in cells if int(c.split('M')[1]) <= 48]
        hi_m = [c for c in cells if int(c.split('M')[1]) >= 96]
        cmp = {c: T[c]['comparisons'][f'{P}-vs-{best[c]}'] for c in cells}
        wins = [c for c in cells if cmp[c]['cls'] == 'win']
        out[g] = dict(best_generic=best,
                      pooled_M_le_48=pooled(by, lo_m, P, best), pooled_M_ge_96=pooled(by, hi_m, P, best),
                      per_cell_vs_best_generic={c: dict(mean=cmp[c]['mean'], cls=cmp[c]['cls']) for c in cells},
                      wins=len(wins), wins_with_mean_below_minus_0p5_dB=int(sum(cmp[c]['mean'] < -0.5 for c in wins)))
    return out


def a3(G):
    rs, cells = G['room']
    by = by_cell(rs)
    methods = sorted(set.intersection(*[set(r['nmse']) for r in by['125Hz-M192']]))
    per_bin_median = {}
    for m in methods:
        arr = np.array([r['nmse'][m] for r in by['125Hz-M192']], float)
        per_bin_median[m] = [float(np.median(arr[:, f])) for f in range(arr.shape[1])]
    flat_in_M = {m: [float(np.mean([r['e'][m] for r in by[f'125Hz-M{M}']])) for M in (24, 48, 96, 192)]
                 for m in (P, 'sinc_ev', 'rbf', 'adk_standin', 'two_stage_point', 'ref_point')}
    z = np.load(INPUTS['meshrir_prep'], allow_pickle=False)
    H, band, freqs = z['H'], z['band'], z['freqs']
    pw = {}
    for fc in (125.0, 1000.0):
        sel = np.where(band == fc)[0][1::2]                        # the six registered bins of the band
        pw[str(int(fc))] = [float(10 * np.log10(np.median(np.abs(H[:, :, b]) ** 2))) for b in sel]
        pw[f'{int(fc)}_freqs_hz'] = [float(freqs[b]) for b in sel]
    ref = float(np.mean(pw['1000']))
    rel = [v - ref for v in pw['125']]
    spread = max(v[-1] - v[0] for v in flat_in_M.values())
    flat = bool(all(abs(v[-1] - v[0]) < 1.0 for v in flat_in_M.values()))
    reading = ('floor attributed to low source output below about 100 Hz' if (flat and min(rel) <= -20)
               else 'cause of the floor not known')
    return dict(per_bin_median_nmse_M192=per_bin_median, mean_db_125Hz_by_M=flat_in_M,
                band_power_db_median_over_positions=pw, rel_power_125_bins_vs_1k_mean_db=rel,
                flat_in_M_all_methods_within_1dB=flat, max_change_24_to_192_db=spread, reading=reading,
                rule='flat in M (within 1 dB) for every listed method AND some 125 Hz bin >= 20 dB below the 1 kHz band')


def a4():
    d = json.loads(INPUTS['sphere_diag'].read_text())
    sh = d['shared']
    return dict(source='posthoc/review3/reproduction-diagnostic-summary.json (copied, not refitted)',
                shared_failing_bins=sh['failing'], shared_failing_with_point_within_1cm=sh['failing_with_point'],
                shared_max_point_distance_m=sh['max_dist_m_all_bins'],
                shared_failing_weight_within_decade_of_bound=sh['failing_first_weight_within_a_decade_of_bound'],
                shared_failing_meeting_prestated_collapse=sh['failing_collapsed'],
                sinc_condition_median_by_freq=sh['sinc_cond_median_by_freq'])


def a5(G, seed=20261108, n_rep=200):
    """Split units in half (fixed seed); select the best generic on half A; compare its mean on A (in-sample) with
    its mean on B (out of sample). Positive out-minus-in = the selected comparator looks better in-sample."""
    out = {}
    rng = np.random.default_rng(seed)
    for g, (rs, cells) in G.items():
        by = by_cell(rs)
        res = {}
        for c in cells:
            per = defaultdict(lambda: defaultdict(list))
            for r in by[c]:
                for m in AV.GENERIC:
                    per[r['source']][m].append(r['e'][m])
            units = sorted(per)
            U = {m: np.array([np.mean(per[u][m]) for u in units]) for m in AV.GENERIC}
            n = len(units)

            def one(idx):
                A = np.zeros(n, bool); A[idx[:n // 2]] = True
                mA = {m: float(np.mean(U[m][A])) for m in AV.GENERIC}
                ok = [m for m in AV.GENERIC if np.isfinite(mA[m])]
                b = min(ok, key=lambda m: mA[m])
                return b, float(np.mean(U[b][~A]) - mA[b])
            b0, gap0 = one(np.random.default_rng(seed).permutation(n))
            reps = [one(rng.permutation(n))[1] for _ in range(n_rep)]
            reps = [v for v in reps if np.isfinite(v)]
            res[c] = dict(selected_on_A=b0, out_minus_in_db=gap0, mean_over_splits=float(np.mean(reps)) if reps else None)
        vals = [v['mean_over_splits'] for v in res.values() if v['mean_over_splits'] is not None]
        out[g] = dict(per_cell=res, cells_out_minus_in_positive_fixed_split=int(sum(v['out_minus_in_db'] > 0 for v in res.values()
                                                                                    if np.isfinite(v['out_minus_in_db']))),
                      cells=len(cells), mean_out_minus_in_over_cells_db=float(np.mean(vals)) if vals else None)
    return out


def main():
    G = grids()
    out = dict(status='post hoc; saved outputs only; no estimate refitted; posthoc/review4/PLAN.md part A',
               inputs={k: sha(v) for k, v in INPUTS.items()},
               A1_single_point_decomposition=a1(G), A2_effect_size_by_M=a2(G), A3_room_125Hz_floor=a3(G),
               A4_sphere_failure_counts=a4(), A5_selection_bias=a5(G))
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
