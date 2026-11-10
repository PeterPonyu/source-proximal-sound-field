"""Checks of the multi-frequency SBL comparator (PLAN.md) -> checks.json.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python3 -B checks.py [--workers 3]
1 reduction to sbl_guard.sbl_guarded with one bin; 2 noiseless known answer over six bins; 3 finiteness on 8 real problems;\n4 descriptive known answers at A2's distance (on a node, between nodes).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(S / 'studies' / 'TB-M3-MIRACLE'))
import mfsbl as MF  # noqa: E402
import run_m3 as RM  # noqa: E402  (imports also apply the TB-M3 corrections; only the extract loader and builder are used)
import estimators as E  # noqa: E402
import problems_v2 as PB  # noqa: E402
import sbl_guard as G  # noqa: E402
import v2  # noqa: E402

EXTRACT = S / 'studies/TB-M3-MIRACLE/extract'
MESH = S / 'B3-measured/meshrir-prep/meshrir-prep.npz'
MESH_DATA = {}


def init():
    RM.init(EXTRACT, ['A1'])
    z = np.load(MESH, allow_pickle=False)
    MESH_DATA.update({k: z[k] for k in z.files})


def a1_seed(spk_rank, M, draw):
    """Seed of the registered TB-M3 A1 problem for the spk_rank-th source, microphone count M and draw."""
    t = [t for t in RM.tasks(['A1']) if t[3] == M and t[4] == draw]
    return sorted({(x[1], x[5]) for x in t})[spk_rank]


def real_band(kind, fc, M, draw=0):
    if kind == 'A1':
        spk, seed = a1_seed(0, M, draw)
        band, T, _, _ = RM.build('A1', spk, fc, M, seed)
    else:
        seed = 403000 + {24: 0, 48: 2, 96: 4, 192: 6}[M] + draw          # TB-M2 order (loudspeaker 0, M, draw)
        band, T, _, _ = PB.meshrir_band(MESH_DATA, 0, fc, M, seed)
    return band, T, seed


def one_bin(band, f=0):
    return v2.Band(band.x, band.probs[f].y[:, None], band.q, band.freqs[f:f + 1], c=band.c, box=band.box)


def check1(job):
    kind, fc, rule = job
    band, _, seed = real_band(kind, fc, 24)
    b1 = one_bin(band)
    ref = G.sbl_guarded(b1.probs[0], rule)
    got = MF.fit(b1, rule)
    rel = float(np.linalg.norm(got['preds'][0] - ref['pred']) / np.linalg.norm(ref['pred']))
    return dict(set=kind, band=fc, M=24, seed=seed, bin_hz=float(b1.freqs[0]), rule=rule, rel_diff=rel,
                iterations=[got['iterations'], ref['iterations']], active=[got['active'], ref['active']],
                passed=bool(rel <= 1e-6 and got['iterations'] == ref['iterations'] and got['active'] == ref['active']))


def check2(job):
    """variant 'planar': the check fixed in PLAN.md. 'lifted' (added after 'planar' failed on the mirror ambiguity of a
    planar array): every second microphone 0.1 m off the plane, which breaks the mirror symmetry."""
    fc, rule, variant = job
    prep = RM.DATA['A1']
    pos = prep['pos_mic']
    mic = np.arange(0, len(pos), 9)                                        # 49 microphones, no random draw
    tgt = np.setdiff1d(np.arange(len(pos)), mic)
    normal = int(np.argmin(np.ptp(pos, axis=0)))
    x = pos[mic].copy()
    if variant == 'lifted':
        x[::2, normal] += 0.1
    freqs = prep['freqs'][prep['band'] == fc][1::2]
    box = (pos.min(0), pos.max(0)); c = 343.0
    shell = v2.Band(x, np.ones((len(mic), len(freqs))), pos[tgt], freqs, c=c, box=box)
    nodes, _ = MF.dictionary(shell)
    want = prep['pos_src'][np.sort(np.random.default_rng(414600).choice(64, 32, replace=False))[0]]
    idx = int(np.argmin(np.linalg.norm(nodes - want, axis=1)))
    s = nodes[idx][None, :]
    mir = s[0].copy(); mir[normal] = 2 * pos[0, normal] - mir[normal]
    imir = int(np.argmin(np.linalg.norm(nodes - mir, axis=1)))
    k = 2 * np.pi * freqs / c
    Y = np.stack([E.green(x, s, kk)[:, 0] for kk in k], 1)
    T = np.stack([E.green(pos[tgt], s, kk)[:, 0] for kk in k], 1)
    band = v2.Band(x, Y, pos[tgt], freqs, c=c, box=box)
    r = MF.fit(band, rule)
    gm = r['gamma'][:len(r['nodes'])]
    top = int(np.argmax(gm))
    share = float(r['gamma'][idx] / r['gamma'].sum())
    share_pair = float((r['gamma'][idx] + (r['gamma'][imir] if imir != idx else 0.0)) / r['gamma'].sum())
    e_mf = 10 * np.log10(np.mean([E.nmse(r['preds'][f], T[:, f]) for f in range(band.F)]))
    e_sf = 10 * np.log10(np.mean([E.nmse(G.sbl_guarded(p, rule)['pred'], T[:, f]) for f, p in enumerate(band.probs)]))
    ok_a = top == idx; ok_b = share >= 0.5; ok_c = bool(e_mf <= e_sf + 0.1 or (e_mf < -40 and e_sf < -40))
    out = dict(variant=variant, band=fc, rule=rule, M=len(mic), source_m=s[0].tolist(),
               distance_to_grid_m=float(np.linalg.norm(pos - s, axis=1).min()), mirror_node_offset_m=float(np.linalg.norm(nodes[imir] - mir)),
               top_is_true=bool(ok_a), top_is_mirror=bool(top == imir), top_offset_m=float(np.linalg.norm(r['nodes'][top] - s[0])),
               true_share=share, true_plus_mirror_share=share_pair, active=r['active'], iterations=r['iterations'], stopped_by_tolerance=r['stopped_by_tolerance'],
               error_db_multi=float(e_mf), error_db_single=float(e_sf), passed=bool(ok_a and ok_b and ok_c),
               parts=dict(a=bool(ok_a), b=bool(ok_b), c=ok_c))
    if variant == 'planar':                                               # mirror-aware reading, added after the failure
        ok_a2 = top in (idx, imir); ok_b2 = share_pair >= 0.5
        out['mirror_aware'] = dict(a=bool(ok_a2), b=bool(ok_b2), c=ok_c, passed=bool(ok_a2 and ok_b2 and ok_c))
    return out


def check3(job):
    kind, fc, M = job
    band, T, seed = real_band(kind, fc, M)
    res = []
    for rule in ('em', 'mackay'):
        r = MF.fit(band, rule)
        g, n = r['gamma'], np.asarray(r['noise'])
        ok = bool(np.all(np.isfinite(g)) and np.all(g >= 0) and np.all(np.isfinite(n)) and np.all(n > 0)
                  and all(np.all(np.isfinite(p)) for p in r['preds']))
        res.append(dict(rule=rule, ok=ok, iterations=r['iterations'], stopped_by_tolerance=r['stopped_by_tolerance'], active=r['active'],
                        noise_min=float(n.min()), gamma_max=float(g.max()),
                        error_db=float(10 * np.log10(np.mean([E.nmse(r['preds'][f], T[:, f]) for f in range(band.F)])))))
    return dict(set=kind, band=fc, M=M, seed=seed, rules=res, passed=all(x['ok'] for x in res))


def check4(job):
    """Descriptive (PLAN.md, added before running): monopole 1.467 m from the A1 grid plane, on a node or between nodes."""
    fc, rule, placement = job
    prep = RM.DATA['A1']
    pos = prep['pos_mic']
    mic = np.arange(0, len(pos), 9)
    tgt = np.setdiff1d(np.arange(len(pos)), mic)
    normal = int(np.argmin(np.ptp(pos, axis=0)))
    freqs = prep['freqs'][prep['band'] == fc][1::2]
    box = (pos.min(0), pos.max(0)); c = 343.0
    shell = v2.Band(pos[mic], np.ones((len(mic), len(freqs))), pos[tgt], freqs, c=c, box=box)
    nodes, _ = MF.dictionary(shell)
    arr = prep['pos_src'][np.sort(np.random.default_rng(414600).choice(64, 32, replace=False))[0]]
    side = np.sign(arr[normal] - pos[0, normal])
    want = arr.copy(); want[normal] = pos[0, normal] + side * 1.467
    idx = int(np.argmin(np.linalg.norm(nodes - want, axis=1)))
    step = np.array([np.min(np.diff(np.unique(np.round(nodes[:, j], 9)))) for j in range(3)])
    s = nodes[idx].copy()
    if placement == 'between':
        shift = 0.5 * step
        shift[normal] *= -side                                              # towards the grid, so it stays inside the box
        s = s + shift
    near = int(np.argmin(np.linalg.norm(nodes - s, axis=1)))
    mir = s.copy(); mir[normal] = 2 * pos[0, normal] - mir[normal]
    imir = int(np.argmin(np.linalg.norm(nodes - mir, axis=1)))
    k = 2 * np.pi * freqs / c
    Y = np.stack([E.green(pos[mic], s[None, :], kk)[:, 0] for kk in k], 1)
    T = np.stack([E.green(pos[tgt], s[None, :], kk)[:, 0] for kk in k], 1)
    band = v2.Band(pos[mic], Y, pos[tgt], freqs, c=c, box=box)
    r = MF.fit(band, rule)
    g = r['gamma']; tot = g.sum()
    top = int(np.argmax(g[:len(r['nodes'])]))
    e_mf = 10 * np.log10(np.mean([E.nmse(r['preds'][f], T[:, f]) for f in range(band.F)]))
    e_sf = 10 * np.log10(np.mean([E.nmse(G.sbl_guarded(p, rule)['pred'], T[:, f]) for f, p in enumerate(band.probs)]))
    finite = bool(np.all(np.isfinite(g)) and np.all(np.isfinite(r['noise'])) and all(np.all(np.isfinite(p)) for p in r['preds']))
    return dict(placement=placement, band=fc, rule=rule, M=len(mic), source_m=s.tolist(),
                distance_to_grid_plane_m=float(abs(s[normal] - pos[0, normal])), node_spacing_m=step.tolist(),
                source_to_nearest_node_m=float(np.linalg.norm(nodes[near] - s)),
                nearest_share=float(g[near] / tot), nearest_plus_mirror_share=float((g[near] + (g[imir] if imir != near else 0)) / tot),
                top_to_source_m=float(np.linalg.norm(r['nodes'][top] - s)), top_is_mirror=bool(top == imir),
                active=r['active'], iterations=r['iterations'], stopped_by_tolerance=r['stopped_by_tolerance'],
                error_db_multi=float(e_mf), error_db_single=float(e_sf), passed=finite)


def run(job):
    t0 = time.process_time()
    out = {1: check1, 2: check2, 3: check3, 4: check4}[job[0]](job[1])
    out['process_seconds'] = round(time.process_time() - t0, 1)
    return job[0], out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    jobs = [(1, (k, fc, r)) for k, fc in (('A1', 250.0), ('A1', 1000.0), ('room', 500.0)) for r in ('em', 'mackay')]
    jobs += [(2, (fc, r, v)) for v in ('planar', 'lifted') for fc in (250.0, 500.0, 1000.0) for r in ('em', 'mackay')]
    jobs += [(3, (k, fc, M)) for k in ('A1', 'room') for fc in (250.0, 1000.0) for M in (24, 48)]
    jobs += [(4, (fc, r, pl)) for pl in ('node', 'between') for fc in (250.0, 500.0, 1000.0) for r in ('em', 'mackay')]
    res = {1: [], 2: [], 3: [], 4: []}
    with Pool(a.workers, initializer=init) as pool:
        for n, r in pool.imap_unordered(run, jobs):
            res[n].append(r); print(n, json.dumps({k: v for k, v in r.items() if k not in ('rules',)}, default=float), flush=True)
    planar = [x for x in res[2] if x['variant'] == 'planar']; lifted = [x for x in res[2] if x['variant'] == 'lifted']
    out = dict(reduction_to_sbl_guarded=dict(passed=bool(all(x['passed'] for x in res[1])), cases=res[1]),
               known_answer_six_bins=dict(passed=bool(all(x['passed'] for x in planar)), note='as fixed in PLAN.md', cases=planar),
               finite_on_real_problems=dict(passed=bool(all(x['passed'] for x in res[3])), cases=res[3]))
    out['all_passed'] = bool(all(v['passed'] for v in out.values()))
    out['a2_distance_known_answers'] = dict(note='descriptive, written into PLAN.md before running; only finiteness is checked',
                                            all_finite=bool(all(x['passed'] for x in res[4])), cases=res[4])
    out['added_after_failure'] = dict(
        note='written after the planar known answer failed; not part of the preset pass criteria. Registered reading '
             '(PLAN.md deviation 2026-11-09): mirror-aware; EM is a parity variant',
        planar_mirror_aware=dict(passed=bool(all(x['mirror_aware']['passed'] for x in planar)),
                                 by_rule={r: bool(all(x['mirror_aware']['passed'] for x in planar if x['rule'] == r)) for r in ('em', 'mackay')}),
        lifted_array=dict(passed=bool(all(x['passed'] for x in lifted)),
                          by_rule={r: bool(all(x['passed'] for x in lifted if x['rule'] == r)) for r in ('em', 'mackay')}, cases=lifted))
    (HERE / 'checks.json').write_text(json.dumps(out, indent=1, default=float))
    print('all passed' if out['all_passed'] else 'FAILED', {k: v['passed'] for k, v in out.items() if isinstance(v, dict) and 'passed' in v},
          {k: v.get('by_rule') for k, v in out['added_after_failure'].items() if isinstance(v, dict)})


if __name__ == '__main__':
    main()
