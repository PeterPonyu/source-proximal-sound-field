"""Version-2 development run (V2-DESIGN.md): every piece, its ablation and the comparators on development data.

OPENBLAS_NUM_THREADS=1 nice -n 10 python3 -B run_dev.py OUT_DIR [--workers 21] [--only sim|meas] [--limit N]
Development numbers only; never paper numbers.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common'))
import estimators as E  # noqa: E402
import mprir as P  # noqa: E402
import sim  # noqa: E402
import v2  # noqa: E402

SIM_CELLS = ('base', 'A1-in', 'A1-beside0.6', 'A2-f125-M24', 'A2-f125-M192', 'A2-f250-M48', 'A2-f500-M192', 'A2-f1000-M24',
             'A2-f1000-M192', 'A4-beta0.45', 'A5-delta0.4', 'A6-second1.0', 'A9-snr10', 'A9-snr40', 'FF-06')
SIM_SEEDS = range(400920, 400936)
MEAS_SEEDS = range(400936, 400960)          # one microphone draw per (loudspeaker, count, draw), shared by the bands
BAND_OF = {125: 125.0, 250: 250.0, 500: 500.0, 1000: 1000.0}
DEV = {}


def band_freqs(fc, n=6):
    return fc * 2.0 ** ((np.arange(n) + 0.5) / n - 0.5)


def sim_band(cell_name, seed):
    cell = dict(sim.cells()[cell_name])
    fc = float(cell['f'])
    beta = sim.BETA_REALISTIC[fc] if cell['beta'] == 'realistic' else float(cell['beta'])
    Y, T, fd0 = [], [], None
    for f in band_freqs(fc):
        c2 = dict(cell, f=float(f), beta=beta)
        fd = sim.build(c2, seed)
        if fd0 is None:
            fd0 = fd
        assert np.allclose(fd['x'], fd0['x']) and np.allclose(fd['q'], fd0['q'])
        Y.append(fd['y']); T.append(fd['truth'])
    band = v2.Band(fd0['x'], np.stack(Y, 1), fd0['q'], band_freqs(fc), c=fd0['c_assumed'], box=fd0['box'])
    return band, np.stack(T, 1), fd0['sources'], fd0['primary'], fd0['room']


def two_stage_centre(x, t, box):
    """Microphone-only localisation from arrival times at the training microphones (multi-start robust fit)."""
    lo, hi = box
    c = (lo + hi) / 2
    starts = [c] + [c + d for d in (np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]),
                                     np.array([0, 1.0, 0]), np.array([0, -1.0, 0]), np.array([0, 0, 1.0]))]
    best = None
    for s0 in starts:
        r = P.fit_centre(x, t, s0)
        if 300 < r['c'] < 380 and (best is None or r['residual_rms_s'] < best['residual_rms_s']):
            best = r
    return best


def meas_band(spk, fc, M, draw_seed):
    z, meta = DEV['z'], DEV['meta'][spk]
    pos, H = z[f'{spk}_positions'], z[f'{spk}_H']
    sel = np.where(z['band'] == fc)[0]
    rng = np.random.default_rng(np.random.SeedSequence([draw_seed, 9203]))
    mic = np.sort(rng.choice(len(pos), M, replace=False)); tgt = np.setdiff1d(np.arange(len(pos)), mic)
    ts = two_stage_centre(pos[mic], z[f'{spk}_arrival_s'][mic], (np.asarray(meta['box_lo']), np.asarray(meta['box_hi'])))
    c = ts['c'] if ts else 343.0
    Y = H[mic][:, sel]; T = H[tgt][:, sel]
    sc = np.sqrt(np.mean(np.abs(Y) ** 2, axis=0))
    band = v2.Band(pos[mic], Y / sc, pos[tgt], z['freqs'][sel], c=c, box=(np.asarray(meta['box_lo']), np.asarray(meta['box_hi'])))
    return band, T / sc, np.asarray(meta['acoustic_centre_all_mics'])[None, :], ts


def per_bin(band, T, pred_fn):
    return [E.nmse(pred_fn(f), T[:, f]) for f in range(band.F)]


def run_task(task):
    t0 = time.process_time()
    try:
        timing = {}
        out, info = {}, {}

        def tm(name, fn):
            a = time.process_time(); r = fn(); timing[name] = round(time.process_time() - a, 2); return r
        if task[0] == 'sim':
            _, cell, seed = task
            band, T, sources, primary, room = sim_band(cell, seed)
            ref_sources = sources; loc_ref = primary
            info.update(kind='sim', cell=cell, seed=seed)
        else:
            _, spk, fc, M, draw, seed = task
            band, T, centre_all, ts = meas_band(spk, fc, M, seed)
            ref_sources = centre_all; loc_ref = centre_all[0]
            info.update(kind='meas', source=spk, band=fc, M=M, draw=draw, seed=seed, c_train=band.c,
                        two_stage_ok=bool(ts), two_stage_centre=None if not ts else ts['centre'].tolist())
        info.update(F=band.F, M_mics=band.M)
        # version 1 per bin (registered pipeline, cap 3, Schwarz)
        v1 = tm('v1', lambda: [E.fit_sources(p, 3) for p in band.probs])
        kb = [E.bic_order(m, p.M)[0] for m, p in zip(v1, band.probs)]
        out['v1'] = [E.nmse(E.model_prediction(p, m[k]), T[:, f]) for f, (p, m, k) in enumerate(zip(band.probs, v1, kb))]
        out['v1_k1'] = [E.nmse(E.model_prediction(p, m[1]), T[:, f]) for f, (p, m) in enumerate(zip(band.probs, v1))]
        info['v1_K'] = kb
        info['v1_k1_loc_m'] = [float(np.linalg.norm(m[1]['S'][0] - loc_ref)) for m in v1]
        out['hybrid_v1'] = tm('hybrid_v1', lambda: [E.nmse(E.mo26_fit(p, init='greedy', init_models=m)['pred'], T[:, f])
                                                     for f, (p, m) in enumerate(zip(band.probs, v1))])
        # (c) ablation: each bin alone, version-2 code, cap 5
        ind = tm('ind', lambda: v2.fit_independent(band, 5))
        ks = [v2.schwarz_order(m, b1)[0] for b1, m in ind]
        out['ind_sw'] = [E.nmse(v2.predict_points(b1, m[k], 0), T[:, f]) for f, ((b1, m), k) in enumerate(zip(ind, ks))]
        info['ind_k1_loc_m'] = [float(np.linalg.norm(m[1]['S'][0] - loc_ref)) for _, m in ind]
        # joint over the bins (c), cap 5, selectors (b), extension (a)
        def joint_block(bd, tag):
            models = tm(f'{tag}_fit', lambda: v2.fit_band(bd, 5))
            perK = {K: per_bin(bd, T, lambda f, m=m: v2.predict_points(bd, m, f)) for K, m in models.items()}
            for K, v in perK.items():
                out[f'{tag}_K{K}'] = v
            ksw, _ = v2.schwarz_order(models, bd)
            kla, _ = tm(f'{tag}_laplace', lambda: v2.laplace_order(models, bd))
            ksw3, _ = v2.schwarz_order({K: m for K, m in models.items() if K <= 3}, bd)
            folds = E.make_folds(bd.M, 5, task[-1])
            kcv, _ = tm(f'{tag}_cv', lambda: v2.cv_order(bd, 5, folds))
            info[f'{tag}_K'] = dict(sw=ksw, la=kla, sw3=ksw3, cv=kcv)
            info[f'{tag}_k1_loc_m'] = float(np.linalg.norm(models[1]['S'][0] - loc_ref))
            for sname, k in (('sw', ksw), ('la', kla), ('cv', kcv), ('sw3', ksw3)):
                out[f'{tag}_{sname}'] = perK[k]
            for sname, k in (('sw', ksw), ('la', kla), ('cv', kcv)):
                if k == 0:
                    out[f'{tag}_{sname}_ext'] = perK[0]
                    continue
                key = (k,)
                if key not in info.setdefault(f'_{tag}_ext_cache', {}):
                    ext = tm(f'{tag}_ext_K{k}', lambda: v2.extend(bd, models[k], k))
                    info[f'_{tag}_ext_cache'][key] = per_bin(bd, T, lambda f: v2.predict_regions(bd, ext, f))
                out[f'{tag}_{sname}_ext'] = info[f'_{tag}_ext_cache'][key]
            info.pop(f'_{tag}_ext_cache', None)
            return models
        joint_block(band, 'j')
        # (d) remainder family chosen by evidence, then the same joint block
        if task[0] == 'sim':
            bandf, _, _, _, _ = sim_band(task[1], task[2])
        else:
            bandf, _, _, _ = meas_band(task[1], task[2], task[3], task[-1])
        fam = tm('family', lambda: v2.choose_family(bandf))
        info['family'] = fam; info['family_scores'] = bandf.family_info['scores']
        out['fam_K0'] = per_bin(bandf, T, lambda f: v2.predict_points(bandf, dict(S=np.zeros((0, 3)), W=np.zeros((bandf.F, 0)),
                                                                               R=np.array([v2.remainder_rho(p) for p in bandf.probs])), f))
        if fam == 'sinc':
            for k in [k for k in out if k.startswith('j_')]:
                out['f' + k] = out[k]
            info['fj_K'] = info['j_K']
        else:
            joint_block(bandf, 'fj')
        # comparators (one search domain, one cap, version-2 bounds)
        out['sinc_ev'] = per_bin(band, T, lambda f: E.sinc_fit(band.probs[f], 'evidence')['pred'])
        out['rbf'] = tm('rbf', lambda: per_bin(band, T, lambda f: E.rbf_fit(band.probs[f])['pred']))
        out['adk_standin'] = tm('adk', lambda: per_bin(band, T, lambda f: E.adk_fit(band.probs[f])['pred']))
        out['sbl_em_common'] = tm('sbl_em', lambda: per_bin(band, T, lambda f: v2.sbl_common(band.probs[f], 'em')['pred']))
        out['sbl_mackay_common'] = tm('sbl_mackay', lambda: per_bin(band, T, lambda f: v2.sbl_common(band.probs[f], 'mackay')['pred']))
        out['mo26_cap5'] = tm('mo26_cap5', lambda: per_bin(band, T, lambda f: v2.mo26_cap(band.probs[f])['pred']))
        gs = tm('gsmf', lambda: v2.gsmf_fit(band))
        out['gsmf'] = per_bin(band, T, lambda f: v2.predict_regions(band, gs, f)); info['gsmf_regions'] = len(gs['radii']) - gs['pruned']
        # references
        out['ref_point'] = per_bin(band, T, lambda f: E.oracle_fit(band.probs[f], ref_sources)['pred'])
        out['ref_region'] = per_bin(band, T, lambda f: E.mo25_fit(band.probs[f], ref_sources)['pred'])
        if task[0] == 'meas' and ts:
            c2 = np.asarray(ts['centre'])[None, :]
            out['two_stage_point'] = per_bin(band, T, lambda f: E.oracle_fit(band.probs[f], c2)['pred'])
            out['two_stage_region'] = per_bin(band, T, lambda f: E.mo25_fit(band.probs[f], c2)['pred'])
            info['two_stage_loc_m'] = float(np.linalg.norm(ts['centre'] - loc_ref))
        row = dict(info, nmse=out, timing=timing, process_seconds=round(time.process_time() - t0, 1))
        return row
    except Exception:
        return dict(task=list(task), error=traceback.format_exc())


def init():
    z = np.load(HERE / 'prep/measured-dev.npz', allow_pickle=False)
    DEV['z'] = {k: z[k] for k in z.files}
    DEV['meta'] = json.loads((HERE / 'prep/measured-dev.json').read_text())


def tasks(only=None):
    T = []
    if only in (None, 'sim'):
        T += [('sim', c, s) for c in SIM_CELLS for s in SIM_SEEDS]
    if only in (None, 'meas'):
        i = 0
        for spk in ('S4', 'S7', 'S8'):
            for M in (24, 48, 96, 192):
                for d in range(2):
                    seed = MEAS_SEEDS[i]; i += 1
                    T += [('meas', spk, fc, M, d, seed) for fc in (125.0, 250.0, 500.0, 1000.0)]
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--workers', type=int, default=21)
    ap.add_argument('--only', default=None); ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    T = tasks(a.only)
    T = sorted(T, key=lambda t: -(t[3] if t[0] == 'meas' else sim.cells()[t[1]]['M']))   # big problems first
    if a.limit:
        T = T[::max(1, len(T) // a.limit)][:a.limit]
    t0 = time.time()
    with Pool(a.workers, initializer=init) as pool, open(out / 'rows.jsonl', 'a') as fh:
        for n, row in enumerate(pool.imap_unordered(run_task, T), 1):
            fh.write(json.dumps(row, default=float) + '\n'); fh.flush()
            if n % 20 == 0 or n == len(T):
                print(f'{n}/{len(T)} tasks, {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
