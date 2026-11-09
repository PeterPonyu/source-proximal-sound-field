"""Registered version-2 method set on one band problem (TB-S2, TB-M2).

Primary `j_cv_ext`: coordinates shared across the bins, order by five-fold microphone cross-validation (cap 5), then
the kept sources extended to ball regions (region kernel of Matsuda & Otani), sinc remainder.
Ablations: `j_cv` (points, no extension), `j_sw_ext` (Schwarz order), `ind_cv_ext` (each bin alone, same order rule and
extension); descriptive: per-order errors `j_K0`..`j_K5`, `j_sw`, `j_la`, `j_sw3`, `ind_sw`.
Secondary (measured only): `fj_cv_ext`, `fj_cv` with the remainder family chosen by evidence; `fam_K0`.
Comparators: `sinc_ev`, `rbf`, `adk_standin`, `sbl_em_common`, `sbl_mackay_common`, `mo26_cap5`, `gsmf`, `hybrid_v1`,
`v1`. References that receive source coordinates: `ref_point`, `ref_region`; measured microphone-only two-stage
baseline: `two_stage_point`, `two_stage_region`.
"""
from __future__ import annotations
import time
import numpy as np
import estimators as E
import v2


def per_bin(band, T, fn):
    return [E.nmse(fn(f), T[:, f]) for f in range(band.F)]


def _joint(band, T, tag, seed, out, info, tm, selectors=('sw', 'la', 'cv', 'sw3')):
    models = tm(f'{tag}_fit', lambda: v2.fit_band(band, 5))
    for K, m in models.items():
        out[f'{tag}_K{K}'] = per_bin(band, T, lambda f, m=m: v2.predict_points(band, m, f))
    ks = {}
    ks['sw'] = v2.schwarz_order(models, band)[0]
    ks['sw3'] = v2.schwarz_order({K: m for K, m in models.items() if K <= 3}, band)[0]
    if 'la' in selectors:
        ks['la'] = tm(f'{tag}_laplace', lambda: v2.laplace_order(models, band))[0]
    folds = E.make_folds(band.M, 5, seed)
    ks['cv'] = tm(f'{tag}_cv', lambda: v2.cv_order(band, 5, folds))[0]
    info[f'{tag}_K'] = ks
    info[f'{tag}_k1_S'] = models[1]['S'][0].tolist()
    for s, k in ks.items():
        out[f'{tag}_{s}'] = out[f'{tag}_K{k}']
    ext = {}
    for s in ('cv', 'sw'):
        k = ks[s]
        if k == 0:
            out[f'{tag}_{s}_ext'] = out[f'{tag}_K0']; continue
        if k not in ext:
            rm = tm(f'{tag}_ext_K{k}', lambda: v2.extend(band, models[k], k))
            ext[k] = per_bin(band, T, lambda f: v2.predict_regions(band, rm, f))
        out[f'{tag}_{s}_ext'] = ext[k]
    return models


def evaluate(band, T, ref_sources, loc_ref, seed, measured=False, two_stage=None, band_factory=None):
    out, info, timing = {}, {}, {}

    def tm(name, fn):
        a = time.process_time(); r = fn(); timing[name] = round(time.process_time() - a, 2); return r
    # version 1 per bin (registered pipeline, cap 3, Schwarz) and its hybrid
    v1 = tm('v1', lambda: [E.fit_sources(p, 3) for p in band.probs])
    kb = [E.bic_order(m, p.M)[0] for m, p in zip(v1, band.probs)]
    out['v1'] = [E.nmse(E.model_prediction(p, m[k]), T[:, f]) for f, (p, m, k) in enumerate(zip(band.probs, v1, kb))]
    info['v1_K'] = kb
    out['hybrid_v1'] = tm('hybrid_v1', lambda: [E.nmse(E.mo26_fit(p, init='greedy', init_models=m)['pred'], T[:, f])
                                                 for f, (p, m) in enumerate(zip(band.probs, v1))])
    # primary and joint ablations
    jm = _joint(band, T, 'j', seed, out, info, tm)
    info['j_k1_loc_m'] = float(np.linalg.norm(jm[1]['S'][0] - loc_ref))
    # shared-coordinate ablation: each bin alone with the same order rule (cross-validation) and extension
    ind = tm('ind_fit', lambda: v2.fit_independent(band, 5))
    ind_sw, ind_cv_ext, kcv_bins = [], [], []
    for f, (b1, m) in enumerate(ind):
        ksw = v2.schwarz_order(m, b1)[0]
        ind_sw.append(E.nmse(v2.predict_points(b1, m[ksw], 0), T[:, f]))
        kcv = tm(f'ind_cv_{f}', lambda: v2.cv_order(b1, 5, E.make_folds(b1.M, 5, seed)))[0]
        kcv_bins.append(kcv)
        if kcv == 0:
            ind_cv_ext.append(E.nmse(v2.predict_points(b1, m[0], 0), T[:, f]))
        else:
            rm = tm(f'ind_ext_{f}', lambda: v2.extend(b1, m[kcv], kcv))
            ind_cv_ext.append(E.nmse(v2.predict_regions(b1, rm, 0), T[:, f]))
    out['ind_sw'], out['ind_cv_ext'] = ind_sw, ind_cv_ext
    info['ind_K_cv'] = kcv_bins
    info['ind_k1_loc_m'] = [float(np.linalg.norm(m[1]['S'][0] - loc_ref)) for _, m in ind]
    # secondary, measured only: remainder family chosen by evidence
    if measured and band_factory is not None:
        bf = band_factory()
        fam = tm('family', lambda: v2.choose_family(bf))
        info['family'] = fam; info['family_scores'] = bf.family_info['scores']
        out['fam_K0'] = per_bin(bf, T, lambda f: v2.predict_points(bf, dict(S=np.zeros((0, 3)), W=np.zeros((bf.F, 0)),
                                                                          R=np.array([v2.remainder_rho(p) for p in bf.probs])), f))
        if fam == 'sinc':
            for k in ('cv', 'cv_ext'):
                out[f'fj_{k}'] = out[f'j_{k}']
            info['fj_K'] = info['j_K']
        else:
            _joint(bf, T, 'fj', seed, out, info, tm, selectors=('sw', 'cv', 'sw3'))
            for k in [k for k in out if k.startswith('fj_') and k not in ('fj_cv', 'fj_cv_ext')]:
                out.pop(k)
    # comparators: one search domain, one cap, version-2 bounds where they apply
    out['sinc_ev'] = per_bin(band, T, lambda f: E.sinc_fit(band.probs[f], 'evidence')['pred'])
    out['rbf'] = tm('rbf', lambda: per_bin(band, T, lambda f: E.rbf_fit(band.probs[f])['pred']))
    out['adk_standin'] = tm('adk', lambda: per_bin(band, T, lambda f: E.adk_fit(band.probs[f])['pred']))
    out['sbl_em_common'] = tm('sbl_em', lambda: per_bin(band, T, lambda f: v2.sbl_common(band.probs[f], 'em')['pred']))
    out['sbl_mackay_common'] = tm('sbl_mackay', lambda: per_bin(band, T, lambda f: v2.sbl_common(band.probs[f], 'mackay')['pred']))
    out['mo26_cap5'] = tm('mo26_cap5', lambda: per_bin(band, T, lambda f: v2.mo26_cap(band.probs[f])['pred']))
    gs = tm('gsmf', lambda: v2.gsmf_fit(band))
    out['gsmf'] = per_bin(band, T, lambda f: v2.predict_regions(band, gs, f))
    info['gsmf_regions'] = int(len(gs['radii']) - gs['pruned'])
    # references (receive source coordinates) and the measured microphone-only two-stage baseline
    out['ref_point'] = per_bin(band, T, lambda f: E.oracle_fit(band.probs[f], ref_sources)['pred'])
    out['ref_region'] = per_bin(band, T, lambda f: E.mo25_fit(band.probs[f], ref_sources)['pred'])
    if two_stage:
        c2 = np.asarray(two_stage['centre'])[None, :]
        out['two_stage_point'] = per_bin(band, T, lambda f: E.oracle_fit(band.probs[f], c2)['pred'])
        out['two_stage_region'] = per_bin(band, T, lambda f: E.mo25_fit(band.probs[f], c2)['pred'])
        info['two_stage_loc_m'] = float(np.linalg.norm(np.asarray(two_stage['centre']) - loc_ref))
        info['two_stage_c'] = float(two_stage['c'])
    return out, info, timing
