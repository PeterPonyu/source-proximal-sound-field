"""Runs every registered estimator on one problem and returns per-method metrics (shared by both arms)."""
from __future__ import annotations
import time
import numpy as np
import estimators as E

DEPLOYABLE = ('sinc_ev', 'sinc_loo', 'rbf', 'adk', 'sbl_mackay', 'sbl_em', 'mo26')   # microphone-only comparators
PROPOSED = ('prop_bic', 'prop_cv', 'prop_k1')
ABLATIONS = ('prop_k1_menu', 'prop_bic_room')
REFERENCES = ('oracle', 'mo25')                                                      # receive source coordinates
HYBRIDS = ('mo26_greedy',)                    # region kernel started from the proposed greedy localisation


def _outside(S, lo, hi):
    S = np.atleast_2d(S)
    return bool(np.any(S < lo) or np.any(S > hi)) if len(S) else False


def evaluate(prob, truth, sources, seed, room=None, kmax=3, n_folds=5, skip=(), extra=None):
    """All estimators on identical microphones; true `sources` reach only the oracle and the prior-region reference."""
    out, timing = {}, {}

    def run(name, fn):
        if name in skip:
            return None
        t0 = time.process_time()
        r = fn()
        timing[name] = time.process_time() - t0
        return r
    models = run('prop', lambda: E.fit_sources(prob, kmax))
    kb, bic = E.bic_order(models, prob.M)
    out['prop_k1'] = E.nmse(E.model_prediction(prob, models[1]), truth)
    out['prop_bic'] = E.nmse(E.model_prediction(prob, models[kb]), truth)
    for K, m in models.items():
        out[f'prop_K{K}'] = E.nmse(E.model_prediction(prob, m), truth)
    folds = E.make_folds(prob.M, n_folds, seed)
    cv = run('prop_cv', lambda: E.cv_order(prob, kmax, folds))
    kc, cverr = cv if cv is not None else (kb, None)
    out['prop_cv'] = E.nmse(E.model_prediction(prob, models[kc]), truth)
    info = dict(K_bic=int(kb), K_cv=int(kc), cv_error=None if cverr is None else [float(v) for v in cverr],
                bic=[float(bic[K]) for K in sorted(bic)], evidence=[float(models[K]['evidence']) for K in sorted(models)],
                k1_theta=models[1]['S'][0].tolist(), k1_weight=float(models[1]['w'][0]), k1_rho=float(models[1]['rho']),
                k1_converged=bool(models[1]['converged']))
    rlo, rhi = prob.box
    info['k1_outside_region'] = _outside(models[1]['S'], rlo, rhi)
    if room is not None:
        info['k1_outside_room'] = _outside(models[1]['S'], np.zeros(3), np.asarray(room))
    info['k1_loc_error_m'] = float(np.min(np.linalg.norm(np.atleast_2d(sources) - models[1]['S'][0], axis=1)))
    info['sel_theta'] = models[kb]['S'].tolist()
    info['sel_outside_region'] = _outside(models[kb]['S'], rlo, rhi)
    m = run('prop_k1_menu', lambda: E.fit_sources(prob, 1, fixed_menu=True))
    if m is not None:
        out['prop_k1_menu'] = E.nmse(E.model_prediction(prob, m[1]), truth)
    if room is not None:
        info['sel_outside_room'] = _outside(models[kb]['S'], np.zeros(3), np.asarray(room))
        m = run('prop_bic_room', lambda: E.fit_sources(prob, kmax, inside_room=True))
        if m is not None:
            kr, _ = E.bic_order(m, prob.M)
            out['prop_bic_room'] = E.nmse(E.model_prediction(prob, m[kr]), truth)
            out['prop_k1_room'] = E.nmse(E.model_prediction(prob, m[1]), truth)
            info['K_bic_room'] = int(kr)
    for name, fn in (('sinc_ev', lambda: E.sinc_fit(prob, 'evidence')), ('sinc_loo', lambda: E.sinc_fit(prob, 'loo')),
                     ('rbf', lambda: E.rbf_fit(prob)), ('adk', lambda: E.adk_fit(prob)),
                     ('sbl_mackay', lambda: E.sbl_fit(prob, 'mackay')), ('sbl_em', lambda: E.sbl_fit(prob, 'em')),
                     ('oracle', lambda: E.oracle_fit(prob, sources)), ('mo25', lambda: E.mo25_fit(prob, sources)),
                     ('mo26', lambda: E.mo26_fit(prob, init='scan')),
                     ('mo26_greedy', lambda: E.mo26_fit(prob, init='greedy', init_models=models))):
        r = run(name, fn)
        if r is not None:
            out[name] = E.nmse(r['pred'], truth)
    if extra:
        for name, fn in extra.items():
            r = run(name, lambda fn=fn: fn(prob, sources))
            if r is not None:
                out[name] = E.nmse(r['pred'], truth)
    dep = [k for k in DEPLOYABLE if k in out]
    info['best_comparator'] = min(dep, key=lambda k: out[k]) if dep else None
    return out, info, timing
