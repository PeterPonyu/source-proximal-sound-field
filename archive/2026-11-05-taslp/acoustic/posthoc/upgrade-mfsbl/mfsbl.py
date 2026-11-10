"""Multi-frequency sparse Bayesian learning with a support shared across the bins of a band (PLAN.md).

One dictionary for the band: monopoles on the band's search box at a quarter of the mid-band wavelength (at most
v2.SBL_NODES), guarded 5 cm from microphones and targets as in common/sbl_guard.py, plus plane waves. Each bin's matrix
is built at its own wavenumber. One gamma vector is shared by the bins; each bin has its own noise variance; the EM and
MacKay updates of estimators.sbl_fit are pooled over the bins, with the same initialisation, pruning and stopping
constants. Each bin is scaled internally by the root of its mean microphone power. With one bin it is sbl_fit with the
guarded dictionary.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist

S = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(S / 'common'))
import estimators as E  # noqa: E402
import v2  # noqa: E402
from fields import fibonacci_sphere  # noqa: E402

KEYS = {'em': 'mfsbl_em', 'mackay': 'mfsbl_mackay'}


def dictionary(band, stats=None):
    """Monopole nodes and plane-wave directions shared by every bin of the band."""
    lo, hi = band.search_box()
    # mid-band wavelength as the geometric mean of the extreme bins' wavelengths: equal to Band.lam_mid() for log-spaced
    # bins, and bit-identical to the bin's own wavelength when F = 1 (grid_nodes rounds node counts up, so one ulp matters)
    lam = [p.lam for p in band.probs]
    nodes = E.grid_nodes(lo, hi, 0.25 * float(np.sqrt(min(lam) * max(lam))), v2.SBL_NODES)
    keep_mic = cdist(band.x, nodes).min(axis=0) > E.GUARD
    keep_tgt = cdist(band.q, nodes).min(axis=0) > E.GUARD
    if stats is not None:
        stats.update(nodes=int(len(nodes)), removed_by_mics=int((~keep_mic).sum()),
                     removed_by_targets_only=int((keep_mic & ~keep_tgt).sum()))
    nodes = nodes[keep_mic & keep_tgt]
    R = 0.5 * float(np.linalg.norm(band.box[1] - band.box[0]))
    k = max(p.k for p in band.probs)
    N = int(np.clip(np.ceil(k * R) + 1, 3, 24))
    return nodes, fibonacci_sphere(2 * (N + 1) ** 2)


def fit(band, rule='mackay', max_iter=None, tol=1e-4):
    """Returns per-bin target predictions (in the band's units) and the shared support."""
    if max_iter is None:
        max_iter = 1000 if rule == 'mackay' else 200
    stats = {}
    nodes, dirs = dictionary(band, stats)
    F, M = band.F, band.M
    Phi, nu, scale, Y = [], [], [], []
    for p in band.probs:
        g = E.green(p.x, nodes, p.k)
        n = np.sqrt(np.mean(np.abs(g) ** 2, axis=0))
        Phi.append(np.hstack((g / n[None, :], np.exp(1j * p.k * p.x @ dirs.T))))
        nu.append(n)
        s = float(np.sqrt(np.vdot(p.y, p.y).real / M))
        s = s if s > 0 else 1.0
        scale.append(s)
        Y.append(p.y / s)
    N = Phi[0].shape[1]
    yy = np.array([float(np.vdot(y, y).real) / M for y in Y])
    gamma = np.full(N, float(np.mean(yy)) / N * 10)
    s2 = 0.01 * yy
    active = np.arange(N)
    it, change = 0, np.inf
    for it in range(max_iter):
        ga = gamma[active]
        num = np.zeros(len(active))
        den = np.zeros(len(active))
        for f in range(F):
            P = Phi[f][:, active]
            Sy = s2[f] * np.eye(M) + (P * ga[None, :]) @ P.conj().T
            fac = cho_factor(Sy, lower=True, check_finite=False)
            iy = cho_solve(fac, Y[f])
            mu = ga * (P.conj().T @ iy)
            SP = cho_solve(fac, P)
            diagS = ga - ga ** 2 * np.real(np.sum(P.conj() * SP, axis=0))
            resid = Y[f] - P @ mu
            r2 = float(np.vdot(resid, resid).real)
            used = float(np.sum(1 - diagS / ga))
            if rule == 'em':
                num += np.abs(mu) ** 2 + diagS
                s2[f] = max((r2 + s2[f] * used) / M, 1e-12 * yy[f])
            else:
                num += np.abs(mu) ** 2
                den += 1 - diagS / ga
                s2[f] = max(r2 / max(M - used, 1.0), 1e-12 * yy[f])
        new = num / F if rule == 'em' else num / np.maximum(den, 1e-12)
        change = np.max(np.abs(new - ga)) / max(np.max(ga), 1e-300)
        gamma[active] = new
        active = active[gamma[active] > 1e-8 * gamma[active].max()]
        if change < tol:
            break
    nn = len(nodes)
    mono = active[active < nn]
    pw = active[active >= nn] - nn
    preds = []
    for f, p in enumerate(band.probs):
        P = Phi[f][:, active]
        Sy = s2[f] * np.eye(M) + (P * gamma[active][None, :]) @ P.conj().T
        mu = gamma[active] * (P.conj().T @ np.linalg.solve(Sy, Y[f]))
        pred = np.zeros(len(p.q), complex)
        if len(mono):
            pred += (E.green(p.q, nodes[mono], p.k) / nu[f][mono][None, :]) @ mu[:len(mono)]
        if len(pw):
            pred += np.exp(1j * p.k * p.q @ dirs[pw].T) @ mu[len(mono):]
        preds.append(pred * scale[f])
    g_out = np.zeros(N)
    g_out[active] = gamma[active]
    return dict(method=KEYS[rule], preds=preds, iterations=it + 1, stopped_by_tolerance=bool(change < tol), active=int(len(active)),
                active_monopoles=int(len(mono)), noise=[float(v * s ** 2) for v, s in zip(s2, scale)], noise_scaled=s2.tolist(),
                gamma=g_out, nodes=nodes, dirs=dirs, dictionary=int(N), node_stats=stats)


def add_methods(band, T, out, info, timing):
    """Adds mfsbl_em and mfsbl_mackay to the dicts of methods_v2.evaluate (per-bin NMSE, info, process seconds)."""
    for rule, key in KEYS.items():
        a = time.process_time()
        r = fit(band, rule)
        timing[key] = round(time.process_time() - a, 2)
        out[key] = [E.nmse(r['preds'][f], T[:, f]) for f in range(band.F)]
        info[key] = dict(iterations=r['iterations'], stopped_by_tolerance=r['stopped_by_tolerance'], active=r['active'],
                         active_monopoles=r['active_monopoles'], dictionary=r['dictionary'], noise=r['noise'],
                         gamma_finite_nonneg=bool(np.all(np.isfinite(r['gamma'])) and np.all(r['gamma'] >= 0)), **r['node_stats'])
    return out, info, timing
