"""Version-2 estimators (development; V2-DESIGN.md). Builds on the frozen version-1 module `estimators` without
changing it.

Pieces: (a) extension of the selected sources to ball regions (region kernel of Matsuda & Otani), (b) order selection
by the Schwarz penalty, a Laplace evidence or microphone cross-validation, with a cap of 5, (c) source coordinates
shared across the frequency bins of a band (Matsuda & Otani, ICASSP 2026, share the source layout across frequency),
with closed-form scan weights, (d) a remainder kernel family (sinc, Gaussian, sinc plus Gaussian) chosen by evidence.
Comparators with parity: region kernel (cap 5), group-sparse multi-frequency region kernel (reimplementation from the
ICASSP 2026 abstract), sparse Bayesian learning over the common search domain.
"""
from __future__ import annotations
import contextlib
import math
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import estimators as E

KMAX = 5
LOGW2 = E.LOGW        # version-1 ranges kept: wider ranges (weights to 1e12, ridge to 1e4) made the covariance
LOGRHO2 = E.LOGRHO    # ill-conditioned and stalled the optimiser in development checks
LOGRAD = (np.log(0.005), np.log(0.5))
SCAN_NODES = 12000
SBL_NODES = 6000


# ------------------------------------------------------------------ band problems and remainder families

def gauss_k(a, b, ell):
    return np.exp(-0.5 * cdist(a, b) ** 2 / ell ** 2)


class Band:
    """F frequency bins observed with the same microphones and targets; one version-1 Problem per bin."""

    def __init__(self, x, Y, q, freqs, c=343.0, box=None):
        Y = np.asarray(Y, complex)
        self.freqs = np.asarray(freqs, float)
        self.probs = [E.Problem(x, Y[:, i], q, f, c, box) for i, f in enumerate(self.freqs)]
        self.x, self.q, self.c = self.probs[0].x, self.probs[0].q, float(c)
        self.M, self.F = len(self.x), len(self.freqs)
        self.box = self.probs[0].box
        self.family = 'sinc'
        self.family_info = {}

    def lam_max(self):
        return self.c / self.freqs.min()

    def lam_mid(self):
        return self.c / math.exp(np.mean(np.log(self.freqs)))

    def search_box(self):
        """Region box enlarged by max(lambda, 1.5 m) at the lowest bin (the largest box of the band)."""
        m = max(self.lam_max(), E.MARGIN_M)
        return self.box[0] - m, self.box[1] + m

    def subset(self, idx):
        b = Band(self.x[idx], np.stack([p.y[idx] for p in self.probs], 1), self.q, self.freqs, self.c, self.box)
        if self.family != 'sinc':
            for pb, p in zip(b.probs, self.probs):
                set_remainder(pb, self.family, p._v2_hyper)
            b.family, b.family_info = self.family, self.family_info
        return b


def _evidence(H, y, rho):
    M = len(y)
    try:
        fac = cho_factor(H + (rho + E.JITTER) * np.eye(M), lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return -1e30
    a = cho_solve(fac, y)
    return -M * np.log(float(np.vdot(y, a).real)) - 2 * float(np.sum(np.log(np.diag(fac[0]).real)))


def _family_matrices(p, family, h):
    S = E.sinc_k(p.k, p.x, p.x); Sq = E.sinc_k(p.k, p.q, p.x)
    if family == 'sinc':
        return S, Sq
    G = gauss_k(p.x, p.x, h['ell']); Gq = gauss_k(p.q, p.x, h['ell'])
    if family == 'gauss':
        return G, Gq
    return S + h['beta'] * G, Sq + h['beta'] * Gq


def set_remainder(p, family, h):
    p.H, p.Hq = _family_matrices(p, family, h)
    p._eig = None
    p._v2_hyper = dict(h)


def family_evidence(p, family):
    """Source-free evidence of one bin for a remainder family, hyperparameters by multi-start L-BFGS-B."""
    if family == 'sinc':
        r = E.sinc_fit(p, 'evidence')
        return r['evidence'], dict(rho=r['rho'])
    best = None
    if family == 'gauss':
        def obj(t):
            return -_evidence(gauss_k(p.x, p.x, np.exp(t[0])), p.y, np.exp(t[1]))
        bounds = [(np.log(0.01 * p.lam), np.log(5 * p.lam)), E.LOGRHO]
        starts = [[np.log(e * p.lam), np.log(r)] for e in (0.1, 0.3) for r in (1e-4, 1e-2)]
    else:
        S = E.sinc_k(p.k, p.x, p.x)

        def obj(t):
            return -_evidence(S + np.exp(t[2]) * gauss_k(p.x, p.x, np.exp(t[0])), p.y, np.exp(t[1]))
        bounds = [(np.log(0.01 * p.lam), np.log(5 * p.lam)), E.LOGRHO, (np.log(1e-3), np.log(1e3))]
        starts = [[np.log(e * p.lam), np.log(r), np.log(b)] for e in (0.1, 0.3) for r in (1e-3,) for b in (0.1, 1.0)]
    for s in starts:
        r = minimize(obj, s, method='L-BFGS-B', bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    t = best.x
    h = dict(ell=float(np.exp(t[0])), rho=float(np.exp(t[1])))
    if family == 'sum':
        h['beta'] = float(np.exp(t[2]))
    return -float(best.fun), h


EXTRA_HYPER = dict(sinc=0, gauss=1, sum=2)


def choose_family(band, families=('sinc', 'gauss', 'sum')):
    """Evidence of each family summed over the bins, charging 0.5 ln(2M) per extra hyperparameter and bin."""
    scores, hyp = {}, {}
    for fam in families:
        evs, hs = zip(*[family_evidence(p, fam) for p in band.probs])
        hyp[fam] = hs
        scores[fam] = float(np.sum(evs)) - 0.5 * EXTRA_HYPER[fam] * band.F * np.log(2 * band.M)
    fam = max(scores, key=scores.get)
    if fam != 'sinc':
        for p, h in zip(band.probs, hyp[fam]):
            set_remainder(p, fam, h)
    band.family, band.family_info = fam, dict(scores=scores)
    return fam


def remainder_rho(p):
    """Ridge of the source-free fit of one bin under its current remainder kernel."""
    h = getattr(p, '_v2_hyper', None)
    return h['rho'] if h and 'rho' in h else E.sinc_fit(p, 'evidence')['rho']


def remainder_evidence(p, rho):
    return _evidence(p.H, p.y, rho)


# ------------------------------------------------------------------ joint point-source model over the bins

def _split(theta, K, F):
    S = theta[:3 * K]
    lw = theta[3 * K:3 * K + K * F].reshape(F, K)
    lr = theta[3 * K + K * F:3 * K + K * F + F]
    return S, lw, lr


def joint_objective(theta, band, K):
    S, lw, lr = _split(theta, K, band.F)
    tot, g = 0.0, np.zeros_like(theta)
    for f, p in enumerate(band.probs):
        v, gr = E.source_objective(np.concatenate((S, lw[f], [lr[f]])), p, K)
        if v >= 1e29:
            return 1e30, np.zeros_like(theta)
        tot += v
        g[:3 * K] += gr[:3 * K]
        g[3 * K + f * K:3 * K + (f + 1) * K] = gr[3 * K:4 * K]
        g[3 * K + K * band.F + f] = gr[4 * K]
    return tot, g


def band_scan(band, nodes, S, W, R, starts, suppress):
    """Sum over bins of the rank-one evidence gain of one more point at each node, with the closed-form optimal
    weight per bin: for u = w b / (1 + w b), the gain -M ln(1 - a u / Q) + ln(1 - u) peaks at u = (M a - Q) / (a (M - 1)),
    a = |g^H alpha|^2 / (g^H C^-1 g)."""
    S = np.atleast_2d(S) if len(S) else np.zeros((0, 3))
    d = cdist(band.x, nodes)
    valid = d.min(axis=0) > E.GUARD
    nodes, d = nodes[valid], d[:, valid]
    total = np.zeros(len(nodes)); Wn = np.zeros((band.F, len(nodes)))
    inv_nu = 1 / np.sqrt(np.mean(1 / d ** 2, axis=0))
    for f, p in enumerate(band.probs):
        if len(S):
            b, _, _ = E.basis_grad(p.x, S, p.k)
            C = p.H + (b * W[f][None, :]) @ b.conj().T + (R[f] + E.JITTER) * np.eye(p.M)
        else:
            C = p.H + (R[f] + E.JITTER) * np.eye(p.M)
        fac = cho_factor(C, lower=True, check_finite=False)
        alpha = cho_solve(fac, p.y)
        Q = float(np.vdot(p.y, alpha).real)
        B = (np.exp(1j * p.k * d) / d) * inv_nu[None, :]
        CB = cho_solve(fac, B)
        beta = np.maximum(np.real(np.sum(B.conj() * CB, axis=0)), 1e-300)
        gam = np.abs(B.conj().T @ alpha) ** 2
        a = gam / beta
        u = np.clip((p.M * a - Q) / (a * (p.M - 1) + 1e-300), 0.0, 1 - 1e-12)
        w = np.clip(u / (beta * (1 - u)), np.exp(LOGW2[0]), np.exp(LOGW2[1]))
        den = 1 + w * beta
        gain = -p.M * np.log(np.maximum(1 - w * gam / (den * Q), 1e-300)) - np.log(den)
        total += gain; Wn[f] = w
    order = np.argsort(-total)
    chosen = []
    for i in order:
        if all(np.linalg.norm(nodes[i] - nodes[j]) > suppress for j in chosen):
            chosen.append(i)
        if len(chosen) == starts:
            break
    return [(nodes[i], Wn[:, i], total[i]) for i in chosen]


def _bounds_points(band, K, lo, hi):
    return ([(lo[c], hi[c]) for _ in range(K) for c in range(3)] + [LOGW2] * (K * band.F) + [LOGRHO2] * band.F)


def fit_band(band, kmax=KMAX, starts=3, maxiter=200):
    """Greedy sequential localisation with coordinates shared across the bins; models for every order 0..kmax."""
    lo, hi = band.search_box()
    nodes = E.grid_nodes(lo, hi, 0.25 * band.lam_mid(), max_nodes=SCAN_NODES)
    R0 = np.array([remainder_rho(p) for p in band.probs])
    ev0 = float(sum(remainder_evidence(p, r) for p, r in zip(band.probs, R0)))
    models = {0: dict(S=np.zeros((0, 3)), W=np.zeros((band.F, 0)), R=R0, evidence=ev0, converged=True)}
    S, W, R = np.zeros((0, 3)), np.zeros((band.F, 0)), R0
    for K in range(1, kmax + 1):
        cands = band_scan(band, nodes, S, W, R, starts, 0.5 * band.lam_mid())
        bounds = _bounds_points(band, K, lo, hi)
        lb, ub = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
        best = None
        for node, wn, _ in cands:
            S0 = np.vstack((S, node[None, :]))
            lw0 = np.log(np.maximum(np.hstack((W, wn[:, None])), 1e-4))
            th0 = np.clip(np.concatenate((S0.ravel(), lw0.ravel(), np.log(R))), lb, ub)
            r = minimize(joint_objective, th0, args=(band, K), jac=True, method='L-BFGS-B', bounds=bounds,
                         options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-6))
            if best is None or r.fun < best.fun:
                best = r
        Sv, lw, lr = _split(best.x, K, band.F)
        S, W, R = Sv.reshape(K, 3), np.exp(lw), np.exp(lr)
        models[K] = dict(S=S.copy(), W=W.copy(), R=R.copy(), evidence=-float(best.fun), converged=bool(best.success),
                         theta=best.x.copy())
    return models


def fit_independent(band, kmax=KMAX, starts=3):
    """Ablation of (c): each bin localised alone by the same version-2 code with one bin per band."""
    out = []
    for f, p in enumerate(band.probs):
        b1 = Band(band.x, p.y[:, None], band.q, band.freqs[f:f + 1], band.c, band.box)
        if band.family != 'sinc':
            set_remainder(b1.probs[0], band.family, p._v2_hyper)
            b1.family = band.family
        out.append((b1, fit_band(b1, kmax, starts=starts)))
    return out


def predict_points(band, m, f, q=None):
    p = band.probs[f]
    if len(m['S']) == 0:
        hq = p.Hq if q is None else _cross(p, q)
        return hq @ np.linalg.solve(p.H + (m['R'][f] + E.JITTER) * np.eye(p.M), p.y)
    if q is None:
        return E.source_predict(p, m['S'], m['W'][f], m['R'][f])
    S = np.atleast_2d(m['S'])
    b, _, nu = E.basis_grad(p.x, S, p.k)
    C = p.H + (b * m['W'][f][None, :]) @ b.conj().T + (m['R'][f] + E.JITTER) * np.eye(p.M)
    alpha = cho_solve(cho_factor(C, lower=True), p.y)
    return _cross(p, q) @ alpha + E.target_basis(q, S, p.k, nu) @ (m['W'][f] * (b.conj().T @ alpha))


def _cross(p, q):
    h = getattr(p, '_v2_hyper', None)
    S = E.sinc_k(p.k, q, p.x)
    if not h or 'ell' not in h:
        return S
    G = gauss_k(q, p.x, h['ell'])
    return G if 'beta' not in h else S + h['beta'] * G


# ------------------------------------------------------------------ order selection (b)

def schwarz_order(models, band):
    F, M = band.F, band.M
    score = {K: m['evidence'] - 0.5 * (3 + F) * K * np.log(2 * M * F) for K, m in models.items()}
    return max(score, key=score.get), score


def laplace_order(models, band, h_coord=1e-4, h_logw=1e-3):
    """Laplace evidence over coordinates and log weights (ridges held at their optima). The uniform priors
    (coordinates on the search box, log weights on their bounds) are moment-matched by Gaussians of precision
    P = 12 / width^2, so the Occam factor is -0.5 ln det(I + P^-1/2 H P^-1/2), H the Hessian of the negative profiled
    evidence (finite differences of the analytic gradient). An unidentified direction (a point whose weights are at
    the lower bound) then costs nothing and earns nothing. ln K! accounts for the label symmetry of the points."""
    lo, hi = band.search_box()
    F = band.F
    score = {0: models[0]['evidence']}
    for K, m in models.items():
        if K == 0:
            continue
        th = m['theta']
        n = 3 * K + K * F
        width = np.r_[np.tile(hi - lo, K), np.full(K * F, LOGW2[1] - LOGW2[0])]
        hs = np.r_[np.full(3 * K, h_coord), np.full(K * F, h_logw)]
        Hm = np.zeros((n, n))
        for i in range(n):
            tp, tm = th.copy(), th.copy(); tp[i] += hs[i]; tm[i] -= hs[i]
            Hm[:, i] = (joint_objective(tp, band, K)[1][:n] - joint_objective(tm, band, K)[1][:n]) / (2 * hs[i])
        Hm = 0.5 * (Hm + Hm.T)
        D = width / np.sqrt(12.0)
        A = np.eye(n) + (D[:, None] * Hm) * D[None, :]
        lam = np.maximum(np.linalg.eigvalsh(A), 1e-12)
        score[K] = m['evidence'] - 0.5 * float(np.sum(np.log(lam))) + math.lgamma(K + 1)
    return max(score, key=score.get), score


def cv_order(band, kmax, folds, **kw):
    err = np.zeros(kmax + 1)
    for held in folds:
        train = np.setdiff1d(np.arange(band.M), held)
        sub = band.subset(train)
        models = fit_band(sub, kmax, **kw)
        for K, m in models.items():
            for f, p in enumerate(band.probs):
                pred = predict_points(sub, m, f, q=band.x[held])
                err[K] += float(np.sum(np.abs(p.y[held] - pred) ** 2)) / float(np.vdot(p.y, p.y).real)
    return int(np.argmin(err)), err


# ------------------------------------------------------------------ joint ball regions over the bins (a, comparators)

def _rsplit(t, J, F):
    cen = t[:3 * J]; lrad = t[3 * J:4 * J]
    lw = t[4 * J:4 * J + J * F].reshape(F, J); lr = t[4 * J + J * F:4 * J + J * F + F]
    return cen, lrad, lw, lr


def region_joint_objective(t, band, J):
    cen, lrad, lw, lr = _rsplit(t, J, band.F)
    tot, g = 0.0, np.zeros_like(t)
    for f, p in enumerate(band.probs):
        v, gr = E.mo26_objective(np.concatenate((cen, lrad, lw[f], [lr[f]])), p, J)
        if v >= 1e29:
            return 1e30, np.zeros_like(t)
        tot += v
        g[:4 * J] += gr[:4 * J]
        g[4 * J + f * J:4 * J + (f + 1) * J] = gr[4 * J:5 * J]
        g[4 * J + J * band.F + f] = gr[5 * J]
    return tot, g


def fit_regions(band, J, starts, maxiter=300):
    """Joint ball regions (shared centres and radii, per-bin weights and ridge); `starts` is a list of
    (centres (J,3), weights (F,J), ridges (F,)) and the highest evidence is kept."""
    lo, hi = band.search_box()
    bounds = [(lo[c], hi[c]) for _ in range(J) for c in range(3)] + [LOGRAD] * J + [LOGW2] * (J * band.F) + [LOGRHO2] * band.F
    lb, ub = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    best = None
    for cen0, W0, R0 in starts:
        t0 = np.clip(np.concatenate((np.asarray(cen0).ravel(), np.full(J, np.log(0.05)),
                                     np.log(np.maximum(W0, 1e-4)).ravel(), np.log(R0))), lb, ub)
        r = minimize(region_joint_objective, t0, args=(band, J), jac=True, method='L-BFGS-B', bounds=bounds,
                     options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    cen, lrad, lw, lr = _rsplit(best.x, J, band.F)
    return dict(centres=cen.reshape(J, 3), radii=np.exp(lrad), lw=lw, R=np.exp(lr), evidence=-float(best.fun),
                converged=bool(best.success), pruned=int(np.sum(np.all(lw <= LOGW2[0] + 1e-3, axis=0))))


def scan_start(band, J):
    lo, hi = band.search_box()
    nodes = E.grid_nodes(lo, hi, 0.25 * band.lam_mid(), max_nodes=SCAN_NODES)
    R0 = np.array([remainder_rho(p) for p in band.probs])
    st = band_scan(band, nodes, np.zeros((0, 3)), np.zeros((band.F, 0)), R0, J, 0.5 * band.lam_mid())
    cen = np.array([n for n, _, _ in st]); W = np.stack([w for _, w, _ in st], 1)
    return cen, W, R0


def predict_regions(band, rm, f):
    p = band.probs[f]
    return E.region_predict(p, rm['centres'], rm['radii'], rm['lw'][f], rm['R'][f])


def extend(band, m, K):
    """Piece (a): the K selected points refined as ball regions, from the points and from the region kernel's own
    scan; the higher joint evidence is kept."""
    starts = [(m['S'], m['W'], m['R'])]
    cen, W, R0 = scan_start(band, K)
    if len(cen) == K:
        starts.append((cen, W, R0))
    return fit_regions(band, K, starts)


def gsmf_fit(band, J=KMAX):
    """Comparator: group-sparse multi-frequency region kernel (reimplementation from the abstract of Matsuda &
    Otani, ICASSP 2026): up to J regions shared across the bins, started from the summed scan; regions whose weight
    reaches the lower bound in every bin are pruned."""
    cen, W, R0 = scan_start(band, J)
    return fit_regions(band, len(cen), [(cen, W, R0)])


# ------------------------------------------------------------------ comparators with the common search domain

@contextlib.contextmanager
def _common_domain_dictionary():
    """Sparse Bayesian learning with the version-1 updates, but its monopole dictionary on the common search
    domain (region box + max(lambda, 1.5 m)), spacing lambda/4 coarsened to at most SBL_NODES, plus plane waves."""
    orig = E.sbl_dictionary

    def common(prob, spacing_wl=0.25, n_dirs=None, max_nodes=SBL_NODES):
        lo, hi = prob.search_box()
        nodes = E.grid_nodes(lo, hi, spacing_wl * prob.lam, max_nodes)
        nodes = nodes[cdist(prob.x, nodes).min(axis=0) > E.GUARD]
        R = 0.5 * float(np.linalg.norm(prob.box[1] - prob.box[0]))
        if n_dirs is None:
            N = int(np.clip(np.ceil(prob.k * R) + 1, 3, 24))
            n_dirs = 2 * (N + 1) ** 2
        from fields import fibonacci_sphere
        return nodes, fibonacci_sphere(n_dirs)
    E.sbl_dictionary = common
    try:
        yield
    finally:
        E.sbl_dictionary = orig


def sbl_common(p, rule):
    with _common_domain_dictionary():
        return E.sbl_fit(p, rule)


@contextlib.contextmanager
def common_bounds():
    """Version-1 functions run with the version-2 weight and ridge bounds (parity of tuning ranges)."""
    orig = E.LOGW, E.LOGRHO
    E.LOGW, E.LOGRHO = LOGW2, LOGRHO2
    try:
        yield
    finally:
        E.LOGW, E.LOGRHO = orig


def mo26_cap(p, J=KMAX):
    """Region kernel (version-1 reimplementation) with the cap raised to J regions, own scan start, and the
    version-2 weight bound."""
    with common_bounds():
        return E.mo26_fit(p, J=J, init='scan')
