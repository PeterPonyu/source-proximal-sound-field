"""Author-recipe variants of the two region-kernel comparators, after the full texts of Matsuda & Otani (ICASSP 2026,
secs. 4-5; JASA 2026, secs. III-V; read, not copied), for a registered sensitivity check (PROTOCOLS/TB-PROX.md; design
record PROX-DESIGN.md). Supersedes common/prox.py, whose leave-one-out loss came from the WASPAA 2025 recipe before the
full texts were available (kept as development history).

Both: kernel K = sum_s gamma_s kappa^(s), source-region kernels of balls with a Gaussian radial density (sigma = rho/3,
G = exp(ikr)/(4 pi r)), not normalised per region, evaluated at the microphones by the spherical-wave series (eq. 13 /
JASA eq. 31) and, for targets inside or near a ball, by a dense cubature; objective: the negative log marginal likelihood
y^H C^-1 y + log det C (C = K + sigma^2 I, sigma^2 fixed) plus a sparsity penalty, weights >= 0 starting at 1/S; no
homogeneous term. Dictionary: a grid at DICT_SPACING over the region box + DICT_MARGIN; each region's radius follows the
papers' rule that every microphone lies outside every ball (rho_s = min(rho_0, d_s - MIC_MARGIN), centres with rho_s <
R_MIN dropped).
gsmf_prox (band), ICASSP 2026: per-bin weights, l2,1 penalty with the non-negative group soft threshold (eq. 24),
  proximal gradient with a fixed step for a fixed number of iterations; rho_0 = delta/sqrt(2); centres and radii fixed.
mo26_prox (one bin), JASA 2026 (Gs, opt): l1 penalty with the non-negative soft threshold (eq. 41); Algorithm 1 (fixed
  step, stopped when ||d gamma||_2 <= eps1 or after iters); then stage 2 of Algorithm 2 on the active set: joint steps of
  the weights (proximal), radii and centres (plain gradient, Cartesian), stopped as soon as the objective rises (keeping
  the previous iterate), at most refine_max steps; rho_0 = (sqrt 3 / 2) delta.
Scale-free transfer: each bin's pressures are divided by their RMS over the microphones; sigma^2 and the starting kernel
keep each paper's ratios to its own data power (recomputed from its stated set-up in prox2_checks.py); steps, zeta and
eps1 are unchanged by that joint rescaling. Settings: the dicts GSMF and MO26.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
import estimators as E
import mo2025 as MO

SIGMA_RATIO = 1 / 3        # Gaussian radial density: sigma = rho / 3 (both papers)
ZETA_GRID = tuple(10.0 ** l for l in range(-3, 3))

# Each paper's setting in its own data units, and the data units fixed by its stated set-up (recomputed in
# prox2_checks.py): mean microphone power P and mean diagonal K0 of the starting kernel (Gamma = 1/S, G = exp(ikr)/(4 pi r)).
# The scale-free transfer divides the pressures by their RMS and keeps sigma^2/P and K0/P; step sizes, zeta and the
# stopping tolerance are unchanged by that joint rescaling.
#   ICASSP 2026 (group-sparse, gsmf_prox): 16 microphones on a 1 m circle, sources at (0.37, 0, 0) and (-0.42, -0.4, 0)
#     m, unit variance; dictionary 0.1 m inside 0.8 m (S = 197), rho = delta/sqrt(2); sigma^2 = 1e-3, eta = 1e-4.
#   JASA 2026 (mo26_prox): 36 microphones on a 1 m sphere, one unit source in the 0.4 m cube at (0.2, 0, 0); dictionary
#     0.2 m inside 0.8 m (S = 257), rho = (sqrt 3 / 2) delta; sigma^2 = 1e-3, eta_gamma = 1e-3, eta_rho = eta_chi = 1e-5,
#     zeta = 0.1, stage-1 stop ||dgamma||_2 <= 1e-6 (at most 2,000), stage 2 at most 100 iterations, stopped when the loss rises.
ICASSP_POWER, ICASSP_K0 = 0.01690, 0.00975
JASA_POWER, JASA_K0 = 0.006509, 0.007272
GSMF = dict(sigma2=1e-3 / ICASSP_POWER, k0_rel=ICASSP_K0 / ICASSP_POWER, eta=1e-4, zeta=0.1, iters=2000, eps1=None,
            rho_factor=1 / np.sqrt(2))
MO26 = dict(sigma2=1e-3 / JASA_POWER, k0_rel=JASA_K0 / JASA_POWER, eta=1e-3, zeta=0.1, iters=2000, eps1=1e-6,
            rho_factor=np.sqrt(3) / 2, refine_max=100, eta_rho=1e-5, eta_chi=1e-5)
# the dictionary: both papers' grid spacing 0.2 m (JASA's tuned value) over a bounded source volume, the region box
# enlarged by DICT_MARGIN (None: the registered common search domain); S_MAX only guards against coarsening
DICT_SPACING = 0.2
DICT_MARGIN = 1.0
S_MAX = 20000
MIC_MARGIN = 0.05          # every microphone stays at least this far outside every ball (the series is valid only
                           # outside the ball; prox2_checks.py validates it 5 cm outside)
R_MIN = 0.02               # centres whose radius rho_s = min(rho_0, d_s - MIC_MARGIN) falls below this are dropped
SINC = False
RAD_BOUNDS = (0.005, 1.0)
CHUNK = 32


# ------------------------------------------------------------------ dictionary and kernels

def grid_spacing(nodes):
    return max(float(np.max(np.diff(np.unique(nodes[:, c])))) if len(np.unique(nodes[:, c])) > 1 else 0.0 for c in range(3))


def dictionary(x, lo, hi, rho_factor):
    """Grid centres on the box (spacing DICT_SPACING; coarsened only beyond S_MAX). Each region's radius follows the
    papers' rule that every microphone lies outside every ball: rho_s = min(rho_0, d_s - MIC_MARGIN), rho_0 =
    rho_factor * delta, d_s the distance to the nearest microphone; centres with rho_s < R_MIN are dropped."""
    nodes = E.grid_nodes(lo, hi, DICT_SPACING, max_nodes=S_MAX)
    delta = grid_spacing(nodes)
    rho0 = rho_factor * delta
    rad = np.minimum(rho0, cdist(nodes, x).min(axis=1) - MIC_MARGIN)
    keep = rad >= R_MIN
    return nodes[keep], rad[keep], rho0, delta


def unit_quadrature(rho, k):
    """Ball quadrature of mo2025 for this radius, expressed on the unit ball: nodes u (n, 3), weights w (n,).
    With sigma = rho/3 the weights do not depend on the radius, so a region (c, r) has nodes c + r u."""
    nodes, w = MO.ball_quadrature(rho, k, sigma_ratio=SIGMA_RATIO)
    return nodes / rho, w


def factors(points, cen, rad, k, uq):
    """Columns G(points | c + r u_n) sqrt(w_n), G = exp(ikd)/(4 pi d): (P, J, n)."""
    u, w = uq
    nodes = cen[:, None, :] + np.asarray(rad, float).reshape(-1, 1, 1) * u[None]
    d = np.linalg.norm(points[:, None, None, :] - nodes[None], axis=-1)
    return np.exp(1j * k * d) / (4 * np.pi * d) * np.sqrt(w)


def series_kernel(xa, xb, c, rho, k, nmax=60, n_s=200):
    """The same kernel by the spherical-wave series (valid when every point lies outside the ball): sum over orders
    of xi_n (2n+1)/(4 pi) h_n(k r_a) conj(h_n(k r_b)) P_n(cos), xi_n = k^2 int_0^rho j_n(ks)^2 eta(s) s^2 ds."""
    from scipy.special import spherical_jn, spherical_yn, eval_legendre
    sig = SIGMA_RATIO * rho
    s, ws = np.polynomial.legendre.leggauss(n_s)
    s = 0.5 * rho * (s + 1); ws = 0.5 * rho * ws
    eta = (8 * np.pi ** 3 * sig ** 6) ** -0.5 * np.exp(-s ** 2 / (2 * sig ** 2))
    da, db = xa - c, xb - c
    ra, rb = np.linalg.norm(da, axis=1), np.linalg.norm(db, axis=1)
    cos = np.clip((da / ra[:, None]) @ (db / rb[:, None]).T, -1, 1)
    out = np.zeros((len(xa), len(xb)), complex)
    for n in range(nmax + 1):
        xi = k ** 2 * float(np.sum(ws * spherical_jn(n, k * s) ** 2 * eta * s ** 2))
        ha = spherical_jn(n, k * ra) + 1j * spherical_yn(n, k * ra)
        hb = spherical_jn(n, k * rb) + 1j * spherical_yn(n, k * rb)
        out += xi * (2 * n + 1) / (4 * np.pi) * np.outer(ha, hb.conj()) * eval_legendre(n, cos)
    return out


NMAX = 40                  # series truncation order (prox2_checks.py: below 1e-11 relative at 5 cm outside the ball)
N_S = 200                  # radial Gauss-Legendre nodes for xi_n
FINE_QUAD = (24, 1200)     # quadrature for target points inside or within 2 cm of a ball


def _eta_hat():
    """Radial Gaussian density on the unit ball (sigma = 1/3), times t^2, with Gauss-Legendre nodes t and weights."""
    t, wt = np.polynomial.legendre.leggauss(N_S)
    t = 0.5 * (t + 1); wt = 0.5 * wt
    sig = SIGMA_RATIO
    return t, wt * (8 * np.pi ** 3 * sig ** 6) ** -0.5 * np.exp(-t ** 2 / (2 * sig ** 2)) * t ** 2


def xi(rho, k, deriv=False):
    """a_n = xi_n (2n+1)/(4 pi), xi_n = k^2 int_0^rho j_n(ks)^2 eta(s) s^2 ds (eta: Gaussian, sigma = rho/3), n = 0..NMAX;
    with deriv, also d a_n / d rho."""
    from scipy.special import spherical_jn
    t, w = _eta_hat()
    n = np.arange(NMAX + 1)[:, None]
    j = spherical_jn(n, k * rho * t[None, :])
    a = k ** 2 * (j ** 2 @ w) * (2 * n[:, 0] + 1) / (4 * np.pi)
    if not deriv:
        return a
    dj = spherical_jn(n, k * rho * t[None, :], derivative=True)
    da = k ** 2 * ((2 * j * dj * k * t[None, :]) @ w) * (2 * n[:, 0] + 1) / (4 * np.pi)
    return a, da


def xi_many(rads, k):
    """a_n for each radius (S, NMAX+1), in chunks."""
    out = np.empty((len(rads), NMAX + 1))
    for b in range(0, len(rads), 256):
        out[b:b + 256] = np.array([xi(r, k) for r in rads[b:b + 256]])
    return out


def _hankel(r, k, deriv=False):
    from scipy.special import spherical_jn, spherical_yn
    n = np.arange(NMAX + 1).reshape((-1,) + (1,) * np.ndim(r))
    h = spherical_jn(n, k * r) + 1j * spherical_yn(n, k * r)
    if not deriv:
        return h
    return h, spherical_jn(n, k * r, derivative=True) + 1j * spherical_yn(n, k * r, derivative=True)


def series_stacks(x, cen, a, k):
    """Sub-kernels at the microphones by the series, (S, M, M): sum_n a_n h_n(k r_i) conj(h_n(k r_j)) P_n(u_ij).
    a: (NMAX+1,) for a common radius or (S, NMAX+1) per region."""
    S, M = len(cen), len(x)
    a = np.asarray(a)
    if a.ndim == 1:
        a = np.broadcast_to(a, (S, len(a)))
    K = np.empty((S, M, M), complex)
    B = CHUNK * 4
    for b in range(0, S, B):
        d = x[None, :, :] - cen[b:b + B, None, :]
        r = np.linalg.norm(d, axis=-1)
        dh = d / r[..., None]
        u = np.clip(np.einsum('smc,snc->smn', dh, dh), -1, 1)
        h = _hankel(r, k)
        ab = a[b:b + B]
        P0, P1 = np.ones_like(u), u
        acc = ab[:, 0, None, None] * h[0][:, :, None] * h[0].conj()[:, None, :] * P0
        acc += ab[:, 1, None, None] * h[1][:, :, None] * h[1].conj()[:, None, :] * P1
        for n in range(1, NMAX):
            P0, P1 = P1, ((2 * n + 1) * u * P1 - n * P0) / (n + 1)
            acc += ab[:, n + 1, None, None] * h[n + 1][:, :, None] * h[n + 1].conj()[:, None, :] * P1
        K[b:b + B] = acc
    return K


def _legendre(u):
    """P_n(u) and dP_n/du for n = 0..NMAX (recurrences without division by 1 - u^2)."""
    P = [np.ones_like(u), u]
    D = [np.zeros_like(u), np.ones_like(u)]
    for n in range(1, NMAX):
        P.append(((2 * n + 1) * u * P[n] - n * P[n - 1]) / (n + 1))
        D.append(D[n - 1] + (2 * n + 1) * P[n])
    return P, D


def series_region_kernel(x, c, rho, k):
    d = x - c
    r = np.linalg.norm(d, axis=1)
    dh = d / r[:, None]
    u = np.clip(dh @ dh.T, -1, 1)
    a = xi(rho, k)
    h = _hankel(r, k)
    P, _ = _legendre(u)
    return sum(a[n] * (h[n][:, None] * h[n].conj()[None, :]) * P[n] for n in range(NMAX + 1))


def series_region_grad(x, c, rho, k, B):
    """For one region: Re tr(B K), and its gradients with respect to the centre (3,) and the radius."""
    a, da = xi(rho, k, deriv=True)
    d = x - c
    r = np.linalg.norm(d, axis=1)
    dh = d / r[:, None]
    u = np.clip(dh @ dh.T, -1, 1)
    h, hp = _hankel(r, k, deriv=True)
    P, D = _legendre(u)
    BT = B.T
    trK = 0.0; trKr = 0.0
    gc = np.zeros(3, complex)
    R = np.zeros((len(x), len(x)), complex)
    for n in range(NMAX + 1):
        hh = h[n][:, None] * h[n].conj()[None, :]
        Q = BT * P[n]
        t = np.sum(Q * hh)
        trK += a[n] * t; trKr += da[n] * t
        gc += -k * a[n] * ((hp[n] * (Q @ h[n].conj()))[:, None] * dh).sum(0)
        gc += -k * a[n] * ((hp[n].conj() * (Q.T @ h[n]))[:, None] * dh).sum(0)
        R += a[n] * BT * hh * D[n]
    ir = 1 / r
    du = (np.einsum('ij,jc->c', R * ir[:, None], dh) - np.einsum('ij,ic->c', R * u * ir[:, None], dh)
          + np.einsum('ij,ic->c', R * ir[None, :], dh) - np.einsum('ij,jc->c', R * u * ir[None, :], dh))
    gc += -du
    return float(np.real(trK)), np.real(gc), float(np.real(trKr))


def cross_kernel(q, x, c, rho, k, a):
    """Kernel between targets q and microphones x for one region: the series where the target lies more than 2 cm
    outside the ball, a fine quadrature otherwise (the series needs every point outside the ball)."""
    dx = x - c
    rx = np.linalg.norm(dx, axis=1); hx = _hankel(rx, k)
    dq = q - c
    rq = np.linalg.norm(dq, axis=1)
    out = np.zeros((len(q), len(x)), complex)
    far = rq > rho + 0.02
    if far.any():
        hq = _hankel(rq[far], k)
        u = np.clip((dq[far] / rq[far, None]) @ (dx / rx[:, None]).T, -1, 1)
        P0, P1 = np.ones_like(u), u
        acc = a[0] * hq[0][:, None] * hx[0].conj()[None, :] * P0 + a[1] * hq[1][:, None] * hx[1].conj()[None, :] * P1
        for n in range(1, NMAX):
            P0, P1 = P1, ((2 * n + 1) * u * P1 - n * P0) / (n + 1)
            acc += a[n + 1] * hq[n + 1][:, None] * hx[n + 1].conj()[None, :] * P1
        out[far] = acc
    if (~far).any():
        nodes, w = MO.ball_quadrature(rho, k, sigma_ratio=SIGMA_RATIO, n_r=FINE_QUAD[0], n_a=FINE_QUAD[1])
        Fq = MO.green4pi(q[~far], c[None] + nodes, k) * np.sqrt(w)
        Fx = MO.green4pi(x, c[None] + nodes, k) * np.sqrt(w)
        out[~far] = Fq @ Fx.conj().T
    return out


# ------------------------------------------------------------------ objective, gradient, proximal step

def nlml(A, y):
    """L = y^H A^-1 y + log det A and B = A^-1 - a a^H (dL/dtheta = Re tr(B dA/dtheta) for Hermitian dA)."""
    M = len(y)
    fac = cho_factor(A, lower=True, check_finite=False)
    a = cho_solve(fac, y, check_finite=False)
    L = float(np.real(np.vdot(y, a))) + 2 * float(np.sum(np.log(np.real(np.diag(fac[0])))))
    B = cho_solve(fac, np.eye(M, dtype=complex), check_finite=False) - np.outer(a, a.conj())
    return L, B


def group_prox(Z, thr):
    """Non-negative group soft threshold (the paper's eq. 24): project on the non-negative orthant, then shrink
    each region's weight vector across the bins by thr. With one bin: max(0, z - thr)."""
    Z = np.maximum(Z, 0.0)
    n = np.linalg.norm(Z, axis=0)
    s = np.where(n > thr, 1.0 - thr / np.where(n > 0, n, 1.0), 0.0)
    return Z * s[None, :]


def weights_loss(Bases, ys, Ks, flat, W):
    F, S = W.shape
    g = np.empty((F, S)); Lt = 0.0
    for f in range(F):
        act = np.flatnonzero(W[f] > 0)
        if len(act) > S // 8:
            A = Bases[f] + (W[f].astype(complex) @ flat[f]).reshape(Ks[f].shape[1:])
        else:
            A = Bases[f] + np.tensordot(W[f, act], Ks[f][act], axes=1)
        L, B = nlml(A, ys[f])
        Lt += L
        g[f] = np.real(flat[f] @ np.ascontiguousarray(B.T).reshape(-1))
    return Lt, g


def stage_weights(Bases, ys, Ks, zeta, eta, iters, eps1=None, record=None):
    """Proximal gradient on the weights from 1/S: at most `iters` steps; with eps1, stop once ||dW||_2 <= eps1."""
    F, S = len(ys), Ks[0].shape[0]
    W = np.full((F, S), 1.0 / S)
    flat = [K.reshape(S, -1) for K in Ks]
    for _ in range(iters):
        Lt, g = weights_loss(Bases, ys, Ks, flat, W)
        if record is not None:
            record.append(Lt + zeta * float(np.sum(np.linalg.norm(W, axis=0))))
        Wn = group_prox(W - eta * g, eta * zeta)
        done = eps1 is not None and float(np.linalg.norm(Wn - W)) <= eps1
        W = Wn
        if done:
            break
    return W


def refine_loss_grads(x, ks, cs, Bases, ys, cen, rad, W):
    """Summed NLML of regions (cen, rad) with weights W (F, J) and per-bin kernel constants cs, and its gradients
    (series kernel, analytic derivatives)."""
    F, J = W.shape
    gW = np.zeros_like(W); gc = np.zeros((J, 3)); gr = np.zeros(J); Lt = 0.0
    for f in range(F):
        Kj = [series_region_kernel(x, cen[j], rad[j], ks[f]) for j in range(J)]
        A = Bases[f].astype(complex)
        for j in range(J):
            if W[f, j] > 0:
                A = A + cs[f] * W[f, j] * Kj[j]
        L, B = nlml(A, ys[f])
        Lt += L
        for j in range(J):
            tK, tc, tr = series_region_grad(x, cen[j], rad[j], ks[f], B)
            gW[f, j] = cs[f] * tK
            gc[j] += cs[f] * W[f, j] * tc
            gr[j] += cs[f] * W[f, j] * tr
    return Lt, gW, gc, gr


def _project(cen, rad, cen_old, rad_old, x, lo, hi):
    cen = np.clip(cen, lo, hi)
    rad = np.clip(rad, *RAD_BOUNDS)
    rad = np.minimum(rad, cdist(cen, x).min(axis=1) - MIC_MARGIN)
    bad = rad < RAD_BOUNDS[0]
    cen[bad], rad[bad] = cen_old[bad], rad_old[bad]
    return cen, rad


def stage_refine(x, ks, cs, Bases, ys, cen, rad, W, lo, hi, cfg, record=None):
    """JASA 2026 Algorithm 2, stage 2: on the active set, joint steps of the weights (proximal), radii and centres
    (plain gradient); stop as soon as the objective rises, keeping the previous iterate; at most refine_max steps."""
    obj = lambda L, W: L + cfg['zeta'] * float(np.sum(np.linalg.norm(W, axis=0)))
    Lt, gW, gc, gr = refine_loss_grads(x, ks, cs, Bases, ys, cen, rad, W)
    cur = obj(Lt, W)
    if record is not None:
        record.append(cur)
    for _ in range(cfg['refine_max']):
        Wn = group_prox(W - cfg['eta'] * gW, cfg['eta'] * cfg['zeta'])
        cn, rn = _project(cen - cfg['eta_chi'] * gc, rad - cfg['eta_rho'] * gr, cen, rad, x, lo, hi)
        Ln, gWn, gcn, grn = refine_loss_grads(x, ks, cs, Bases, ys, cn, rn, Wn)
        new = obj(Ln, Wn)
        if record is not None:
            record.append(new)
        if new > cur:
            break
        W, cen, rad, gW, gc, gr, cur = Wn, cn, rn, gWn, gcn, grn, new
    return cen, rad, W


# ------------------------------------------------------------------ the two variants

class Prepared:
    """Per-band pieces for one variant (cfg = GSMF or MO26): the dictionary on the bounded source volume, the scaled
    sub-kernel stacks of every bin (series, eq. 13) and the unit-power pressures."""

    def __init__(self, band, cfg):
        self.band, self.cfg = band, cfg
        self.x, self.q = band.x, band.q
        self.ks = [p.k for p in band.probs]
        if DICT_MARGIN is None:
            self.lo, self.hi = band.search_box()
        else:
            self.lo, self.hi = band.box[0] - DICT_MARGIN, band.box[1] + DICT_MARGIN
        self.cen, self.rad, self.rho, self.delta = dictionary(self.x, self.lo, self.hi, cfg['rho_factor'])
        self.scale = [float(np.sqrt(np.mean(np.abs(p.y) ** 2))) for p in band.probs]
        self.ys = [p.y / s for p, s in zip(band.probs, self.scale)]
        M = len(self.x)
        self.sigma2 = cfg['sigma2']
        self.Bases = [(p.H if SINC else np.zeros((M, M))) + self.sigma2 * np.eye(M) for p in band.probs]
        self.Ks, self.cs = [], []
        for k in self.ks:
            K = series_stacks(self.x, self.cen, xi_many(self.rad, k), k)
            c = cfg['k0_rel'] / float(np.mean(np.real(np.einsum('smm->sm', K))))
            K *= c
            self.Ks.append(K); self.cs.append(c)

    def predict(self, f, cen, rad, w):
        """Posterior mean at the targets for bin f with regions (cen, rad) and weights w (J,)."""
        p = self.band.probs[f]
        M = len(self.x)
        Kx = self.Bases[f] - self.sigma2 * np.eye(M)
        Kq = (p.Hq if SINC else np.zeros((len(self.q), M))).astype(complex)
        for i in np.flatnonzero(w > 0):
            s = self.cs[f] * w[i]
            Kx = Kx + s * series_region_kernel(self.x, cen[i], rad[i], self.ks[f])
            Kq = Kq + s * cross_kernel(self.q, self.x, cen[i], rad[i], self.ks[f], xi(rad[i], self.ks[f]))
        a = np.linalg.solve(Kx + self.sigma2 * np.eye(M), self.ys[f])
        return self.scale[f] * (Kq @ a)


def gsmf_prox(prep, record=None):
    """ICASSP 2026: the group-sparse multi-frequency method (prep built with GSMF)."""
    cfg = prep.cfg
    rec = []
    W = stage_weights(prep.Bases, prep.ys, prep.Ks, cfg['zeta'], cfg['eta'], cfg['iters'], cfg['eps1'], rec)
    if record is not None:
        record.extend(rec)
    rad = prep.rad
    preds = [prep.predict(f, prep.cen, rad, W[f]) for f in range(len(prep.ks))]
    live = np.any(W > 0, axis=0)
    return dict(preds=preds, info=dict(dictionary=int(len(prep.cen)), delta=prep.delta, rho0=prep.rho, regions=int(live.sum()),
                                       shrunk=int(np.sum(prep.rad < prep.rho - 1e-12)), min_mic_clearance_m=_min_clear(prep.x, prep.cen, prep.rad),
                                       iterations=len(rec), min_target_distance_m=_min_target(prep.q, prep.cen[live], rad[live])))


def mo26_prox(prep, f, record=None):
    """JASA 2026 (Gs, opt) for bin f (prep built with MO26): Algorithm 1, then stage 2 of Algorithm 2."""
    cfg = prep.cfg
    r1, r2 = [], []
    W = stage_weights([prep.Bases[f]], [prep.ys[f]], [prep.Ks[f]], cfg['zeta'], cfg['eta'], cfg['iters'], cfg['eps1'], r1)
    keep = np.flatnonzero(W[0] > 0)
    cen, rad, Wk = prep.cen[keep].copy(), prep.rad[keep].copy(), W[:, keep].copy()
    if len(keep):
        cen, rad, Wk = stage_refine(prep.x, [prep.ks[f]], [prep.cs[f]], [prep.Bases[f]], [prep.ys[f]], cen, rad, Wk,
                                    prep.lo, prep.hi, cfg, r2)
    if record is not None:
        record.extend(r1 + r2)
    pred = prep.predict(f, cen, rad, Wk[0])
    live = Wk[0] > 0
    return dict(pred=pred, info=dict(after_stage1=int(len(keep)), regions=int(live.sum()), iterations=[len(r1), len(r2)],
                                     radii=rad[live].tolist(), min_mic_clearance_m=_min_clear(prep.x, cen, rad),
                                     min_target_distance_m=_min_target(prep.q, cen[live], rad[live])))


def _min_clear(x, cen, rad):
    """Smallest distance from a microphone to the surface of any ball (positive: every microphone is outside)."""
    return float((cdist(x, cen).min(axis=0) - rad).min()) if len(cen) else None


def _min_target(q, cen, rad):
    return float((cdist(q, cen).min(axis=0) - rad).min()) if len(cen) else None
