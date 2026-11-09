"""Author-recipe variants of the two region-kernel comparators, for a registered sensitivity check
(PROTOCOLS/TB-PROX.md; design and development record PROX-DESIGN.md).

Neither 2026 paper of Matsuda & Otani has public code or text. The first author's WASPAA 2025 paper (Matsuda, Ribeiro,
Akiyama & Trevino, arXiv:2508.03087, sec. 5.2, eqs. 26-27) gives the learning recipe that the 2026 abstracts describe.
This module changes only what that recipe specifies:
  * the kernel is a non-negative weighted sum of Q sub-kernels from a fixed dictionary, weights started uniform at 1/Q;
  * the loss is the leave-one-out cross-validation error of kernel ridge regression, with the ridge held at 1e-2;
  * the weights take ITERS proximal gradient steps w <- max(0, w - ETA dL/dw - ETA TAU), TAU = 1e-2, ETA = 1e-1;
  * shape parameters (here the centres and radii of the surviving regions) take plain gradient steps.

mo26_prox (one bin): ball regions on the common search domain; ITERS steps of the l1 proximal update above; then ITERS
  joint steps in which the surviving regions' centres and radii move by plain gradient steps while their weights keep
  the proximal step.
gsmf_prox (band): the same dictionary shared by the bins, with per-bin weights; the loss is summed over the bins; the
  proximal step is the non-negative group soft threshold of each candidate's weight vector across the bins (the l2,1
  proximal operator); centres and radii are shared. With one bin it is mo26_prox (prox_checks.py).

Kept from the registered comparators where the recipe is silent: the search domain, the 5 cm microphone guard, the
region kernel (33-node ball quadrature of Green's functions) and its normalisation to unit mean diagonal over the
microphones, the unit-weight sinc term, the bins, the radius bounds.

Scaling, explicit: each bin's pressures are divided by their RMS over the microphones, so the loss is the mean squared
leave-one-out residual in those units (the leave-one-out NMSE), and TAU, ETA and the ridge refer to unit-power data and
unit-diagonal sub-kernels. Predictions are scaled back.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.spatial.distance import cdist
import estimators as E

# the recipe
TAU, ETA, RIDGE, ITERS = 1e-2, 1e-1, 1e-2, 400
# free choices, fixed on development data only (PROX-DESIGN.md)
SPACING_WL = 0.5        # dictionary grid spacing in wavelengths (bin; band: mid-band), coarsened to Q_MAX centres
Q_MAX = 512             # at most this many centres per radius
RADII = (0.05,)         # radius menu (m)
ETA_C = 1e-2            # centre step (m per unit gradient)
ETA_R = 1e-3            # radius step (m per unit gradient)
FINAL_RIDGE = 'fixed'   # 'fixed': the learning ridge; 'loo': chosen by leave-one-out error after learning
LOSS = 'mean'           # leave-one-out loss over the microphones: 'mean' (the leave-one-out NMSE) or 'sum'
STEP = 'backtracking'   # 'backtracking' (Beck & Teboulle 2009, cited by the 2026 paper) or 'fixed' (ETA every step)
T_MIN = 1e-8           # smallest step factor of the backtracking
RAD_BOUNDS = (0.005, 0.5)   # the registered radius bounds (v2.LOGRAD)


# ------------------------------------------------------------------ region kernel pieces

def factors(points, cen, rad, k):
    """Ball-quadrature Green columns F (P, Q, n) of Q regions at the points (the registered region kernel)."""
    nodes = cen[:, None, :] + rad[:, None, None] * E.BALL[None, :, :]
    d = np.linalg.norm(points[:, None, None, :] - nodes[None], axis=-1)
    return np.exp(1j * k * d) / d * np.sqrt(E.BALL_W)[None, None, :]


def gram_stack(Fx):
    """Normalised sub-kernels K_q = F_q F_q^H / mean_m (F_q F_q^H)_mm, stacked (Q, M, M), and the normalisers."""
    nrm = np.mean(np.sum(np.abs(Fx) ** 2, axis=2), axis=0)
    Fq = np.ascontiguousarray(Fx.transpose(1, 0, 2))
    return np.matmul(Fq, np.conj(Fq.transpose(0, 2, 1))) / nrm[:, None, None], nrm


def dictionary(x, lo, hi, lam):
    """Candidate ball regions: grid centres on the search box (spacing SPACING_WL lambda, coarsened to Q_MAX), one copy
    per radius of RADII, without the candidates that break the microphone guard."""
    nodes = E.grid_nodes(lo, hi, SPACING_WL * lam, max_nodes=Q_MAX)
    dmin = cdist(nodes, x).min(axis=1)
    cen, rad = [], []
    for R in RADII:
        ok = dmin - R > E.GUARD
        cen.append(nodes[ok]); rad.append(np.full(int(ok.sum()), float(R)))
    return np.vstack(cen), np.concatenate(rad)


# ------------------------------------------------------------------ leave-one-out loss and its gradient

def _ls(M):
    """Loss scale: 1/M for the mean over the microphones, 1 for the sum."""
    return 1.0 / M if LOSS == 'mean' else 1.0


def loo(A, y):
    """Leave-one-out loss L = _ls(M) sum_i |e_i|^2 with e_i = (A^-1 y)_i / (A^-1)_ii, and the matrix G such that
    dL/dtheta = 2 _ls(M) Re tr(dA/dtheta G) for every real parameter theta of the Hermitian matrix A."""
    M = len(y)
    fac = cho_factor(A, lower=True, check_finite=False)
    B = cho_solve(fac, np.eye(M, dtype=complex), check_finite=False)
    a = B @ y
    b = np.real(np.diag(B))
    e = a / b
    L = _ls(M) * float(np.sum(np.abs(e) ** 2))
    u = np.conj(e) / b
    v = np.abs(e) ** 2 / b
    G = (B * v[None, :]) @ B - np.outer(a, u @ B)
    return L, G


def group_prox(Z, step):
    """Proximal operator of step * TAU * sum_q ||W[:, q]||_2 plus non-negativity: project on the non-negative orthant,
    then shrink each candidate's vector across the bins. With one bin: max(0, z - step * TAU)."""
    Z = np.maximum(Z, 0.0)
    n = np.linalg.norm(Z, axis=0)
    s = np.where(n > step * TAU, 1.0 - step * TAU / np.where(n > 0, n, 1.0), 0.0)
    return Z * s[None, :]


# ------------------------------------------------------------------ the two stages

def _weights_loss(Hr, ys, Ks, flat, W):
    """Summed leave-one-out loss of the weights W (F, Q) and its gradient (F, Q)."""
    F, Q = W.shape
    M = len(ys[0])
    g = np.empty((F, Q)); Lt = 0.0
    for f in range(F):
        act = np.flatnonzero(W[f] > 0)
        if len(act) > Q // 8:
            A = Hr[f] + (W[f].astype(complex) @ flat[f]).reshape(M, M)
        else:
            A = Hr[f] + np.tensordot(W[f, act], Ks[f][act], axes=1)
        L, G = loo(A, ys[f])
        Lt += L
        g[f] = (2.0 * _ls(M)) * np.real(flat[f] @ np.ascontiguousarray(G.T).reshape(-1))
    return Lt, g


def _stage_weights(Hr, ys, Ks, iters=ITERS, record=None):
    """ITERS proximal gradient steps on the weights W (F, Q), started uniform at 1/Q. With STEP == 'backtracking'
    the step starts at ETA and is halved until the quadratic upper bound of Beck & Teboulle (2009) holds; it never
    grows again. With STEP == 'fixed' every step is ETA."""
    F, Q = len(ys), Ks[0].shape[0]
    W = np.full((F, Q), 1.0 / Q)
    flat = [K.reshape(Q, -1) for K in Ks]
    t = ETA
    Lt, g = _weights_loss(Hr, ys, Ks, flat, W)
    for _ in range(iters):
        if record is not None:
            record.append(Lt)
        while True:
            Wn = group_prox(W - t * g, t)
            Ln, gn = _weights_loss(Hr, ys, Ks, flat, Wn)
            D = Wn - W
            if STEP == 'fixed' or Ln <= Lt + float(np.sum(g * D)) + float(np.sum(D * D)) / (2 * t) or t < T_MIN:
                break
            t *= 0.5
        W, Lt, g = Wn, Ln, gn
    return W


def region_terms(x, cen, rad, k):
    """For J regions at once (the registered kernel; vectorised form of estimators._region_parts): F (M, J, n),
    dF/dcentre (M, J, n, 3), dF/dradius (M, J, n), normalisers (J,) and their derivatives (J, 3) and (J,)."""
    sv = np.sqrt(E.BALL_W)
    nodes = cen[:, None, :] + rad[:, None, None] * E.BALL[None, :, :]          # (J, n, 3)
    diff = nodes[None] - x[:, None, None, :]                                   # (M, J, n, 3)
    d = np.linalg.norm(diff, axis=-1)
    g = np.exp(1j * k * d) / d
    dg = (g * (1j * k - 1 / d))[..., None] * diff / d[..., None]              # d g / d node
    Fx = g * sv
    dFc = dg * sv[None, None, :, None]
    dFa = np.einsum('mjnc,nc->mjn', dg, E.BALL) * sv
    nrm = np.mean(np.sum(E.BALL_W / d ** 2, axis=2), axis=0)
    dn_node = np.mean(-2 * E.BALL_W[None, None, :, None] * diff / d[..., None] ** 4, axis=0)   # (J, n, 3)
    return Fx, dFc, dFa, nrm, dn_node.sum(axis=1), np.einsum('jnc,nc->j', dn_node, E.BALL)


def loss_and_grads(x, ks, Hr, ys, cen, rad, W):
    """Summed leave-one-out loss of the regions (cen, rad) with weights W (F, J), and its gradients."""
    F, J = W.shape
    M = len(x)
    c2 = 2.0 * _ls(M)
    gW = np.zeros_like(W); gc = np.zeros((J, 3)); gr = np.zeros(J); Lt = 0.0
    for f in range(F):
        Fx, dFc, dFa, nrm, dnc, dna = region_terms(x, cen, rad, ks[f])
        n = Fx.shape[2]
        Fs = (Fx * np.sqrt(np.maximum(W[f], 0.0) / nrm)[None, :, None]).reshape(M, J * n)
        A = Hr[f] + Fs @ Fs.conj().T
        L, G = loo(A, ys[f])
        Lt += L
        P2 = (G @ Fx.reshape(M, J * n)).reshape(M, J, n)
        P1 = (G.T @ Fx.conj().reshape(M, J * n)).reshape(M, J, n)
        trK = np.sum(Fx.conj() * P2, axis=(0, 2))                                  # tr(F_j F_j^H G)
        gW[f] = c2 * np.real(trK) / nrm
        t_c = np.einsum('mjnc,mjn->jc', dFc, P1) + np.einsum('mjnc,mjn->jc', dFc.conj(), P2)
        t_a = np.sum(dFa * P1, axis=(0, 2)) + np.sum(dFa.conj() * P2, axis=(0, 2))
        wf = W[f]
        gc += c2 * wf[:, None] * np.real(t_c / nrm[:, None] - trK[:, None] * dnc / nrm[:, None] ** 2)
        gr += c2 * wf * np.real(t_a / nrm - trK * dna / nrm ** 2)
    return Lt, gW, gc, gr


def _project(cen, rad, cen_old, rad_old, x, lo, hi):
    """Centres into the search box, radii into their bounds and inside the microphone guard; a region that cannot
    satisfy the guard keeps its previous centre and radius."""
    cen = np.clip(cen, lo, hi)
    rad = np.clip(rad, *RAD_BOUNDS)
    dmin = cdist(cen, x).min(axis=1)
    rad = np.minimum(rad, dmin - E.GUARD - 1e-6)
    bad = rad < RAD_BOUNDS[0]
    cen[bad], rad[bad] = cen_old[bad], rad_old[bad]
    return cen, rad


def _stage_refine(x, ks, Hr, ys, cen, rad, W, lo, hi, iters=ITERS, record=None):
    """ITERS joint steps: the weights take the proximal step, centres and radii plain (projected) gradient steps at
    rates ETA_C and ETA_R. With STEP == 'backtracking' one factor s (start 1, halved, never grown) scales all three
    rates until the quadratic upper bound holds in the metric of the rates."""
    s = 1.0
    Lt, gW, gc, gr = loss_and_grads(x, ks, Hr, ys, cen, rad, W)
    for _ in range(iters):
        if record is not None:
            record.append(Lt)
        while True:
            Wn = group_prox(W - s * ETA * gW, s * ETA)
            cn, rn = _project(cen - s * ETA_C * gc, rad - s * ETA_R * gr, cen, rad, x, lo, hi)
            Ln, gWn, gcn, grn = loss_and_grads(x, ks, Hr, ys, cn, rn, Wn)
            dW, dc, dr = Wn - W, cn - cen, rn - rad
            lin = float(np.sum(gW * dW) + np.sum(gc * dc) + np.sum(gr * dr))
            quad = (float(np.sum(dW * dW)) / ETA + float(np.sum(dc * dc)) / ETA_C + float(np.sum(dr * dr)) / ETA_R) / (2 * s)
            if STEP == 'fixed' or Ln <= Lt + lin + quad or s < T_MIN:
                break
            s *= 0.5
        W, cen, rad, Lt, gW, gc, gr = Wn, cn, rn, Ln, gWn, gcn, grn
    return cen, rad, W


# ------------------------------------------------------------------ fit and predict

def _final_ridge(Kx, y):
    if FINAL_RIDGE == 'fixed':
        return RIDGE
    lam, U = eigh(Kx)
    lam = np.maximum(lam, 0.0)
    uy = U.conj().T @ y
    best = None
    for r in np.exp(np.linspace(E.LOGRHO[0], E.LOGRHO[1], 121)):
        a = U @ (uy / (lam + r))
        dg = np.real(np.einsum('ij,j,ij->i', U, 1 / (lam + r), U.conj()))
        err = float(np.sum(np.abs(a / dg) ** 2))
        if best is None or err < best[0]:
            best = (err, r)
    return float(best[1])


def fit(x, q, ks, Hs, Hqs, Ys, lo, hi, lam, iters=ITERS, trace=False):
    """Shared core: F bins observed at the same microphones x and targets q. Hs, Hqs: the unit-weight sinc matrices
    of each bin (microphones, targets x microphones); Ys (M, F) pressures. Returns per-bin predictions and a record."""
    F, M = len(ks), len(x)
    scale = np.sqrt(np.mean(np.abs(Ys) ** 2, axis=0))
    ys = [Ys[:, f] / scale[f] for f in range(F)]
    Hr = [Hs[f] + RIDGE * np.eye(M) for f in range(F)]
    cen0, rad0 = dictionary(x, lo, hi, lam)
    Ks = [gram_stack(factors(x, cen0, rad0, k))[0] for k in ks]
    rec1, rec2 = ([], []) if trace else (None, None)
    W = _stage_weights(Hr, ys, Ks, iters, rec1)
    del Ks
    keep = np.flatnonzero(np.any(W > 0, axis=0))
    cen, rad, W = cen0[keep].copy(), rad0[keep].copy(), W[:, keep].copy()
    n_stage1 = len(keep)
    if len(keep):
        cen, rad, W = _stage_refine(x, ks, Hr, ys, cen, rad, W, lo, hi, iters, rec2)
    live = np.flatnonzero(np.any(W > 0, axis=0))
    cen, rad, W = cen[live], rad[live], W[:, live]
    preds, ridges = [], []
    for f in range(F):
        Kx = Hs[f].astype(complex)
        Kq = Hqs[f].astype(complex)
        if len(live):
            Fx = factors(x, cen, rad, ks[f]); Fq = factors(q, cen, rad, ks[f])
            nrm = np.mean(np.sum(np.abs(Fx) ** 2, axis=2), axis=0)
            for j in range(len(live)):
                if W[f, j] > 0:
                    s = W[f, j] / nrm[j]
                    Kx += s * (Fx[:, j] @ Fx[:, j].conj().T)
                    Kq += s * (Fq[:, j] @ Fx[:, j].conj().T)
        rho = _final_ridge(Kx, ys[f])
        preds.append(scale[f] * (Kq @ np.linalg.solve(Kx + rho * np.eye(M), ys[f])))
        ridges.append(rho)
    info = dict(dictionary=int(len(rad0)), after_stage1=int(n_stage1), regions=int(len(live)),
                centres=cen.tolist(), radii=rad.tolist(), weights=W.tolist(), ridge=ridges,
                min_target_distance_m=(float((cdist(q, cen).min(axis=0) - rad).min()) if len(live) else None))
    if trace:
        info.update(loss_stage1=rec1, loss_stage2=rec2)
    return preds, info


def mo26_prox(p, iters=ITERS, trace=False):
    """Author-recipe region kernel, one bin (Problem p)."""
    lo, hi = p.search_box()
    preds, info = fit(p.x, p.q, [p.k], [p.H], [p.Hq], p.y[:, None], lo, hi, p.lam, iters, trace)
    return dict(pred=preds[0], info=info)


def gsmf_prox(band, iters=ITERS, trace=False):
    """Author-recipe group-sparse multi-frequency region kernel, one band (v2.Band)."""
    lo, hi = band.search_box()
    Ys = np.stack([p.y for p in band.probs], 1)
    preds, info = fit(band.x, band.q, [p.k for p in band.probs], [p.H for p in band.probs], [p.Hq for p in band.probs],
                      Ys, lo, hi, band.lam_mid(), iters, trace)
    return dict(preds=preds, info=info)
