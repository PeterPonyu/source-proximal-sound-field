"""Single-frequency sound-field estimators for source-proximal regions (TASLP study).

Every estimator receives only: microphone positions x (M, 3), complex pressures y (M,), the frequency, the
assumed sound speed and the region box. The prior-region comparator and the oracle additionally receive source
coordinates and are labelled as such. Predictions are posterior means at target points q.

Proposed model (C1): y = sum_j a_j b_j(x) + u_H(x) + e, with u_H a Gaussian process with the homogeneous-Helmholtz
kernel j0(k |x - x'|) (sinc), a_j ~ CN(0, s2 w_j), e ~ CN(0, s2 rho I), b_j the free-field Green function of the
point s_j normalised to unit root-mean-square over the microphones. The single-source form is Albert & Rath (2020).
New here: greedy sequential localisation with model order K in {0..Kmax} chosen by a penalised evidence or by
microphone cross-validation; continuous multi-start tuning of (s_j, w_j, rho) with analytic evidence gradients;
a coordinate search box that is the region box enlarged by one wavelength and ignores the room walls.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

GUARD = 0.05            # m: a fitted point source keeps at least this distance from every microphone
LOGW = (np.log(1e-4), np.log(1e4))
LOGRHO = (np.log(1e-7), np.log(1e1))
W_MENU = (0.1, 1.0, 10.0, 100.0)
JITTER = 1e-10
MARGIN_M = 1.5          # m: physical floor of the coordinate-search margin around the region box


def nmse(pred, truth):
    den = float(np.vdot(truth, truth).real)
    return float(np.vdot(pred - truth, pred - truth).real / den)


def sinc_k(k, a, b):
    return np.sinc(k * cdist(a, b) / np.pi)


def green(points, nodes, k):
    d = cdist(points, np.atleast_2d(nodes))
    return np.exp(1j * k * d) / d


def basis_grad(x, S, k):
    """Normalised Green columns b (M, K) at the microphones and their coordinate derivatives db (M, K, 3)."""
    S = np.atleast_2d(S)
    diff = S[None, :, :] - x[:, None, :]
    d = np.linalg.norm(diff, axis=2)
    g = np.exp(1j * k * d) / d
    nu = np.sqrt(np.mean(1 / d ** 2, axis=0))
    dd = diff / d[..., None]
    dg = (g * (1j * k - 1 / d))[..., None] * dd
    dnu = -np.mean(diff / d[..., None] ** 4, axis=0) / nu[:, None]
    b = g / nu[None, :]
    db = dg / nu[None, :, None] - g[..., None] * dnu[None, :, :] / nu[None, :, None] ** 2
    return b, db, nu


def target_basis(q, S, k, nu):
    return green(q, S, k) / nu[None, :]


class Problem:
    """One estimation problem: microphones, pressures, targets, frequency, assumed sound speed, region box."""

    def __init__(self, x, y, q, frequency, c=343.0, box=None, room=None):
        self.x, self.y, self.q = np.asarray(x, float), np.asarray(y, complex), np.asarray(q, float)
        self.M = len(self.x)
        self.f, self.c = float(frequency), float(c)
        self.k = 2 * np.pi * self.f / self.c
        self.lam = self.c / self.f
        pts = np.vstack((self.x, self.q))
        self.box = (pts.min(0), pts.max(0)) if box is None else (np.asarray(box[0], float), np.asarray(box[1], float))
        self.room = room
        self.H = sinc_k(self.k, self.x, self.x)
        self.Hq = sinc_k(self.k, self.q, self.x)
        self._eig = None

    def search_box(self, margin=None, inside_room=False):
        """Region box enlarged by `margin` metres (default: the larger of one wavelength and 1.5 m) on every side.

        The room walls do not bound it unless inside_room=True (ablation only)."""
        margin = max(self.lam, MARGIN_M) if margin is None else float(margin)
        lo, hi = self.box[0] - margin, self.box[1] + margin
        if inside_room:
            if self.room is None:
                raise ValueError('room needed')
            lo, hi = np.maximum(lo, 0.02), np.minimum(hi, np.asarray(self.room) - 0.02)
        return lo, hi

    def eig(self):
        if self._eig is None:
            lam, U = eigh(self.H)
            self._eig = (np.maximum(lam, 0.0), U)
        return self._eig

    def subset(self, idx):
        """Same targets, a subset of microphones (used by cross-validation)."""
        p = Problem(self.x[idx], self.y[idx], self.q, self.f, self.c, self.box, self.room)
        return p


# ------------------------------------------------------------------ source-free sinc kernel (K = 0)

def sinc_evidence_curve(prob, rhos):
    lam, U = prob.eig()
    z = np.abs(U.conj().T @ prob.y) ** 2
    out = []
    for r in rhos:
        out.append(-prob.M * np.log(np.sum(z / (lam + r))) - np.sum(np.log(lam + r)))
    return np.array(out)


def sinc_fit(prob, tuning='evidence'):
    """Source-free kernel ridge regression with the sinc kernel (Ueno et al. 2018); rho by evidence or LOO."""
    lam, U = prob.eig()
    uy = U.conj().T @ prob.y
    grid = np.exp(np.linspace(LOGRHO[0], LOGRHO[1], 121))
    if tuning == 'evidence':
        score = sinc_evidence_curve(prob, grid)
        i = int(np.argmax(score))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        res = minimize(lambda t: -sinc_evidence_curve(prob, [np.exp(t[0])])[0], [np.log(grid[i])],
                       bounds=[(np.log(lo), np.log(hi))], method='L-BFGS-B')
        rho = float(np.exp(res.x[0]))
    elif tuning == 'loo':
        best = None
        for r in grid:
            alpha = U @ (uy / (lam + r))
            diag = np.real(np.einsum('ij,j,ij->i', U, 1 / (lam + r), U.conj()))
            e = float(np.sum(np.abs(alpha / diag) ** 2))
            if best is None or e < best[0]:
                best = (e, r)
        rho = float(best[1])
    else:
        raise ValueError(tuning)
    alpha = U @ (uy / (lam + rho))
    ev = float(sinc_evidence_curve(prob, [rho])[0])
    return dict(method='sinc-' + tuning, rho=rho, evidence=ev, pred=prob.Hq @ alpha, K=0,
                predict_mics=lambda xm: sinc_k(prob.k, xm, prob.x) @ alpha)


# ------------------------------------------------------------------ point-source model (C1)

def _unpack(theta, K):
    S = theta[:3 * K].reshape(K, 3)
    return S, np.exp(theta[3 * K:4 * K]), float(np.exp(theta[4 * K]))


def source_objective(theta, prob, K, grad=True):
    """Negative profiled log evidence and its gradient for coordinates, log weights and log ridge."""
    S, w, rho = _unpack(theta, K)
    b, db, _ = basis_grad(prob.x, S, prob.k)
    C = prob.H + (b * w[None, :]) @ b.conj().T + (rho + JITTER) * np.eye(prob.M)
    try:
        fac = cho_factor(C, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return (1e30, np.zeros_like(theta)) if grad else 1e30
    alpha = cho_solve(fac, prob.y)
    Q = float(np.vdot(prob.y, alpha).real)
    logdet = 2 * float(np.sum(np.log(np.diag(fac[0]).real)))
    ev = -prob.M * np.log(Q) - logdet
    d = cdist(prob.x, S)
    short = np.maximum(GUARD - d, 0.0)
    pen = 1e6 * float(np.sum(short ** 2))
    if not grad:
        return -ev + pen
    Cb = cho_solve(fac, b)
    ba = b.conj().T @ alpha
    g_logw = w * (prob.M * np.abs(ba) ** 2 / Q - np.real(np.sum(b.conj() * Cb, axis=0)))
    trinv = float(np.real(np.trace(cho_solve(fac, np.eye(prob.M)))))
    g_logrho = rho * (prob.M * float(np.vdot(alpha, alpha).real) / Q - trinv)
    a_db = np.einsum('m,mkc->kc', alpha.conj(), db)
    tr = np.real(np.einsum('mk,mkc->kc', Cb.conj(), db))
    g_S = w[:, None] * (prob.M * 2 * np.real(a_db * ba[:, None]) / Q - 2 * tr)
    diff = S[None, :, :] - prob.x[:, None, :]
    g_pen = np.einsum('mk,mkc->kc', -2e6 * short / np.maximum(d, 1e-12), diff)
    gradient = np.concatenate((-g_S.ravel() + g_pen.ravel(), -g_logw, [-g_logrho]))
    return -ev + pen, gradient


def source_predict(prob, S, w, rho, q=None):
    S = np.atleast_2d(S)
    q = prob.q if q is None else q
    b, _, nu = basis_grad(prob.x, S, prob.k)
    C = prob.H + (b * w[None, :]) @ b.conj().T + (rho + JITTER) * np.eye(prob.M)
    alpha = cho_solve(cho_factor(C, lower=True), prob.y)
    hq = prob.Hq if q is prob.q else sinc_k(prob.k, q, prob.x)
    bq = target_basis(q, S, prob.k, nu)
    return hq @ alpha + bq @ (w * (b.conj().T @ alpha))


def grid_nodes(lo, hi, spacing, max_nodes=12000):
    while True:
        axes = [np.linspace(a, b, max(2, int(np.ceil((b - a) / spacing)) + 1)) for a, b in zip(lo, hi)]
        n = np.prod([len(a) for a in axes])
        if n <= max_nodes:
            return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
        spacing *= 1.1


def _scan(prob, nodes, S, w, rho, starts, suppress):
    """Rank-one evidence of adding one point source at each node on top of the current model."""
    if len(S):
        b, _, _ = basis_grad(prob.x, S, prob.k)
        C = prob.H + (b * w[None, :]) @ b.conj().T + (rho + JITTER) * np.eye(prob.M)
    else:
        C = prob.H + (rho + JITTER) * np.eye(prob.M)
    fac = cho_factor(C, lower=True, check_finite=False)
    alpha = cho_solve(fac, prob.y)
    Q = float(np.vdot(prob.y, alpha).real)
    logdet = 2 * float(np.sum(np.log(np.diag(fac[0]).real)))
    d = cdist(prob.x, nodes)
    valid = d.min(axis=0) > GUARD
    nodes, d = nodes[valid], d[:, valid]
    g = np.exp(1j * prob.k * d) / d
    B = g / np.sqrt(np.mean(1 / d ** 2, axis=0))[None, :]
    CB = cho_solve(fac, B)
    bCb = np.real(np.sum(B.conj() * CB, axis=0))
    ba2 = np.abs(B.conj().T @ alpha) ** 2
    best = np.full(len(nodes), -np.inf); bw = np.zeros(len(nodes))
    for wm in W_MENU:
        den = 1 + wm * bCb
        ev = -prob.M * np.log(np.maximum(Q - wm * ba2 / den, 1e-300)) - logdet - np.log(den)
        better = ev > best
        best[better], bw[better] = ev[better], wm
    order = np.argsort(-best)
    chosen = []
    for i in order:
        if all(np.linalg.norm(nodes[i] - nodes[j]) > suppress for j in chosen):
            chosen.append(i)
        if len(chosen) == starts:
            break
    return [(nodes[i], bw[i], best[i]) for i in chosen]


def _refine(prob, theta0, K, lo, hi, maxiter):
    bounds = [(lo[c], hi[c]) for _ in range(K) for c in range(3)] + [LOGW] * K + [LOGRHO]
    theta0 = np.clip(theta0, [b[0] for b in bounds], [b[1] for b in bounds])
    res = minimize(source_objective, theta0, args=(prob, K), jac=True, method='L-BFGS-B', bounds=bounds,
                   options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-6))
    return res


def fit_sources(prob, kmax=3, margin=None, inside_room=False, spacing_wl=0.25, starts=3, maxiter=200,
                fixed_menu=False):
    """Greedy sequential localisation; returns the fitted model for every order K = 0..kmax.

    fixed_menu=True reproduces the finite-menu tuning of the earlier registered estimator (weights {0.3, 1, 3},
    ridges {3e-4, 3e-3, 3e-2, 0.3}; Nelder-Mead on coordinates only), used for the continuous-tuning comparison.
    """
    lo, hi = prob.search_box(margin, inside_room)
    nodes = grid_nodes(lo, hi, spacing_wl * prob.lam, max_nodes=20000)
    k0 = sinc_fit(prob, 'evidence')
    models = {0: dict(S=np.zeros((0, 3)), w=np.zeros(0), rho=k0['rho'], evidence=k0['evidence'], converged=True)}
    S, w, rho = np.zeros((0, 3)), np.zeros(0), k0['rho']
    for K in range(1, kmax + 1):
        cands = _scan(prob, nodes, S, w, rho, starts, 0.5 * prob.lam)
        best = None
        for node, wn, _ in cands:
            if fixed_menu:
                res = _menu_refine(prob, np.vstack((S, node[None, :])), lo, hi)
            else:
                theta0 = np.concatenate((np.vstack((S, node[None, :])).ravel(), np.log(np.r_[w, wn]), [np.log(rho)]))
                r = _refine(prob, theta0, K, lo, hi, maxiter)
                res = dict(theta=r.x, evidence=-float(r.fun), converged=bool(r.success), nit=int(r.nit))
            if best is None or res['evidence'] > best['evidence']:
                best = res
        S, w, rho = _unpack(best['theta'], K)
        models[K] = dict(S=S.copy(), w=w.copy(), rho=rho, evidence=best['evidence'], converged=best['converged'])
    return models


MENU_W, MENU_R = (0.3, 1.0, 3.0), (0.0003, 0.003, 0.03, 0.3)


def _menu_refine(prob, S0, lo, hi):
    K = len(S0)
    best = None
    for wv in MENU_W:
        for r in MENU_R:
            def obj(s):
                th = np.concatenate((s, np.full(K, np.log(wv)), [np.log(r)]))
                if np.any(s.reshape(K, 3) < lo) or np.any(s.reshape(K, 3) > hi):
                    return 1e30
                return source_objective(th, prob, K, grad=False)
            res = minimize(obj, S0.ravel(), method='Nelder-Mead', options=dict(maxfev=240, xatol=1e-4, fatol=1e-8))
            ev = -float(res.fun)
            if best is None or ev > best['evidence']:
                best = dict(theta=np.concatenate((res.x, np.full(K, np.log(wv)), [np.log(r)])), evidence=ev,
                            converged=bool(res.success))
    return best


def bic_order(models, M):
    """Penalised evidence: four real parameters (three coordinates, one weight) per source, 2M real data."""
    score = {K: m['evidence'] - 2.0 * K * np.log(2 * M) for K, m in models.items()}
    return max(score, key=score.get), score


def cv_order(prob, kmax, folds, **kw):
    """Microphone cross-validation of the model order: every fold refits localisation and hyperparameters."""
    err = np.zeros(kmax + 1)
    for held in folds:
        train = np.setdiff1d(np.arange(prob.M), held)
        sub = prob.subset(train)
        models = fit_sources(sub, kmax, **kw)
        for K, m in models.items():
            pred = source_predict(sub, m['S'], m['w'], m['rho'], q=prob.x[held]) if K else \
                sinc_k(sub.k, prob.x[held], sub.x) @ np.linalg.solve(sub.H + (m['rho'] + JITTER) * np.eye(sub.M), sub.y)
            err[K] += float(np.sum(np.abs(prob.y[held] - pred) ** 2))
    err /= float(np.vdot(prob.y, prob.y).real)
    return int(np.argmin(err)), err


def make_folds(M, n_folds, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7771]))
    perm = rng.permutation(M)
    return [np.sort(f) for f in np.array_split(perm, n_folds)]


def model_prediction(prob, m):
    if len(m['S']) == 0:
        lam, U = prob.eig()
        alpha = U @ ((U.conj().T @ prob.y) / (lam + m['rho']))
        return prob.Hq @ alpha
    return source_predict(prob, m['S'], m['w'], m['rho'])


def oracle_fit(prob, sources, maxiter=200):
    """Reference only: point sources at the given true coordinates; weights and ridge tuned by evidence."""
    S = np.atleast_2d(sources)
    K = len(S)
    best = None
    for wn in W_MENU:
        def obj(t):
            f, g = source_objective(np.concatenate((S.ravel(), t)), prob, K)
            return f, g[3 * K:]
        r = minimize(obj, np.r_[np.full(K, np.log(wn)), np.log(1e-2)], jac=True, method='L-BFGS-B',
                     bounds=[LOGW] * K + [LOGRHO], options=dict(maxiter=maxiter))
        if best is None or r.fun < best.fun:
            best = r
    w, rho = np.exp(best.x[:K]), float(np.exp(best.x[K]))
    return dict(S=S, w=w, rho=rho, evidence=-float(best.fun), pred=source_predict(prob, S, w, rho))


# ------------------------------------------------------------------ isotropic RBF (Gaussian) kernel

def rbf_objective(t, prob):
    ell, rho = np.exp(t[0]), np.exp(t[1])
    D2 = cdist(prob.x, prob.x) ** 2
    Kx = np.exp(-0.5 * D2 / ell ** 2)
    C = Kx + (rho + 1e-8) * np.eye(prob.M)
    try:
        fac = cho_factor(C, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return 1e30, np.zeros(2)
    alpha = cho_solve(fac, prob.y)
    Q = float(np.vdot(prob.y, alpha).real)
    ev = -prob.M * np.log(Q) - 2 * float(np.sum(np.log(np.diag(fac[0]).real)))
    Ci = cho_solve(fac, np.eye(prob.M))
    dK = Kx * D2 / ell ** 2
    g_ell = prob.M * float(np.real(alpha.conj() @ dK @ alpha)) / Q - float(np.real(np.sum(Ci * dK.T)))
    g_rho = rho * (prob.M * float(np.vdot(alpha, alpha).real) / Q - float(np.real(np.trace(Ci))))
    return -ev, -np.array([g_ell, g_rho])


def rbf_fit(prob):
    """Isotropic Gaussian kernel, length scale and ridge by multi-start marginal likelihood (Caviedes-Nozal 2021)."""
    best = None
    for ell0 in (0.05, 0.1, 0.2, 0.4):
        for r0 in (1e-4, 1e-2):
            r = minimize(rbf_objective, [np.log(ell0 * prob.lam), np.log(r0)], args=(prob,), jac=True,
                         method='L-BFGS-B', bounds=[(np.log(0.01 * prob.lam), np.log(5 * prob.lam)), LOGRHO])
            if best is None or r.fun < best.fun:
                best = r
    ell, rho = np.exp(best.x)
    Kx = np.exp(-0.5 * cdist(prob.x, prob.x) ** 2 / ell ** 2) + (rho + 1e-8) * np.eye(prob.M)
    alpha = np.linalg.solve(Kx, prob.y)
    pred = np.exp(-0.5 * cdist(prob.q, prob.x) ** 2 / ell ** 2) @ alpha
    return dict(method='rbf', ell=float(ell), rho=float(rho), evidence=-float(best.fun), pred=pred)


# ------------------------------------------------------------------ sparse Bayesian learning over equivalent sources

def sbl_dictionary(prob, spacing_wl=1 / 3, n_dirs=None, max_nodes=5000):
    """Monopoles on a lambda/3 grid over the region box enlarged by one wavelength, plus plane waves."""
    lo, hi = prob.search_box(prob.lam)
    nodes = grid_nodes(lo, hi, spacing_wl * prob.lam, max_nodes)
    nodes = nodes[cdist(prob.x, nodes).min(axis=0) > GUARD]
    R = 0.5 * float(np.linalg.norm(prob.box[1] - prob.box[0]))
    if n_dirs is None:
        N = int(np.clip(np.ceil(prob.k * R) + 1, 3, 24))
        n_dirs = 2 * (N + 1) ** 2
    from fields import fibonacci_sphere
    dirs = fibonacci_sphere(n_dirs)
    return nodes, dirs


def sbl_fit(prob, rule='mackay', max_iter=None, tol=1e-4):
    """Single-snapshot SBL over monopoles on a grid plus plane waves.

    rule 'mackay': Tipping (2001) fixed-point updates run to convergence (at most 1000 iterations).
    rule 'em': Wipf & Rao (2004) expectation-maximisation updates stopped at 200 iterations.
    """
    if max_iter is None:
        max_iter = 1000 if rule == 'mackay' else 200
    from fields import fibonacci_sphere  # noqa: F401  (dictionary helper lives with the forward models)
    nodes, dirs = sbl_dictionary(prob)
    g = green(prob.x, nodes, prob.k)
    nu = np.sqrt(np.mean(np.abs(g) ** 2, axis=0))
    Phi = np.hstack((g / nu[None, :], np.exp(1j * prob.k * prob.x @ dirs.T)))
    N = Phi.shape[1]
    yy = float(np.vdot(prob.y, prob.y).real) / prob.M
    gamma = np.full(N, yy / N * 10)
    s2 = 0.01 * yy
    active = np.arange(N)
    it = 0
    for it in range(max_iter):
        P = Phi[:, active]
        ga = gamma[active]
        Sy = s2 * np.eye(prob.M) + (P * ga[None, :]) @ P.conj().T
        fac = cho_factor(Sy, lower=True, check_finite=False)
        iy = cho_solve(fac, prob.y)
        mu = ga * (P.conj().T @ iy)
        SP = cho_solve(fac, P)
        diagS = ga - ga ** 2 * np.real(np.sum(P.conj() * SP, axis=0))
        resid = prob.y - P @ mu
        r2 = float(np.vdot(resid, resid).real)
        used = float(np.sum(1 - diagS / ga))
        if rule == 'em':
            new = np.abs(mu) ** 2 + diagS
            s2 = max((r2 + s2 * used) / prob.M, 1e-12 * yy)
        else:
            new = np.abs(mu) ** 2 / np.maximum(1 - diagS / ga, 1e-12)
            s2 = max(r2 / max(prob.M - used, 1.0), 1e-12 * yy)
        change = np.max(np.abs(new - ga)) / max(np.max(ga), 1e-300)
        gamma[active] = new
        active = active[gamma[active] > 1e-8 * gamma[active].max()]
        if change < tol:
            break
    P = Phi[:, active]
    Sy = s2 * np.eye(prob.M) + (P * gamma[active][None, :]) @ P.conj().T
    mu = gamma[active] * (P.conj().T @ np.linalg.solve(Sy, prob.y))
    nn = len(nodes)
    mono = active[active < nn]
    pw = active[active >= nn] - nn
    pred = np.zeros(len(prob.q), complex)
    if len(mono):
        pred += (green(prob.q, nodes[mono], prob.k) / nu[mono][None, :]) @ mu[:len(mono)]
    if len(pw):
        pred += np.exp(1j * prob.k * prob.q @ dirs[pw].T) @ mu[len(mono):]
    return dict(method='sbl-' + rule, pred=pred, iterations=it + 1, active=int(len(active)), noise=float(s2),
                dictionary=int(N))


# ------------------------------------------------------------------ adaptive directional (plane-wave-weight) kernel

def real_sph_harm(L, dirs):
    from scipy.special import sph_harm_y
    theta = np.arccos(np.clip(dirs[:, 2], -1, 1))
    phi = np.arctan2(dirs[:, 1], dirs[:, 0])
    cols = []
    for n in range(L + 1):
        for m in range(-n, n + 1):
            Y = sph_harm_y(n, abs(m), theta, phi)
            cols.append(np.sqrt(2) * (-1) ** m * (Y.imag if m < 0 else Y.real) if m != 0 else Y.real)
    return np.array(cols).T


ADK_SCALE = 0.05   # optimiser variable scaling of the log-weight coefficients (conditioning only)


def adk_objective(t, prob, E, Yd):
    c, rho = t[:-1] * ADK_SCALE, float(np.exp(t[-1]))
    u = np.clip(Yd @ c, -30.0, 30.0)
    gam = np.exp(u) / E.shape[1]
    C = (E * gam[None, :]) @ E.conj().T + (rho + JITTER) * np.eye(prob.M)
    try:
        fac = cho_factor(C, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return 1e30, np.zeros_like(t)
    alpha = cho_solve(fac, prob.y)
    Q = float(np.vdot(prob.y, alpha).real)
    ev = -prob.M * np.log(Q) - 2 * float(np.sum(np.log(np.diag(fac[0]).real)))
    CE = cho_solve(fac, E)
    du = gam * (prob.M * np.abs(E.conj().T @ alpha) ** 2 / Q - np.real(np.sum(E.conj() * CE, axis=0)))
    trinv = float(np.real(np.trace(cho_solve(fac, np.eye(prob.M)))))
    g_rho = rho * (prob.M * float(np.vdot(alpha, alpha).real) / Q - trinv)
    pen = 0.5 * float(c[1:] @ c[1:]) / 4.0          # weak Gaussian prior (variance 4) on non-constant log-weight terms
    g_pen = np.r_[0.0, c[1:] / 4.0]
    return -ev + pen, np.r_[(-(Yd.T @ du) + g_pen) * ADK_SCALE, -g_rho]


def adk_fit(prob, L=4):
    """Source-free kernel with a smooth adaptive plane-wave weighting gamma(eta) = exp(sum_lm c_lm Y_lm(eta)).

    A reimplementation in the spirit of the adaptive physics-constrained kernels (Ribeiro et al. 2024), whose
    neural weighting is replaced by a low-order spherical-harmonic log-weight fitted by marginal likelihood.
    With c = 0 the kernel is the isotropic sinc kernel.
    """
    from fields import fibonacci_sphere
    R = 0.5 * float(np.linalg.norm(prob.box[1] - prob.box[0]))
    N = int(np.clip(np.ceil(prob.k * R) + 2, 4, 24))
    dirs = fibonacci_sphere(2 * (N + 1) ** 2)
    E = np.exp(1j * prob.k * prob.x @ dirs.T)
    Yd = real_sph_harm(L, dirs)
    s0 = sinc_fit(prob)
    best = None
    for r0 in (s0['rho'], 1e-2):
        t0 = np.r_[np.zeros(Yd.shape[1]), np.log(r0)]
        r = minimize(adk_objective, t0, args=(prob, E, Yd), jac=True, method='L-BFGS-B',
                     bounds=[(-8 / ADK_SCALE, 8 / ADK_SCALE)] * Yd.shape[1] + [LOGRHO], options=dict(maxiter=300))
        if best is None or r.fun < best.fun:
            best = r
    c, rho = best.x[:-1] * ADK_SCALE, float(np.exp(best.x[-1]))
    gam = np.exp(np.clip(Yd @ c, -30.0, 30.0)) / E.shape[1]
    C = (E * gam[None, :]) @ E.conj().T + (rho + JITTER) * np.eye(prob.M)
    alpha = np.linalg.solve(C, prob.y)
    Eq = np.exp(1j * prob.k * prob.q @ dirs.T)
    pred = (Eq * gam[None, :]) @ (E.conj().T @ alpha)
    return dict(method='adk', pred=pred, rho=rho, evidence=-float(best.fun), anisotropy=float(np.std(Yd @ c)))


# ------------------------------------------------------------------ source-region kernels (Matsuda & Otani style)

def _ball_nodes():
    """Fixed 33-node quadrature of the unit ball: centre plus two Fibonacci shells (radii 0.5, 0.85), volume weights."""
    from fields import fibonacci_sphere
    d = fibonacci_sphere(16)
    pts = np.vstack((np.zeros((1, 3)), 0.5 * d, 0.85 * d))
    # volume of the shells [0, 0.3], [0.3, 0.7], [0.7, 1] of the unit ball assigned to centre, inner, outer
    vol = np.array([0.3 ** 3, 0.7 ** 3 - 0.3 ** 3, 1 - 0.7 ** 3])
    w = np.r_[vol[0], np.full(16, vol[1] / 16), np.full(16, vol[2] / 16)]
    return pts, w / w.sum()


BALL, BALL_W = _ball_nodes()


def region_factor(points, centre, radius, k):
    """Columns G_n(points) sqrt(v_n) for the ball quadrature: kernel = F F^H (white source density in the ball)."""
    nodes = np.asarray(centre, float)[None, :] + radius * BALL
    return green(points, nodes, k) * np.sqrt(BALL_W)[None, :]


def region_objective(t, prob, J, fit_geometry, fixed=None):
    if fit_geometry:
        cen = t[:3 * J].reshape(J, 3); rad = np.exp(t[3 * J:4 * J]); lw = t[4 * J:5 * J]; rho = np.exp(t[5 * J])
    else:
        cen, rad = fixed; lw = t[:J]; rho = np.exp(t[J])
    C = prob.H + (rho + JITTER) * np.eye(prob.M)
    for j in range(J):
        if np.min(cdist(prob.x, cen[j:j + 1])) - rad[j] <= GUARD:
            return 1e30
        Fm = region_factor(prob.x, cen[j], rad[j], prob.k)
        nrm = np.mean(np.real(np.sum(np.abs(Fm) ** 2, axis=1)))
        C = C + np.exp(lw[j]) / nrm * (Fm @ Fm.conj().T)
    try:
        fac = cho_factor(C, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return 1e30
    alpha = cho_solve(fac, prob.y)
    return prob.M * np.log(float(np.vdot(prob.y, alpha).real)) + 2 * float(np.sum(np.log(np.diag(fac[0]).real)))


def region_predict(prob, cen, rad, lw, rho):
    C = prob.H + (rho + JITTER) * np.eye(prob.M)
    Cq = prob.Hq.astype(complex)
    for j in range(len(cen)):
        Fm = region_factor(prob.x, cen[j], rad[j], prob.k)
        Fq = region_factor(prob.q, cen[j], rad[j], prob.k)
        s = np.exp(lw[j]) / np.mean(np.real(np.sum(np.abs(Fm) ** 2, axis=1)))
        C = C + s * (Fm @ Fm.conj().T)
        Cq = Cq + s * (Fq @ Fm.conj().T)
    return Cq @ np.linalg.solve(C, prob.y)


def mo25_fit(prob, sources, radii=(0.05, 0.10, 0.15, 0.20)):
    """Reference with prior source information (reimplementation of Matsuda & Otani 2025): one ball region per
    given source with their Gaussian source density (sigma = R/3, truncated at R), plus the homogeneous sinc term
    their paper proposes as the extension to rooms. For each radius in their menu the region weights and the ridge
    are fitted by marginal likelihood; the radius is chosen by leave-one-out error, as their tuning does."""
    import mo2025 as MO
    cen = np.atleast_2d(sources); J = len(cen)
    best = None
    for R in radii:
        q = MO.ball_quadrature(R, prob.k)
        Fx = [MO.region_factor(prob.x, c, R, prob.k, q) for c in cen]
        Kx = [F_ @ F_.conj().T for F_ in Fx]
        Kx = [K_ / np.mean(np.real(np.diag(K_))) for K_ in Kx]

        def obj(t):
            C = prob.H + (np.exp(t[J]) + JITTER) * np.eye(prob.M)
            for j in range(J):
                C = C + np.exp(t[j]) * Kx[j]
            try:
                fac = cho_factor(C, lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                return 1e30
            a = cho_solve(fac, prob.y)
            return prob.M * np.log(float(np.vdot(prob.y, a).real)) + 2 * float(np.sum(np.log(np.diag(fac[0]).real)))
        r = min((minimize(obj, np.r_[np.full(J, lw0), np.log(1e-2)], method='L-BFGS-B', bounds=[LOGW] * J + [LOGRHO])
                 for lw0 in (0.0, 2.0)), key=lambda r: r.fun)
        C = prob.H + (np.exp(r.x[J]) + JITTER) * np.eye(prob.M)
        for j in range(J):
            C = C + np.exp(r.x[j]) * Kx[j]
        Ci = np.linalg.inv(C)
        a = Ci @ prob.y
        loo = float(np.sum(np.abs(a / np.real(np.diag(Ci))) ** 2))
        if best is None or loo < best[0]:
            best = (loo, R, r.x, Fx, q)
    _, R, t, Fx, q = best
    Cq = prob.Hq.astype(complex)
    for j in range(J):
        Fq = MO.region_factor(prob.q, cen[j], R, prob.k, q)
        nrm = np.mean(np.real(np.sum(np.abs(Fx[j]) ** 2, axis=1)))
        Cq = Cq + np.exp(t[j]) / nrm * (Fq @ Fx[j].conj().T)
    C = prob.H + (np.exp(t[J]) + JITTER) * np.eye(prob.M)
    for j in range(J):
        C = C + np.exp(t[j]) / np.mean(np.real(np.sum(np.abs(Fx[j]) ** 2, axis=1))) * (Fx[j] @ Fx[j].conj().T)
    return dict(method='mo25', pred=Cq @ np.linalg.solve(C, prob.y), radius=float(R))


def _region_parts(points, cen, rad, k, x_ref):
    """F (P, n) and its derivatives w.r.t. centre (P, n, 3) and radius (P, n); normaliser over x_ref and its derivatives."""
    nodes = cen[None, :] + rad * BALL
    sv = np.sqrt(BALL_W)

    def parts(p):
        diff = nodes[None, :, :] - p[:, None, :]
        d = np.linalg.norm(diff, axis=2)
        g = np.exp(1j * k * d) / d
        dg = (g * (1j * k - 1 / d))[..., None] * diff / d[..., None]      # d g / d node
        Fm = g * sv[None, :]
        dF_c = dg * sv[None, :, None]
        dF_a = np.einsum('pnc,nc->pn', dg, BALL) * sv[None, :]
        return Fm, dF_c, dF_a, d, diff
    Fx, dFx_c, dFx_a, d, diff = parts(x_ref)
    nrm = float(np.mean(np.sum(BALL_W[None, :] / d ** 2, axis=1)))
    dn_node = np.mean(-2 * BALL_W[None, :, None] * diff / d[..., None] ** 4, axis=0)   # (n, 3)
    dn_c = dn_node.sum(axis=0)
    dn_a = float(np.sum(dn_node * BALL))
    return parts, (Fx, dFx_c, dFx_a), nrm, dn_c, dn_a


def mo26_objective(t, prob, J):
    """Negative profiled evidence and analytic gradient for J ball regions (centres, log radii, log weights, log ridge)."""
    cen = t[:3 * J].reshape(J, 3); lrad = t[3 * J:4 * J]; lw = t[4 * J:5 * J]; rho = float(np.exp(t[5 * J]))
    rad = np.exp(lrad)
    C = prob.H + (rho + JITTER) * np.eye(prob.M)
    store = []
    pen, g_pen = 0.0, np.zeros(3 * J + J)
    for j in range(J):
        _, (Fx, dFc, dFa), nrm, dnc, dna = _region_parts(prob.x, cen[j], rad[j], prob.k, prob.x)
        sc = np.exp(lw[j]) / nrm
        C = C + sc * (Fx @ Fx.conj().T)
        store.append((Fx, dFc, dFa, nrm, dnc, dna, sc))
        dist = np.linalg.norm(prob.x - cen[j], axis=1) - rad[j]
        short = np.maximum(GUARD - dist, 0.0)
        pen += 1e6 * float(np.sum(short ** 2))
        u = (cen[j][None, :] - prob.x) / np.maximum(np.linalg.norm(prob.x - cen[j], axis=1), 1e-12)[:, None]
        g_pen[3 * j:3 * j + 3] += -2e6 * (short[:, None] * u).sum(axis=0)
        g_pen[3 * J + j] += 2e6 * float(np.sum(short)) * rad[j]
    try:
        fac = cho_factor(C, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return 1e30, np.zeros_like(t)
    alpha = cho_solve(fac, prob.y)
    Q = float(np.vdot(prob.y, alpha).real)
    ev = -prob.M * np.log(Q) - 2 * float(np.sum(np.log(np.diag(fac[0]).real)))
    trinv = float(np.real(np.trace(cho_solve(fac, np.eye(prob.M)))))
    grad = np.zeros_like(t)
    for j, (Fx, dFc, dFa, nrm, dnc, dna, sc) in enumerate(store):
        CF = cho_solve(fac, Fx)
        Fa = Fx.conj().T @ alpha                                   # (n,)
        quad = float(np.real(np.vdot(Fa, Fa)))                    # alpha^H F F^H alpha
        trK = float(np.real(np.sum(Fx.conj() * CF)))              # tr(C^-1 F F^H)
        # d/d log w
        grad[4 * J + j] = sc * (prob.M * quad / Q - trK)
        # centre
        aF_c = np.einsum('m,mnc->nc', alpha.conj(), dFc)          # alpha^H dF
        q_c = 2 * np.real(np.einsum('nc,n->c', aF_c, Fa))          # alpha^H (dF F^H + F dF^H) alpha
        t_c = 2 * np.real(np.einsum('mn,mnc->c', CF.conj(), dFc))  # tr(C^-1 (dF F^H + F dF^H))
        grad[3 * j:3 * j + 3] = sc * (prob.M * q_c / Q - t_c) - sc * dnc / nrm * (prob.M * quad / Q - trK)
        aF_a = alpha.conj() @ dFa
        q_a = 2 * float(np.real(aF_a @ Fa))
        t_a = 2 * float(np.real(np.sum(CF.conj() * dFa)))
        grad[3 * J + j] = rad[j] * (sc * (prob.M * q_a / Q - t_a) - sc * dna / nrm * (prob.M * quad / Q - trK))
    grad[5 * J] = rho * (prob.M * float(np.vdot(alpha, alpha).real) / Q - trinv)
    gvec = -grad
    gvec[:4 * J] += g_pen
    return -ev + pen, gvec


def mo26_fit(prob, J=3, radius0=0.05, maxiter=300, init_models=None, init='scan'):
    """Microphone-only multiple source regions (reimplementation from the abstract of Matsuda & Otani 2026):
    J ball regions with white source density whose centres, radii and weights are fitted jointly with the ridge by
    gradient ascent of the marginal likelihood (analytic gradients). Two initialisations are tried and the higher
    evidence kept: the J best separated nodes of the single-source evidence scan, and the J point sources of the
    greedy fit of the proposed pipeline when it is supplied (the strongest available microphone-only start).
    Regions whose weight is driven to the lower bound are effectively pruned.

    init='scan': the faithful, independent reimplementation (regions start at the J best separated nodes of the
    single-source evidence scan). init='greedy': the hybrid that starts the regions at the J points of the proposed
    greedy fit (`init_models`), i.e. a finite-extent refinement of the proposed localisation."""
    lo, hi = prob.search_box()
    s0 = sinc_fit(prob)
    inits = []
    if init == 'scan':
        nodes = grid_nodes(lo, hi, 0.25 * prob.lam, max_nodes=20000)
        st = _scan(prob, nodes, np.zeros((0, 3)), np.zeros(0), s0['rho'], J, 0.5 * prob.lam)
        J = len(st)
        inits.append((np.array([v for v, _, _ in st]), np.log(np.array([w for _, w, _ in st])), np.log(s0['rho'])))
    elif init == 'greedy':
        m = init_models[J]
        inits.append((m['S'], np.log(np.maximum(m['w'], 1e-3)), np.log(m['rho'])))
    else:
        raise ValueError(init)
    best = None
    bounds = [(lo[c], hi[c]) for _ in range(J) for c in range(3)] + [(np.log(0.005), np.log(0.5))] * J + [LOGW] * J + [LOGRHO]
    for cen0, lw0, lr0 in inits:
        t0 = np.r_[cen0.ravel(), np.full(J, np.log(radius0)), lw0, lr0]
        t0 = np.clip(t0, [b[0] for b in bounds], [b[1] for b in bounds])
        r = minimize(mo26_objective, t0, args=(prob, J), jac=True, method='L-BFGS-B', bounds=bounds,
                     options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    t = best.x
    cen = t[:3 * J].reshape(J, 3); rad = np.exp(t[3 * J:4 * J]); lw = t[4 * J:5 * J]; rho = float(np.exp(t[5 * J]))
    return dict(method='mo26-' + init, pred=region_predict(prob, cen, rad, lw, rho), evidence=-float(best.fun),
                centres=cen.tolist(), radii=rad.tolist(), weights=np.exp(lw).tolist(), converged=bool(best.success))
