"""The adaptive stand-in comparator (`estimators.adk_fit`) with a traced and with a corrected objective (PLAN.md, B1).

The registered objective clips the log weights, u = clip(Y c, -30, 30), but its gradient omits the derivative of the
clip. `adk_objective_traced` is the registered objective, unchanged in value and gradient; it also records, over every
evaluation, the largest |Y c| and whether any direction was clipped. `adk_objective_masked` multiplies the log-weight
gradient by the indicator |Y c| < 30, which is the derivative of the clipped objective away from the clip boundaries.
`adk_fit_variant` is `estimators.adk_fit` with the objective as an argument; everything else is unchanged.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
import estimators as E

CLIP = 30.0


class Trace:
    def __init__(self):
        self.n_eval = 0
        self.n_clipped = 0
        self.max_abs = 0.0

    def as_dict(self):
        return dict(n_eval=self.n_eval, n_clipped_evals=self.n_clipped, max_abs_Yc=self.max_abs)


def _objective(t, prob, Ed, Yd, mask, trace):
    c, rho = t[:-1] * E.ADK_SCALE, float(np.exp(t[-1]))
    Yc = Yd @ c
    if trace is not None:
        trace.n_eval += 1
        m = float(np.max(np.abs(Yc)))
        trace.max_abs = max(trace.max_abs, m)
        trace.n_clipped += int(m > CLIP)
    u = np.clip(Yc, -CLIP, CLIP)
    gam = np.exp(u) / Ed.shape[1]
    C = (Ed * gam[None, :]) @ Ed.conj().T + (rho + E.JITTER) * np.eye(prob.M)
    try:
        fac = cho_factor(C, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return 1e30, np.zeros_like(t)
    alpha = cho_solve(fac, prob.y)
    Q = float(np.vdot(prob.y, alpha).real)
    ev = -prob.M * np.log(Q) - 2 * float(np.sum(np.log(np.diag(fac[0]).real)))
    CE = cho_solve(fac, Ed)
    du = gam * (prob.M * np.abs(Ed.conj().T @ alpha) ** 2 / Q - np.real(np.sum(Ed.conj() * CE, axis=0)))
    if mask:
        du = du * (np.abs(Yc) < CLIP)
    trinv = float(np.real(np.trace(cho_solve(fac, np.eye(prob.M)))))
    g_rho = rho * (prob.M * float(np.vdot(alpha, alpha).real) / Q - trinv)
    pen = 0.5 * float(c[1:] @ c[1:]) / 4.0
    g_pen = np.r_[0.0, c[1:] / 4.0]
    return -ev + pen, np.r_[(-(Yd.T @ du) + g_pen) * E.ADK_SCALE, -g_rho]


def adk_objective_traced(t, prob, Ed, Yd, trace=None):
    return _objective(t, prob, Ed, Yd, False, trace)


def adk_objective_masked(t, prob, Ed, Yd, trace=None):
    return _objective(t, prob, Ed, Yd, True, trace)


def adk_fit_variant(prob, objective, L=4):
    """`estimators.adk_fit` with `objective`; returns the prediction, the trace and the final-state clip flag."""
    from fields import fibonacci_sphere
    R = 0.5 * float(np.linalg.norm(prob.box[1] - prob.box[0]))
    N = int(np.clip(np.ceil(prob.k * R) + 2, 4, 24))
    dirs = fibonacci_sphere(2 * (N + 1) ** 2)
    Ed = np.exp(1j * prob.k * prob.x @ dirs.T)
    Yd = E.real_sph_harm(L, dirs)
    s0 = E.sinc_fit(prob)
    trace = Trace()
    best = None
    for r0 in (s0['rho'], 1e-2):
        t0 = np.r_[np.zeros(Yd.shape[1]), np.log(r0)]
        r = minimize(objective, t0, args=(prob, Ed, Yd, trace), jac=True, method='L-BFGS-B',
                     bounds=[(-8 / E.ADK_SCALE, 8 / E.ADK_SCALE)] * Yd.shape[1] + [E.LOGRHO], options=dict(maxiter=300))
        if best is None or r.fun < best.fun:
            best = r
    c, rho = best.x[:-1] * E.ADK_SCALE, float(np.exp(best.x[-1]))
    Yc = Yd @ c
    gam = np.exp(np.clip(Yc, -CLIP, CLIP)) / Ed.shape[1]
    C = (Ed * gam[None, :]) @ Ed.conj().T + (rho + E.JITTER) * np.eye(prob.M)
    alpha = np.linalg.solve(C, prob.y)
    Eq = np.exp(1j * prob.k * prob.q @ dirs.T)
    pred = (Eq * gam[None, :]) @ (Ed.conj().T @ alpha)
    return dict(pred=pred, trace=trace.as_dict(), final_clipped=bool(np.max(np.abs(Yc)) > CLIP),
                final_max_abs_Yc=float(np.max(np.abs(Yc))), evidence=-float(best.fun))
