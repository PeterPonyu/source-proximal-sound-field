"""Known-answer checks for adk_fix.py (PLAN.md, B1), run before any refit.

(i)   the traced objective equals estimators.adk_objective exactly, in value and gradient, at random states;
(ii)  the masked gradient agrees with central finite differences (relative error < 1e-6) at an interior state and at a
      saturated state with clipped directions;
(iii) the registered gradient at the saturated state, recorded (expected to fail (ii)).

    OPENBLAS_NUM_THREADS=1 python3 -B posthoc/review3/checks.py OUT.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
sys.path.insert(0, str(S / 'common')); sys.path.insert(0, str(HERE))
import estimators as E  # noqa: E402
import adk_fix as AF  # noqa: E402
from fields import fibonacci_sphere  # noqa: E402


def toy():
    rng = np.random.default_rng(20261107)
    x = rng.uniform(-0.5, 0.5, (12, 3))
    q = rng.uniform(-0.5, 0.5, (30, 3))
    k = 2 * np.pi * 400.0 / 343.0
    s = np.array([1.1, 0.3, -0.2])
    y = E.green(x, s[None, :], k)[:, 0] + 0.01 * (rng.normal(size=12) + 1j * rng.normal(size=12))
    p = E.Problem(x, y / np.sqrt(np.mean(np.abs(y) ** 2)), q, 400.0, 343.0)
    R = 0.5 * float(np.linalg.norm(p.box[1] - p.box[0]))
    N = int(np.clip(np.ceil(p.k * R) + 2, 4, 24))
    dirs = fibonacci_sphere(2 * (N + 1) ** 2)
    Ed = np.exp(1j * p.k * p.x @ dirs.T)
    Yd = E.real_sph_harm(4, dirs)
    return p, Ed, Yd


def fd(fun, t, h=1e-6):
    g = np.empty_like(t)
    for i in range(len(t)):
        tp, tm = t.copy(), t.copy(); tp[i] += h; tm[i] -= h
        g[i] = (fun(tp)[0] - fun(tm)[0]) / (2 * h)
    return g


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    out = {}
    p, Ed, Yd = toy()
    rng = np.random.default_rng(7)
    same = []
    for _ in range(5):
        t = np.r_[rng.normal(0, 20, Yd.shape[1]), np.log(1e-2)]
        a = E.adk_objective(t, p, Ed, Yd); b = AF.adk_objective_traced(t, p, Ed, Yd, AF.Trace())
        same.append(bool(a[0] == b[0] and np.array_equal(a[1], b[1])))
    out['i_traced_equals_registered'] = dict(states=len(same), identical=all(same))
    # interior state: small coefficients, no direction clipped
    t_in = np.r_[rng.normal(0, 5, Yd.shape[1]), np.log(1e-2)]
    assert np.max(np.abs(Yd @ (t_in[:-1] * E.ADK_SCALE))) < AF.CLIP
    g = AF.adk_objective_masked(t_in, p, Ed, Yd)[1]
    out['ii_interior'] = dict(max_abs_Yc=float(np.max(np.abs(Yd @ (t_in[:-1] * E.ADK_SCALE)))),
                              rel_err=rel(g, fd(lambda s: AF.adk_objective_masked(s, p, Ed, Yd), t_in)))
    # saturated state: c_l = 7.95 sign(Y_0l) inside the coefficient bounds, so the largest |Y c| exceeds the clip.
    # Three microphones keep the clipped covariance well conditioned, so finite differences are accurate there.
    r3 = np.random.default_rng(7)
    x3 = r3.uniform(-0.3, 0.3, (3, 3)); q3 = r3.uniform(-0.3, 0.3, (5, 3))
    y3 = r3.normal(size=3) + 1j * r3.normal(size=3)
    p3 = E.Problem(x3, y3, q3, 300.0, 343.0)
    R3 = 0.5 * float(np.linalg.norm(p3.box[1] - p3.box[0]))
    dirs3 = fibonacci_sphere(2 * (int(np.clip(np.ceil(p3.k * R3) + 2, 4, 24)) + 1) ** 2)
    Ed3 = np.exp(1j * p3.k * p3.x @ dirs3.T); Yd3 = E.real_sph_harm(4, dirs3)
    c = 7.95 * np.sign(Yd3[0]); c[c == 0] = 7.95
    t_sat = np.r_[c / E.ADK_SCALE, np.log(1e-2)]
    Yc = Yd3 @ c
    gam = np.exp(np.clip(Yc, -AF.CLIP, AF.CLIP)) / Ed3.shape[1]
    cond = float(np.linalg.cond((Ed3 * gam[None, :]) @ Ed3.conj().T + (1e-2 + E.JITTER) * np.eye(3)))
    gm = AF.adk_objective_masked(t_sat, p3, Ed3, Yd3)[1]
    gr = E.adk_objective(t_sat, p3, Ed3, Yd3)[1]
    gfd = fd(lambda s: AF.adk_objective_masked(s, p3, Ed3, Yd3), t_sat)
    out['ii_saturated'] = dict(microphones=3, directions=int(len(dirs3)), clipped_directions=int(np.sum(np.abs(Yc) > AF.CLIP)),
                               max_abs_Yc=float(np.max(np.abs(Yc))), min_margin_to_clip=float(np.min(np.abs(np.abs(Yc) - AF.CLIP))),
                               covariance_condition=cond, rel_err=rel(gm, gfd))
    out['iii_registered_at_saturated'] = dict(rel_err=rel(gr, gfd), note='recorded; expected to exceed the 1e-6 tolerance')
    out['passed'] = bool(out['i_traced_equals_registered']['identical'] and out['ii_interior']['rel_err'] < 1e-6
                         and out['ii_saturated']['clipped_directions'] > 0 and out['ii_saturated']['rel_err'] < 1e-6)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    sys.exit(0 if out['passed'] else 1)


if __name__ == '__main__':
    main()
