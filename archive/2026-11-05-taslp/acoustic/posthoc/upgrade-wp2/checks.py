"""WP2 checks E1-E4 (PLAN.md)."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.special import spherical_jn, spherical_yn
HERE = Path(__file__).resolve().parent; A = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(A / 'common')); sys.path.insert(0, str(A / 'V2-dev'))
import exact_ball as XB  # noqa: E402
import cubature_diag as CD  # noqa: E402

out = {}
# E1 radial closed forms vs quadrature
worst = 0.0
for t1 in (0.3, 2.0, 7.5):
    for n in range(0, 41, 5):
        for (u, v) in ((spherical_jn, spherical_jn), (spherical_yn, spherical_jn)):
            a = 0.2 * t1
            num = quad(lambda t: t * t * u(n, t) * v(n, t), a, t1, limit=400, epsabs=0, epsrel=1e-13)[0]
            ju, yu = XB._jy(n, np.array([a, t1]))
            U = ju if u is spherical_jn else yu
            V = ju if v is spherical_jn else yu
            L = XB._lommel(U, V, np.array([a, t1]), n)[n]
            cf = L[1] - L[0]
            if abs(num) > 1e-200:
                worst = max(worst, abs(cf - num) / abs(num))
out['E1_max_rel_err'] = worst
# E2 vs singularity-free reference
rng = np.random.default_rng(11)
c = np.zeros(3); res = []
for (f, R) in ((125.0, 0.05), (500.0, 0.2), (1000.0, 0.3), (1000.0, 0.45)):
    k = 2 * np.pi * f / 343.0
    d = rng.standard_normal((6, 3)); d /= np.linalg.norm(d, axis=1)[:, None]
    x = d * (R + rng.uniform(0.06, 0.6, 6))[:, None]
    for rq in (0.0, 0.3 * R, 0.8 * R, 0.999 * R, R, 1.001 * R, R + 0.01):
        dq = rng.standard_normal(3); dq /= np.linalg.norm(dq)
        q = c + rq * dq
        ex = XB.cross_K(q[None], x, c, R, k)[0]
        ref = XB.reference_cross_K(q, x, c, R, k)
        res.append(dict(f=f, R=R, kR=k * R, rq_over_R=rq / R, rel=float(np.linalg.norm(ex - ref) / np.linalg.norm(ref))))
out['E2_max_rel_err'] = max(r['rel'] for r in res); out['E2_cases'] = res
# E3 continuity across the surface
k = 2 * np.pi * 800 / 343.0; R = 0.3
x = np.array([[0.7, 0.1, -0.2], [0.0, 0.0, 0.9]])
dq = np.array([0.3, -0.5, 0.8]); dq /= np.linalg.norm(dq)
a1 = XB.cross_K((R * (1 - 1e-9) * dq)[None], x, c, R, k); a2 = XB.cross_K((R * (1 + 1e-9) * dq)[None], x, c, R, k)
out['E3_surface_jump_rel'] = float(np.linalg.norm(a1 - a2) / np.linalg.norm(a1))
# E4 agreement with the existing outside series beyond 2 cm
q = c + np.outer(R + np.array([0.03, 0.1, 0.4]), dq)
out['E4_vs_existing_series_rel'] = float(np.linalg.norm(XB.cross_K(q, x, c, R, k) - CD.cross_K(q, x, c, R, k)) / np.linalg.norm(CD.cross_K(q, x, c, R, k)))
out['pass'] = dict(E1=bool(out['E1_max_rel_err'] <= 1e-8), E2=bool(out['E2_max_rel_err'] <= 1e-4), E3=bool(out['E3_surface_jump_rel'] <= 1e-8),
                   E4=bool(out['E4_vs_existing_series_rel'] <= 1e-12))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if k != 'E2_cases'})
print(sorted(((round(r['rel'], 9), r['f'], r['R'], round(r['rq_over_R'], 3)) for r in res), reverse=True)[:5])
