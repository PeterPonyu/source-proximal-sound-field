"""Known-answer and replication checks for version 3 (PLAN.md), run before any development output.

    OPENBLAS_NUM_THREADS=1 python3 -B upgrade-v3/checks.py OUT.json
"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(A / 'common')); sys.path.insert(0, str(A / 'posthoc' / 'review3'))
sys.path.insert(0, str(A / 'V2-dev'))
import v3  # noqa: E402
import v2  # noqa: E402
import estimators as E  # noqa: E402
import reproduce as RP  # noqa: E402
import run_dev as RD  # noqa: E402


def sphere_band(seed):
    src, q = RP.trial(seed)
    Y, T = zip(*[RP.observe(RP.X_MIC, src, q, f, seed) for f in RP.FREQS])
    return v2.Band(RP.X_MIC, np.stack(Y, 1), q, RP.FREQS, c=RP.C0, box=RP.BOX), T, src


def same_models(a, b, kmax):
    return all(np.array_equal(a[K]['S'], b[K]['S']) and np.array_equal(a[K]['W'], b[K]['W']) and np.array_equal(a[K]['R'], b[K]['R'])
               for K in range(kmax + 1))


def main():
    RP.init(); RD.init()
    out = {}
    # C1 replication switch: v3 with V2_OPTS == v2.fit_band, and the extension == v2.extend
    band, T, src = sphere_band(404008)
    out['C1_sphere_fit_band'] = same_models(v3.fit_band(band, 3, opts=v3.V2_OPTS), v2.fit_band(band, 3), 3)
    bs, Ts, _, _, _ = RD.sim_band('base', 400920)
    m2 = v2.fit_band(bs, 3); m3 = v3.fit_band(bs, 3, opts=v3.V2_OPTS)
    out['C1_sim_fit_band'] = same_models(m3, m2, 3)
    r2 = v2.extend(bs, m2[2], 2); r3 = v3.extend(bs, m3[2], 2, opts=v3.V2_OPTS)
    out['C1_sim_extend'] = bool(np.array_equal(r2['centres'], r3['centres']) and np.array_equal(r2['lw'], r3['lw']))
    folds = E.make_folds(bs.M, 5, 400920)
    out['C1_sim_cv'] = bool(v2.cv_order(bs, 3, folds)[0] == v3.cv_order(bs, 3, folds, opts=v3.V2_OPTS)[0])
    # C2 scaled minimisation returns the objective at the returned point
    th = m2[2]['theta']
    lo, hi = bs.search_box()
    r = v3._min(v2.joint_objective, th, v2._bounds_points(bs, 2, lo, hi), v3._point_scale(bs, 2, v3.V3_OPTS), 50, args=(bs, 2))
    out['C2_scaled_objective_consistent'] = bool(abs(r.fun - v2.joint_objective(r.x, bs, 2)[0]) < 1e-6 * abs(r.fun))
    out['C2_not_worse_than_start'] = bool(r.fun <= v2.joint_objective(th, bs, 2)[0] + 1e-9)
    # C3 the closed-form weight reset of a single point never raises the objective at its own coordinate
    th1 = m2[1]['theta'].copy(); th1[3:3 + bs.F] = v2.LOGW2[0]   # weights at the bound
    v_before = v2.joint_objective(th1, bs, 1)[0]
    v_after = v2.joint_objective(v3.reset_weights(bs, th1, 1), bs, 1)[0]
    out['C3_reset_lowers_objective'] = bool(v_after < v_before)
    # C4 spherical trial 404008, K = 1: version 3 fails in no bin (version 2 failed in 3)
    m = v3.fit_band(band, 1)
    db = [10 * np.log10(E.nmse(v2.predict_points(band, m[1], i), T[i])) for i in range(band.F)]
    out['C4_sphere_404008_K1_max_db'] = float(max(db))
    out['C4_pass'] = bool(max(db) < -10)
    out['all_pass'] = all(v for k, v in out.items() if k.startswith('C') and isinstance(v, bool))
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
