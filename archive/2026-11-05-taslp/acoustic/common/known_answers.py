"""B1 known-answer tests for the TASLP estimator and forward-model modules. Development seeds 400000-400099 only.

Run: OPENBLAS_NUM_THREADS=1 python3 -B known_answers.py OUT.json
Every test records its measured value and tolerance; any failure exits non-zero.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(ROOT / 'archive/2026-10-29-studies/acoustic-estimator-robustness/common'))
import estimators as E  # noqa: E402
import fields as F  # noqa: E402
import walls as W  # noqa: E402  (frozen; imports frozen science as W.s)
s = W.s
RESULTS = []


def check(name, value, tol, kind='<='):
    ok = bool(value <= tol) if kind == '<=' else bool(value >= tol)
    RESULTS.append(dict(test=name, value=float(value), tolerance=float(tol), relation=kind, passed=ok))
    print(('PASS ' if ok else 'FAIL ') + f'{name}: {value:.3e} {kind} {tol:.1e}', flush=True)


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)) / np.linalg.norm(np.asarray(b)))


def test_forward():
    rng = np.random.default_rng(400010)
    room = s.EXTENT
    pts = rng.random((64, 3)) * room
    src = np.array([1.1, 0.9, 0.5])
    for R in (1.0, 0.8, 0.8 * np.exp(1j * np.pi / 2), -0.8):
        a = F.image_pressure(pts, src, 250.0, room, None, 20, air=False, rconst=R)
        b = W.image_green(pts, src, 250.0, 0.0, R, 20)
        check(f'image sum equals frozen walls.image_green, R={np.round(R, 3)}', rel(a, b), 1e-12)
    z_rigid = {w: 1e9 for w in ('x0', 'x1', 'y0', 'y1', 'z0', 'z1')}
    a = F.image_pressure(pts, src, 250.0, room, z_rigid, 20, air=False)
    b = W.image_green(pts, src, 250.0, 0.0, 1.0, 20)
    check('impedance wall with zeta -> infinity equals the rigid image sum', rel(a, b), 1e-6)
    # first-order images only: brute-force angle-dependent coefficient
    zeta = 3.0 + 1.0j
    a = F.image_pressure(pts, src, 250.0, room, zeta, 1, air=False)
    k = 2 * np.pi * 250 / F.C0
    ref = np.exp(1j * k * np.linalg.norm(pts - src, axis=1)) / np.linalg.norm(pts - src, axis=1)
    for ax in range(3):
        for wall in (0.0, room[ax]):
            im = src.copy(); im[ax] = 2 * wall - im[ax]
            d = np.linalg.norm(pts - im, axis=1); cos = np.abs(pts[:, ax] - im[ax]) / d
            ref = ref + (zeta * cos - 1) / (zeta * cos + 1) * np.exp(1j * k * d) / d
    check('first-order angle-dependent image sum equals brute force', rel(a, ref), 1e-12)
    # dipole driver: zero-reflection term equals the analytic direct field
    axis = np.array([0.6, 0.8, 0.0])
    a = F.image_pressure(pts, src, 500.0, room, zeta, 0, beta=0.3, axis=axis, air=True)
    b = F.direct_pressure(pts, src, 500.0, beta=0.3, axis=axis, air=True)
    check('zero-order image sum equals the analytic direct monopole-dipole field', rel(a, b), 1e-12)
    # dipole far field is cos(theta) times the monopole, with the documented near-field factor
    far = src + 50.0 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    pd = F.direct_pressure(far, src, 500.0, beta=1.0, axis=axis, air=False)
    pm = F.direct_pressure(far, src, 500.0, beta=0.0, axis=axis, air=False)
    check('dipole far field equals cos(theta) times monopole', float(np.max(np.abs(pd / pm - axis))), 2e-3)


def test_rigid_decomposition():
    """Box decomposition p = G_free + u_H: with a real wavenumber (wall reflection, no loss in the medium) the
    remainder is source-free in the box and the sinc kernel represents it; it cannot represent p with the source
    inside. The rigid modal sum with bulk loss is checked against the image sum (forward-model agreement only)."""
    room = s.EXTENT
    rng = np.random.default_rng(400020)
    x = s.LAYOUTS['sobol'][:96]
    q = rng.random((256, 3)) * room
    out = {}
    for name, src in (('centre', np.array([1.7, 1.2, 0.85])), ('near-wall', np.array([1.7, 1.2, 0.07]))):
        pts = np.vstack((x, q))
        q_ok = np.linalg.norm(q - src, axis=1) > 0.12
        p = W.modal_green_fast(pts, src, 250.0, 0.05, 180) / (4 * np.pi)
        pi = W.image_green(pts, src, 250.0, 0.05, 1.0, 60) / (4 * np.pi)
        check(f'rigid box modal sum equals image sum with bulk loss ({name})', rel(p, pi), 2e-3)
        pr = F.image_pressure(pts, src, 250.0, room, None, 40, air=False, rconst=0.8)
        d = np.linalg.norm(pts - src, axis=1)
        rem = pr - np.exp(1j * 2 * np.pi * 250.0 / F.C0 * d) / d
        prob = E.Problem(x, rem[:96], q[q_ok], 250.0, box=(np.zeros(3), room))
        e_rem = E.nmse(E.sinc_fit(prob)['pred'], rem[96:][q_ok])
        prob2 = E.Problem(x, pr[:96], q[q_ok], 250.0, box=(np.zeros(3), room))
        e_full = E.nmse(E.sinc_fit(prob2)['pred'], pr[96:][q_ok])
        check(f'sinc represents the source-free remainder, R = 0.8 ({name})', e_rem, 0.05)
        check(f'sinc cannot represent the full field with the source inside ({name})', e_full, 3 * e_rem, '>=')
        out[name] = dict(remainder=e_rem, full=e_full)
    return out


def test_frozen_agreement():
    """Our evidence and posterior mean equal the frozen science.Estimator for the same field and hyperparameters."""
    field = s.make_field(400030, model='analytic')
    est = s.estimator(250.0, 96, 1.0)
    prob = E.Problem(field['x'], field['y'], field['q'], 250.0, box=(np.zeros(3), s.EXTENT))
    ours = E.sinc_evidence_curve(prob, s.RIDGES)
    theirs = [u['evidence'] for u in est.unaware_scores(field) if u['kind'] == 'helmholtz']
    check('sinc evidence equals frozen Estimator.unaware_scores', float(np.max(np.abs(ours - np.array(theirs)))), 1e-8)
    theta = field['source'] + np.array([0.03, -0.02, 0.01])
    frozen = est.source_scores(field, theta)
    bf = s.basis(field['x'], theta, 250.0)[:, 0]
    _, _, nu = E.basis_grad(field['x'], theta, prob.k)
    bo = E.green(field['x'], theta, prob.k)[:, 0] / nu[0]
    scale = float(np.mean(np.abs(bf) ** 2) / np.mean(np.abs(bo) ** 2))
    errs, perr = [], []
    for sc in frozen:
        w_ours = sc['weight'] * scale
        th = np.r_[theta, np.log(w_ours), np.log(sc['ridge'])]
        ev = -E.source_objective(th, prob, 1, grad=False)
        jitter_ev = -E.source_objective(th, prob, 1, grad=False)
        errs.append(abs(ev - sc['evidence']) / abs(sc['evidence']))
        pred = E.source_predict(prob, theta, np.array([w_ours]), sc['ridge'])
        perr.append(rel(pred, sc['prediction']))
        del jitter_ev
    check('point-source evidence equals frozen source_scores (12 weight/ridge pairs, relative)', max(errs), 1e-6)
    check('point-source posterior mean equals frozen source_scores prediction', max(perr), 1e-6)


def test_recovery_and_gradients():
    rng = np.random.default_rng(400040)
    grid = F.region_grid([1.0, 1.0, 0.7])
    idx = rng.choice(len(grid), 96, replace=False)
    x, q = grid[idx], np.delete(grid, idx, 0)
    src = np.array([1.6, 1.6, 0.4])
    y = F.direct_pressure(x, src, 500.0, air=False); t = F.direct_pressure(q, src, 500.0, air=False)
    sc = np.sqrt(np.mean(np.abs(np.r_[y, t]) ** 2)); y, t = y / sc, t / sc
    prob = E.Problem(x, y, q, 500.0)
    m = E.fit_sources(prob, 3)
    check('exact recovery of a pure point-source field: NMSE (K = 1)', E.nmse(E.model_prediction(prob, m[1]), t), 1e-8)
    check('exact recovery: localisation error in wavelengths', float(np.linalg.norm(m[1]['S'][0] - src)) / prob.lam, 1e-4)
    check('exact recovery: penalised evidence selects K = 1', abs(E.bic_order(m, 96)[0] - 1), 0)
    for K in (1, 2):
        th = np.r_[(src + 0.05 * rng.normal(size=(K, 3))).ravel(), np.log(np.full(K, 2.0)), np.log(1e-3)]
        f0, g0 = E.source_objective(th, prob, K)
        n = len(th); num = np.zeros(n)
        for i in range(n):
            e = np.zeros(n); e[i] = 1e-6
            num[i] = (E.source_objective(th + e, prob, K, grad=False) - E.source_objective(th - e, prob, K, grad=False)) / 2e-6
        check(f'analytic evidence gradient equals finite differences (K = {K})', float(np.max(np.abs(num - g0)) / np.max(np.abs(g0))), 1e-5)
    # two equal sources: K = 2 recovers both
    s2 = np.array([0.9, 1.4, 0.45])
    y2 = y + F.direct_pressure(x, s2, 500.0, air=False) / sc; t2 = t + F.direct_pressure(q, s2, 500.0, air=False) / sc
    prob2 = E.Problem(x, y2, q, 500.0)
    m2 = E.fit_sources(prob2, 3)
    check('two-source field: K = 2 NMSE', E.nmse(E.model_prediction(prob2, m2[2]), t2), 1e-6)
    check('two-source field: penalised evidence selects K = 2', abs(E.bic_order(m2, 96)[0] - 2), 0)
    return prob, t, src


def test_comparators(prob, t, src):
    rng = np.random.default_rng(400050)
    x, q = prob.x, prob.q
    d = rng.normal(size=3); d /= np.linalg.norm(d)
    pw = lambda p: np.exp(1j * prob.k * p @ d)
    hom = E.Problem(x, pw(x), q, prob.f)
    e96 = E.nmse(E.sinc_fit(hom)['pred'], pw(q))
    check('sinc kernel recovers a noise-free plane wave (96 microphones)', e96, 2e-2)
    grid = F.region_grid([1.0, 1.0, 0.7]); idx = rng.choice(len(grid), 288, replace=False)
    dense = E.Problem(grid[idx], pw(grid[idx]), np.delete(grid, idx, 0), prob.f)
    check('sinc plane-wave error falls at least tenfold with 288 microphones', E.nmse(E.sinc_fit(dense)['pred'], pw(dense.q)), e96 / 10)
    low = E.Problem(x, np.exp(1j * 2 * np.pi * 125 / F.C0 * x @ d), q, 125.0)
    check('RBF recovers a smooth 125 Hz plane wave', E.nmse(E.rbf_fit(low)['pred'], np.exp(1j * 2 * np.pi * 125 / F.C0 * q @ d)), 1e-3)
    r = E.adk_fit(hom)
    check('adaptive kernel recovers a single plane wave (anisotropic weighting)', E.nmse(r['pred'], pw(q)), 1e-4)
    R = 0.5 * float(np.linalg.norm(hom.box[1] - hom.box[0]))
    N = int(np.clip(np.ceil(hom.k * R) + 2, 4, 24))
    dirs = F.fibonacci_sphere(2 * (N + 1) ** 2)
    Em = np.exp(1j * hom.k * x @ dirs.T)
    check('adaptive kernel with uniform weights equals the sinc kernel (quadrature)', float(np.abs((Em @ Em.conj().T) / len(dirs) - hom.H).max()), 5e-3)
    nodes, dirs = E.sbl_dictionary(prob)
    k = prob.k
    i1, i2 = rng.choice(len(nodes), 2, replace=False)
    g = lambda p, n: E.green(p, nodes[n], k)[:, 0]
    nu = lambda n: np.sqrt(np.mean(np.abs(E.green(x, nodes[n], k)) ** 2))
    ys = g(x, i1) / nu(i1) + 0.7 * g(x, i2) / nu(i2) + 0.5 * np.exp(1j * k * x @ dirs[3])
    ts = g(q, i1) / nu(i1) + 0.7 * g(q, i2) / nu(i2) + 0.5 * np.exp(1j * k * q @ dirs[3])
    sp = E.Problem(x, ys, q, prob.f)
    check('SBL (fixed point) recovers a dictionary-sparse field', E.nmse(E.sbl_fit(sp, 'mackay')['pred'], ts), 1e-4)
    check('prior-region reference recovers a point-source field', E.nmse(E.mo25_fit(prob, src)['pred'], t), 1e-4)
    check('multiple-region kernel recovers a point-source field', E.nmse(E.mo26_fit(prob)['pred'], t), 1e-3)
    J, th = 2, np.r_[(src + 0.05 * rng.normal(size=(2, 3))).ravel(), np.log([0.05, 0.08]), [1.0, 0.2], np.log(3e-3)]
    f0, g0 = E.mo26_objective(th, prob, J); n = len(th); num = np.zeros(n)
    for i in range(n):
        e = np.zeros(n); e[i] = 1e-6
        num[i] = (E.mo26_objective(th + e, prob, J)[0] - E.mo26_objective(th - e, prob, J)[0]) / 2e-6
    check('multiple-region analytic gradient equals finite differences', float(np.max(np.abs(num - g0)) / np.max(np.abs(g0))), 1e-5)
    R = 0.5 * float(np.linalg.norm(hom.box[1] - hom.box[0])); N = int(np.clip(np.ceil(hom.k * R) + 2, 4, 24))
    dd = F.fibonacci_sphere(2 * (N + 1) ** 2); Em = np.exp(1j * hom.k * x @ dd.T); Yd = E.real_sph_harm(4, dd)
    th = np.r_[0.5 * rng.normal(size=25), np.log(1e-2)]
    f0, g0 = E.adk_objective(th, hom, Em, Yd); n = len(th); num = np.zeros(n)
    for i in range(n):
        e = np.zeros(n); e[i] = 1e-6
        num[i] = (E.adk_objective(th + e, hom, Em, Yd)[0] - E.adk_objective(th - e, hom, Em, Yd)[0]) / 2e-6
    check('adaptive-kernel analytic gradient equals finite differences', float(np.max(np.abs(num - g0)) / np.max(np.abs(g0))), 1e-5)
    check('oracle recovers a point-source field', E.nmse(E.oracle_fit(prob, src)['pred'], t), 1e-10)


def timings():
    sys.path.insert(0, str(HERE))
    import sim
    import methods
    cells = sim.cells()
    out = {}
    for name in ('base', 'A2-f1000-M192', 'A2-f125-M24'):
        t0 = time.process_time(); fd = sim.build(cells[name], 400060); tb = time.process_time() - t0
        prob = E.Problem(fd['x'], fd['y'], fd['q'], fd['f'], c=fd['c_assumed'], box=fd['box'], room=fd['room'])
        _, _, tim = methods.evaluate(prob, fd['truth'], fd['sources'], 400060, room=fd['room'])
        out[name] = dict(build_seconds=tb, **{k: round(v, 3) for k, v in tim.items()})
        print(name, out[name], flush=True)
    return out


def main():
    t0 = time.time()
    test_forward()
    dec = test_rigid_decomposition()
    test_frozen_agreement()
    prob, t, src = test_recovery_and_gradients()
    test_comparators(prob, t, src)
    tim = timings()
    res = dict(passed=all(r['passed'] for r in RESULTS), n=len(RESULTS), tests=RESULTS, rigid_decomposition=dec,
               process_seconds_per_code_path=tim, wall_seconds=round(time.time() - t0, 1),
               note='Development seeds 400000-400099 only; no production field is generated.')
    Path(sys.argv[1]).write_text(json.dumps(res, indent=2))
    print('ALL PASS' if res['passed'] else 'FAILURES', res['n'])
    sys.exit(0 if res['passed'] else 1)


if __name__ == '__main__':
    main()
