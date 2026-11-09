"""Post hoc: the proposed estimator's extension with a truncated Gaussian source density (PLAN-D-GAUSSIAN-EXTENSION.md).

The registered extension (`v2.extend`) and prediction (`v2.predict_regions`) are run unchanged, with the unit-ball rule
`estimators.BALL`, `BALL_W` (33 nodes, uniform volume weights) rebound to 48 nodes with Gaussian radial weights
(sigma = 1/3 of the radius, truncated at the radius): a 3-point Gauss rule in radius for r^2 exp(-r^2 / (2 sigma^2)) on
[0, 1] times the registered 16 Fibonacci directions. The objective, gradients, normalization, penalty, bounds, starts
and prediction are the registered code.

    OPENBLAS_NUM_THREADS=1 nice -n 10 python3 -B posthoc/review3/gauss_ext.py OUT --workers 20 [--check-only]
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
sys.path.insert(0, str(S / 'common')); sys.path.insert(0, str(S / 'V2-dev')); sys.path.insert(0, str(HERE))
import estimators as E  # noqa: E402
import mo2025 as MO  # noqa: E402
import v2  # noqa: E402
import refit as RF  # noqa: E402

GATE = 1e-12
SIGMA = 1 / 3


def radial_gauss(n=3, sigma=SIGMA, fine=4000):
    """n-point Gauss rule on [0, 1] for the weight r^2 exp(-r^2/(2 sigma^2)) (Stieltjes on a fine Gauss-Legendre
    discretization, then Golub-Welsch); weights sum to the integral of the weight."""
    x, w = np.polynomial.legendre.leggauss(fine)
    r = 0.5 * (x + 1); w = 0.5 * w * r ** 2 * np.exp(-r ** 2 / (2 * sigma ** 2))
    p_prev, p = np.zeros_like(r), np.ones_like(r)
    a, b = [], []
    nrm_prev = None
    for k in range(n):
        nrm = float(np.sum(w * p * p))
        a.append(float(np.sum(w * r * p * p)) / nrm)
        if k > 0:
            b.append(nrm / nrm_prev)
        p_new = (r - a[-1]) * p - (b[-1] if k > 0 else 0.0) * p_prev
        p_prev, p, nrm_prev = p, p_new, nrm
    J = np.diag(a) + np.diag(np.sqrt(b), 1) + np.diag(np.sqrt(b), -1)
    lam, V = np.linalg.eigh(J)
    return lam, V[0] ** 2 * float(np.sum(w))


def gaussian_nodes():
    from fields import fibonacci_sphere
    rr, ww = radial_gauss()
    d = fibonacci_sphere(16)
    pts = (rr[:, None, None] * d[None]).reshape(-1, 3)
    wt = np.repeat(ww / 16, 16)
    return pts, wt / wt.sum()


GBALL, GBALL_W = gaussian_nodes()


@contextlib.contextmanager
def gaussian_ball():
    old = E.BALL, E.BALL_W
    E.BALL, E.BALL_W = GBALL, GBALL_W
    try:
        yield
    finally:
        E.BALL, E.BALL_W = old


def checks():
    import run_dev as RD
    out = {}
    band, T, *_ = RD.sim_band('A2-f500-M24', 400920)
    p = band.probs[2]
    J = 2
    t = np.r_[np.array([[0.6, 0.6, -0.5], [1.5, 0.2, -0.4]]).ravel(), np.log([0.12, 0.2]), [0.5, -0.3], np.log(1e-2)]
    with gaussian_ball():
        g = E.mo26_objective(t, p, J)[1]
        fd = np.empty_like(t); h = 1e-6
        for i in range(len(t)):
            tp, tm = t.copy(), t.copy(); tp[i] += h; tm[i] -= h
            fd[i] = (E.mo26_objective(tp, p, J)[0] - E.mo26_objective(tm, p, J)[0]) / (2 * h)
    out['i_gradient_rel_err'] = float(np.max(np.abs(g - fd)) / np.max(np.abs(fd)))
    rng = np.random.default_rng(11)
    acc = []
    for R, f in ((0.05, 500.0), (0.1, 1000.0), (0.2, 500.0), (0.3, 500.0), (0.3, 1000.0), (0.15, 1000.0)):
        k = 2 * np.pi * f / 343.0
        for clear in (0.05, 0.15, 0.6):
            dirs = rng.standard_normal((12, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
            x = dirs * (R + clear)
            c = np.zeros(3)
            res = {}
            for dens, nodes in (('gauss', None), ('uniform', None)):
                nd, w = MO.ball_quadrature(R, k, n_r=40, n_a=3000, density=dens)
                Fd = MO.green4pi(x, c[None] + nd, k) * np.sqrt(w)
                Kd = Fd @ Fd.conj().T; Kd /= np.mean(np.real(np.diag(Kd)))
                if dens == 'gauss':
                    with gaussian_ball():
                        Fq = E.region_factor(x, c, R, k)
                else:
                    Fq = E.region_factor(x, c, R, k)
                Kq = Fq @ Fq.conj().T; Kq /= np.mean(np.real(np.diag(Kq)))
                res[dens] = float(np.linalg.norm(Kq - Kd) / np.linalg.norm(Kd))
            acc.append(dict(R=R, f=f, kR=k * R, clearance=clear, gauss48_rel_err=res['gauss'], uniform33_rel_err=res['uniform']))
    out['ii_accuracy'] = acc
    out['nodes'] = dict(n=int(len(GBALL)), radii=sorted({round(float(np.linalg.norm(v)), 6) for v in GBALL}))
    out['passed'] = bool(out['i_gradient_rel_err'] < 1e-6)
    return out


def one(task):
    t0 = time.process_time()
    key, stored = task['key'], task['stored']
    try:
        band, T = RF.build(key)
        models = v2.fit_band(band, 5)
        perK = {K: [E.nmse(v2.predict_points(band, m, f), T[:, f]) for f in range(band.F)] for K, m in models.items()}
        gate = {f'j_K{K}': RF.relmax(v, stored['nmse'][f'j_K{K}']) for K, v in perK.items()}
        K = int(stored['K_cv'])
        row = dict(study=task['study'], key=key, K_cv=K, gate_rel=gate)
        if K == 0:
            row['j_cv_ext_gauss'] = perK[0]; row['radii'] = []
        else:
            with gaussian_ball():
                rm = v2.extend(band, models[K], K)
                row['j_cv_ext_gauss'] = [E.nmse(v2.predict_regions(band, rm, f), T[:, f]) for f in range(band.F)]
            row['radii'] = rm['radii'].tolist(); row['evidence'] = rm['evidence']
        row['seconds'] = round(time.process_time() - t0, 2)
        return row
    except Exception:
        return dict(study=task['study'], key=key, error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--workers', type=int, default=20); ap.add_argument('--check-only', action='store_true')
    a = ap.parse_args()
    out = Path(a.out)
    if a.check_only:
        print(json.dumps(checks(), indent=1)); return
    out.mkdir(parents=True, exist_ok=False)
    ck = checks()
    (out / 'known-answer.json').write_text(json.dumps(ck, indent=1))
    if not ck['passed']:
        raise SystemExit('known-answer checks failed')
    T = [t for t in RF.registered_rows() if t['key']['kind'] in ('mesh', 'sim')]
    import sim
    T = sorted(T, key=lambda t: (0 if t['key']['kind'] != 'sim' else 1,
                                 -(t['key']['M'] if t['key']['M'] else sim.cells()[t['key']['cell']]['M'])))
    (out / 'started.json').write_text(json.dumps(dict(tasks=len(T), workers=a.workers, started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                                      inputs={n: hashlib.sha256((S / n).read_bytes()).hexdigest() for n in
                                                              ('common/estimators.py', 'common/v2.py', 'posthoc/review3/gauss_ext.py',
                                                               'posthoc/review3/refit.py')}), indent=1))
    t0 = time.time(); n_err = 0
    with Pool(a.workers, initializer=RF.init) as pool, open(out / 'rows.jsonl', 'w') as fh:
        for n, row in enumerate(pool.imap_unordered(one, T), 1):
            n_err += 'error' in row
            fh.write(json.dumps(row) + '\n'); fh.flush()
            if n % 100 == 0 or n == len(T):
                print(f'{n}/{len(T)} tasks, {time.time() - t0:.0f} s, errors {n_err}', flush=True)
    print('done', n_err, 'errors')


if __name__ == '__main__':
    main()
