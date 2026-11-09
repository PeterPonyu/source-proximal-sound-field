"""Post hoc reproduction of the published configuration of the sparse region-kernel algorithm (PLAN-C-REPRODUCTION.md).

Free field; 36 microphones on a numerically constructed spherical 8-design of radius 1 m; one unit monopole
G = exp(ikr)/(4 pi r) uniform in the 0.4 m cube centred at (0.2, 0, 0) m; SNR 20 dB per frequency; 100-1500 Hz in
100 Hz steps; 500 evaluation points uniform in the sphere outside 0.1 m of the source; 20 trials (seeds 404000-404019).
Methods: the registered sparse adaptation with the paper's stage-2 stop (prox2_s2.mo26_prox_s2, prox2.MO26), its
dictionary replaced by the paper's grid in the 0.8 m sphere, at three grid spacings, with its stage-1 prediction too;
the proposed estimator on each frequency alone and with coordinates shared across the 15; the sinc kernel.

    OPENBLAS_NUM_THREADS=1 nice -n 10 python3 -B posthoc/review3/reproduce.py OUT --workers 3 [--limit N]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
sys.path.insert(0, str(S / 'common'))
import estimators as E  # noqa: E402
import v2  # noqa: E402
import prox2 as PX  # noqa: E402
import prox2_s2 as PS  # noqa: E402

C0 = 343.0
FREQS = np.arange(100.0, 1501.0, 100.0)
SEEDS = range(404000, 404020)
DELTAS = (0.15, 0.2, 0.25)
INNER = 0.8
BOX = (np.full(3, -1.0), np.full(3, 1.0))
N_MIC, T_DESIGN, N_EVAL = 36, 8, 500


def t_design():
    """A 36-point spherical 8-design: minimize sum_{n=1..8} (2n+1) sum_ij P_n(x_i . x_j) / N^2 from seeds 0, 1, ..."""
    def legendre(G):
        P = [np.ones_like(G), G]; D = [np.zeros_like(G), np.ones_like(G)]
        for n in range(1, T_DESIGN):
            P.append(((2 * n + 1) * G * P[n] - n * P[n - 1]) / (n + 1)); D.append(D[n - 1] + (2 * n + 1) * P[n])
        return P, D

    def f(v):
        V = v.reshape(N_MIC, 3); r = np.linalg.norm(V, axis=1); X = V / r[:, None]
        P, D = legendre(np.clip(X @ X.T, -1, 1))
        e = sum((2 * n + 1) * P[n].sum() for n in range(1, T_DESIGN + 1)) / N_MIC ** 2
        W = sum((2 * n + 1) * D[n] for n in range(1, T_DESIGN + 1)) / N_MIC ** 2
        gX = 2 * W @ X
        return e, ((gX - np.sum(gX * X, 1)[:, None] * X) / r[:, None]).ravel()
    for seed in range(1000):
        v = np.random.default_rng(seed).normal(size=N_MIC * 3)
        r = minimize(f, v, jac=True, method='L-BFGS-B', options=dict(maxiter=20000, ftol=1e-30, gtol=1e-16))
        X = r.x.reshape(N_MIC, 3); X /= np.linalg.norm(X, axis=1)[:, None]
        P, _ = legendre(np.clip(X @ X.T, -1, 1))
        resid = max(abs(P[n].sum()) / N_MIC ** 2 for n in range(1, T_DESIGN + 1))
        if resid < 1e-10:
            return X, dict(seed=seed, residual=float(resid))
    raise RuntimeError('no 8-design found')


def trial(seed):
    rg = np.random.default_rng(np.random.SeedSequence([seed, 7701]))
    src = np.array([0.2, 0.0, 0.0]) + rg.uniform(-0.2, 0.2, 3)
    q = []
    while len(q) < N_EVAL:
        p = rg.uniform(-1, 1, 3)
        if np.linalg.norm(p) <= 1.0 and np.linalg.norm(p - src) >= 0.1:
            q.append(p)
    return src, np.array(q)


def pressures(pts, src, f):
    k = 2 * np.pi * f / C0
    r = np.linalg.norm(pts - src, axis=1)
    return np.exp(1j * k * r) / (4 * np.pi * r)


def observe(X, src, q, f, seed):
    pm, pq = pressures(X, src, f), pressures(q, src, f)
    rn = np.random.default_rng(np.random.SeedSequence([seed, 7702, int(f)]))
    s2 = np.mean(np.abs(pm) ** 2) * 10 ** (-20 / 10)
    return pm + np.sqrt(s2 / 2) * (rn.normal(size=len(pm)) + 1j * rn.normal(size=len(pm))), pq


def paper_dictionary(delta):
    def dictionary(x, lo, hi, rho_factor):
        n = int(np.floor(INNER / delta + 1e-9))
        g = np.arange(-n, n + 1) * delta
        nodes = np.array([(a, b, c) for a in g for b in g for c in g if a * a + b * b + c * c <= INNER ** 2 + 1e-9])
        rho0 = rho_factor * delta
        rad = np.minimum(rho0, cdist(nodes, x).min(axis=1) - PX.MIC_MARGIN)
        keep = rad >= PX.R_MIN
        return nodes[keep], rad[keep], rho0, delta
    return dictionary


def sq(pred, truth):
    return float(np.sum(np.abs(pred - truth) ** 2)), float(np.sum(np.abs(truth) ** 2))


X_MIC, DESIGN = None, None


def init():
    global X_MIC, DESIGN
    X_MIC, DESIGN = t_design()


def one(task):
    t0 = time.process_time()
    try:
        kind, seed, f = task
        src, q = trial(seed)
        out = dict(task=list(task))
        if kind == 'bin':
            y, truth = observe(X_MIC, src, q, f, seed)
            band = v2.Band(X_MIC, y[:, None], q, np.array([f]), c=C0, box=BOX)
            p = band.probs[0]
            out['sinc'] = sq(E.sinc_fit(p, 'evidence')['pred'], truth)
            models = v2.fit_band(band, 5)
            K = v2.cv_order(band, 5, E.make_folds(band.M, 5, seed))[0]
            if K == 0:
                pred = v2.predict_points(band, models[0], 0)
            else:
                pred = v2.predict_regions(band, v2.extend(band, models[K], K), 0)
            out['proposed_bin'] = sq(pred, truth); out['proposed_bin_K'] = int(K)
            out['adaptation'] = {}
            for d in DELTAS:
                real = PX.dictionary
                PX.dictionary = paper_dictionary(d)
                try:
                    prep = PX.Prepared(band, PX.MO26)
                finally:
                    PX.dictionary = real
                cfg = prep.cfg
                W1 = PX.stage_weights([prep.Bases[0]], [prep.ys[0]], [prep.Ks[0]], cfg['zeta'], cfg['eta'], cfg['iters'], cfg['eps1'])
                stage1 = sq(prep.predict(0, prep.cen, prep.rad, W1[0]), truth)
                r = PS.mo26_prox_s2(prep, 0)
                out['adaptation'][str(d)] = dict(opt=sq(r['pred'], truth), stage1=stage1, S=int(len(prep.cen)),
                                                 shrunk=int(np.sum(prep.rad < prep.rho - 1e-12)), regions=r['info']['regions'],
                                                 after_stage1=r['info']['after_stage1'], iterations=r['info']['iterations'])
        else:
            Y, T = zip(*[observe(X_MIC, src, q, f_, seed) for f_ in FREQS])
            band = v2.Band(X_MIC, np.stack(Y, 1), q, FREQS, c=C0, box=BOX)
            models = v2.fit_band(band, 5)
            K = v2.cv_order(band, 5, E.make_folds(band.M, 5, seed))[0]
            rm = v2.extend(band, models[K], K) if K > 0 else None
            out['proposed_shared'] = [sq(v2.predict_regions(band, rm, i) if rm else v2.predict_points(band, models[0], i), T[i])
                                      for i in range(len(FREQS))]
            out['proposed_shared_K'] = int(K)
            if K > 0:
                out['proposed_shared_state'] = dict(centres=rm['centres'].tolist(), radii=rm['radii'].tolist(), source=src.tolist())
        out['seconds'] = round(time.process_time() - t0, 1)
        return out
    except Exception:
        return dict(task=list(task), error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--workers', type=int, default=3); ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    X, d = t_design()
    T = [('band', s, None) for s in SEEDS] + [('bin', s, float(f)) for s in SEEDS for f in FREQS[::-1]]
    if a.limit:
        T = T[:a.limit]
    (out / 'started.json').write_text(json.dumps(dict(tasks=len(T), workers=a.workers, design=d, microphones=X.tolist(),
                                                      started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                                      inputs={n: hashlib.sha256((S / n).read_bytes()).hexdigest() for n in
                                                              ('common/prox2.py', 'common/prox2_s2.py', 'common/v2.py', 'common/estimators.py',
                                                               'posthoc/review3/reproduce.py')}), indent=1))
    t0 = time.time(); n_err = 0
    with Pool(a.workers, initializer=init) as pool, open(out / 'rows.jsonl', 'w') as fh:
        for n, row in enumerate(pool.imap_unordered(one, T), 1):
            n_err += 'error' in row
            fh.write(json.dumps(row) + '\n'); fh.flush()
            if n % 20 == 0 or n == len(T):
                print(f'{n}/{len(T)} tasks, {time.time() - t0:.0f} s, errors {n_err}', flush=True)
    print('done', n_err, 'errors')


if __name__ == '__main__':
    main()
