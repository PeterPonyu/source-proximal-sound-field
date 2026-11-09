"""Post hoc reproduction of the published group-sparse configuration (PLAN-E-GROUP-SPARSE-REPRODUCTION.md).

Free field; 16 microphones uniformly on a circle of radius 1 m in the xy-plane; two monopoles G = exp(ikr)/(4 pi r) at
(0.37, 0, 0) and (-0.42, -0.4, 0) m with zero-mean circular Gaussian amplitudes drawn per trial and frequency; SNR 30 dB
per frequency; c = 340 m/s; 125 Hz to 4 kHz in 125 Hz steps; 500 evaluation points uniform in the disk outside 0.1 m
of either source; 20 trials (seeds 404100-404119).
Methods: the registered group-sparse adaptation (prox2.gsmf_prox, settings prox2.GSMF) with the paper's dictionary (planar
grid inside the 0.8 m circle at delta = 0.1 or 0.05 m) and all 32 bins as one group, for each zeta in the paper's grid;
the proposed estimator on each frequency alone and with coordinates shared across the 32; the sinc kernel.

    OPENBLAS_NUM_THREADS=1 nice -n 10 python3 -B posthoc/review3/reproduce_gsmf.py OUT --workers 4
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
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
sys.path.insert(0, str(S / 'common'))
import estimators as E  # noqa: E402
import v2  # noqa: E402
import prox2 as PX  # noqa: E402

C0 = 340.0
FREQS = np.arange(125.0, 4001.0, 125.0)
SEEDS = range(404100, 404120)
DELTAS = (0.1, 0.05)
ZETAS = tuple(10.0 ** l for l in range(-3, 3))
INNER = 0.8
SRC = np.array([[0.37, 0.0, 0.0], [-0.42, -0.4, 0.0]])
BOX = (np.array([-1.0, -1.0, 0.0]), np.array([1.0, 1.0, 0.0]))
N_MIC, N_EVAL, SNR_DB = 16, 500, 30.0
# Fig. 1 of the paper, "Proposed", read from the plot at each frequency (dB, about +-0.5 dB)
PUBLISHED = {
    '0.1': [-24.7, -23.7, -24.0, -22.4, -22.9, -23.2, -22.6, -19.8, -21.7, -14.3, -17.6, -17.7, -18.3, -17.9, -17.4, -17.0,
            -17.2, -17.9, -17.4, -16.3, -16.8, -14.4, -13.5, -13.4, -15.2, -14.3, -4.3, -10.7, -9.9, -9.0, -10.8, -9.9],
    '0.05': [-28.3, -27.1, -28.3, -27.0, -25.9, -27.4, -26.3, -23.3, -23.7, -15.3, -19.1, -20.0, -19.6, -19.7, -18.9, -18.4,
             -19.4, -18.3, -18.9, -17.8, -18.4, -15.0, -13.6, -14.5, -15.4, -13.9, -14.3, -10.8, -11.9, -9.4, -9.7, -6.7]}


def mics():
    a = 2 * np.pi * np.arange(N_MIC) / N_MIC
    return np.column_stack((np.cos(a), np.sin(a), np.zeros(N_MIC)))


def trial(seed):
    rg = np.random.default_rng(np.random.SeedSequence([seed, 7801]))
    q = []
    while len(q) < N_EVAL:
        p = np.r_[rg.uniform(-1, 1, 2), 0.0]
        if np.linalg.norm(p) <= 1.0 and np.all(np.linalg.norm(SRC - p, axis=1) >= 0.1):
            q.append(p)
    amps = (rg.normal(size=(len(FREQS), 2)) + 1j * rg.normal(size=(len(FREQS), 2))) / np.sqrt(2)
    return np.array(q), amps


def field(pts, f, a):
    k = 2 * np.pi * f / C0
    r = np.linalg.norm(pts[:, None, :] - SRC[None], axis=2)
    return (np.exp(1j * k * r) / (4 * np.pi * r)) @ a


def observe(seed):
    X = mics(); q, amps = trial(seed)
    Y, T = [], []
    for i, f in enumerate(FREQS):
        pm, pq = field(X, f, amps[i]), field(q, f, amps[i])
        rn = np.random.default_rng(np.random.SeedSequence([seed, 7802, i]))
        s2 = np.mean(np.abs(pm) ** 2) * 10 ** (-SNR_DB / 10)
        Y.append(pm + np.sqrt(s2 / 2) * (rn.normal(size=N_MIC) + 1j * rn.normal(size=N_MIC))); T.append(pq)
    return X, q, np.stack(Y, 1), np.stack(T, 1)


def planar_dictionary(delta):
    def dictionary(x, lo, hi, rho_factor):
        n = int(np.floor(INNER / delta + 1e-9))
        g = np.arange(-n, n + 1) * delta
        nodes = np.array([(a, b, 0.0) for a in g for b in g if a * a + b * b <= INNER ** 2 + 1e-9])
        rho0 = rho_factor * delta
        rad = np.minimum(rho0, cdist(nodes, x).min(axis=1) - PX.MIC_MARGIN)
        keep = rad >= PX.R_MIN
        return nodes[keep], rad[keep], rho0, delta
    return dictionary


def sq(pred, truth):
    return [float(np.sum(np.abs(pred - truth) ** 2)), float(np.sum(np.abs(truth) ** 2))]


def one(task):
    t0 = time.process_time()
    try:
        kind, seed, arg = task
        X, q, Y, T = observe(seed)
        out = dict(task=list(task))
        if kind == 'adapt':
            band = v2.Band(X, Y, q, FREQS, c=C0, box=BOX)
            real = PX.dictionary
            PX.dictionary = planar_dictionary(arg)
            try:
                prep = PX.Prepared(band, PX.GSMF)
            finally:
                PX.dictionary = real
            out['S'] = int(len(prep.cen)); out['shrunk'] = int(np.sum(prep.rad < prep.rho - 1e-12))
            out['zeta'] = {}
            for z in ZETAS:
                prep.cfg = dict(PX.GSMF, zeta=z)
                r = PX.gsmf_prox(prep)
                out['zeta'][repr(z)] = dict(sq=[sq(r['preds'][i], T[:, i]) for i in range(len(FREQS))], regions=r['info']['regions'])
        elif kind == 'bin':
            i = int(arg)
            band = v2.Band(X, Y[:, i:i + 1], q, FREQS[i:i + 1], c=C0, box=BOX)
            out['sinc'] = sq(E.sinc_fit(band.probs[0], 'evidence')['pred'], T[:, i])
            models = v2.fit_band(band, 5)
            K = v2.cv_order(band, 5, E.make_folds(band.M, 5, seed))[0]
            pred = v2.predict_points(band, models[0], 0) if K == 0 else v2.predict_regions(band, v2.extend(band, models[K], K), 0)
            out['proposed_bin'] = sq(pred, T[:, i]); out['K'] = int(K)
        else:
            band = v2.Band(X, Y, q, FREQS, c=C0, box=BOX)
            models = v2.fit_band(band, 5)
            K = v2.cv_order(band, 5, E.make_folds(band.M, 5, seed))[0]
            rm = v2.extend(band, models[K], K) if K > 0 else None
            out['proposed_shared'] = [sq(v2.predict_regions(band, rm, i) if rm else v2.predict_points(band, models[0], i), T[:, i])
                                      for i in range(len(FREQS))]
            out['K'] = int(K)
        out['seconds'] = round(time.process_time() - t0, 1)
        return out
    except Exception:
        return dict(task=list(task), error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    T = [('adapt', s, d) for d in DELTAS for s in SEEDS] + [('band', s, None) for s in SEEDS] + \
        [('bin', s, i) for s in SEEDS for i in range(len(FREQS) - 1, -1, -1)]
    if a.limit:
        T = T[:a.limit]
    (out / 'started.json').write_text(json.dumps(dict(tasks=len(T), workers=a.workers, published=PUBLISHED,
                                                      started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                                      inputs={n: hashlib.sha256((S / n).read_bytes()).hexdigest() for n in
                                                              ('common/prox2.py', 'common/v2.py', 'common/estimators.py',
                                                               'posthoc/review3/reproduce_gsmf.py')}), indent=1))
    t0 = time.time(); n_err = 0
    with Pool(a.workers) as pool, open(out / 'rows.jsonl', 'w') as fh:
        for n, row in enumerate(pool.imap_unordered(one, T), 1):
            n_err += 'error' in row
            fh.write(json.dumps(row) + '\n'); fh.flush()
            if n % 20 == 0 or n == len(T):
                print(f'{n}/{len(T)} tasks, {time.time() - t0:.0f} s, errors {n_err}', flush=True)
    print('done', n_err, 'errors')


if __name__ == '__main__':
    main()
