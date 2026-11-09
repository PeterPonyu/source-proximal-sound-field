"""Known-answer checks of the version-2 module (development seeds 400920-400923 only).

python3 -B v2_checks.py OUT.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import estimators as E  # noqa: E402
import v2  # noqa: E402

C0 = 343.0


def box_points(rng, n, origin=np.zeros(3), ext=np.array([1.2, 1.2, 1.05])):
    return origin + rng.random((n, 3)) * ext


def band_field(x, q, freqs, sources, amps, noise=0.0, rng=None):
    Y, T = [], []
    for f in freqs:
        k = 2 * np.pi * f / C0
        y = sum(a * E.green(x, s, k)[:, 0] for s, a in zip(sources, amps))
        t = sum(a * E.green(q, s, k)[:, 0] for s, a in zip(sources, amps))
        sc = np.sqrt(np.mean(np.abs(y) ** 2))
        y, t = y / sc, t / sc
        if noise:
            y = y + noise * (rng.normal(size=len(y)) + 1j * rng.normal(size=len(y))) / np.sqrt(2)
        Y.append(y); T.append(t)
    return np.stack(Y, 1), np.stack(T, 1)


def main():
    out, t0 = [], time.time()

    def check(name, value, tol, rel='<='):
        ok = value <= tol if rel == '<=' else value >= tol
        out.append(dict(test=name, value=float(value), tolerance=float(tol), relation=rel, passed=bool(ok)))
        print(('PASS ' if ok else 'FAIL ') + name, value, flush=True)

    rng = np.random.default_rng(400920)
    x = box_points(rng, 64); q = box_points(rng, 150)
    freqs = np.array([450.0, 500.0, 560.0])
    src = [np.array([0.5, 0.7, -0.3])]
    Y, T = band_field(x, q, freqs, src, [1.0])
    band = v2.Band(x, Y, q, freqs, C0)
    # gradient of the joint objective
    lo, hi = band.search_box()
    th = np.r_[src[0] + 0.03, np.log([1.0, 2.0, 0.5]), np.log([1e-3, 2e-3, 5e-3])]
    v, g = v2.joint_objective(th, band, 1)
    num = np.array([(v2.joint_objective(th + h, band, 1)[0] - v2.joint_objective(th - h, band, 1)[0]) / 2e-6
                    for h in np.eye(len(th)) * 1e-6])
    check('joint evidence gradient equals finite differences (K = 1, three bins)', np.max(np.abs(num - g)) / np.max(np.abs(g)), 1e-4)
    thr = np.r_[src[0] + 0.03, src[0] - [0.4, 0.2, 0.1], np.log([0.05, 0.08]), np.log([1, 2, 1, 0.5, 2, 1]), np.log([1e-3, 2e-3, 5e-3])]
    v, g = v2.region_joint_objective(thr, band, 2)
    num = np.array([(v2.region_joint_objective(thr + h, band, 2)[0] - v2.region_joint_objective(thr - h, band, 2)[0]) / 2e-6
                    for h in np.eye(len(thr)) * 1e-6])
    check('joint region evidence gradient equals finite differences (J = 2, three bins)', np.max(np.abs(num - g)) / np.max(np.abs(g)), 1e-4)
    # exact recovery, one source, noise-free
    models = v2.fit_band(band, kmax=3)
    err = np.max([E.nmse(v2.predict_points(band, models[1], f), T[:, f]) for f in range(3)])
    check('exact recovery of a point-source band (K = 1): worst-bin NMSE', err, 1e-8)
    check('exact recovery: localisation error (m)', np.linalg.norm(models[1]['S'][0] - src[0]), 1e-4)
    # two sources plus a reverberant remainder (eight random plane waves at 0.3 of the direct power) and 30 dB SNR
    src2 = [np.array([0.5, 0.7, -0.3]), np.array([-0.5, 0.2, 0.4])]
    Y2, T2 = band_field(x, q, freqs, src2, [1.0, 0.7j])
    dirs = rng.normal(size=(8, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    ph = np.exp(2j * np.pi * rng.random(8))
    for i, f in enumerate(freqs):
        k = 2 * np.pi * f / C0
        rx = np.exp(1j * k * x @ dirs.T) @ ph; rq = np.exp(1j * k * q @ dirs.T) @ ph
        s_ = np.sqrt(0.3) / np.sqrt(np.mean(np.abs(rx) ** 2))
        Y2[:, i] += s_ * rx; T2[:, i] += s_ * rq
        Y2[:, i] += np.sqrt(np.mean(np.abs(Y2[:, i]) ** 2) * 1e-3 / 2) * (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x)))
    band2 = v2.Band(x, Y2, q, freqs, C0)
    m2 = v2.fit_band(band2, kmax=4)
    mean_err = lambda pred_f: float(np.mean([E.nmse(pred_f(f), T2[:, f]) for f in range(3)]))
    ref = mean_err(lambda f: E.oracle_fit(band2.probs[f], np.array(src2))['pred'])
    e2 = mean_err(lambda f: v2.predict_points(band2, m2[2], f))
    K, _ = v2.schwarz_order(m2, band2)
    check('two sources, reverberant remainder, 30 dB: Schwarz order selects K = 2', abs(K - 2), 0)
    K, _ = v2.laplace_order(m2, band2)
    check('same field: Laplace-selected order predicts within 5% of K = 2 (ratio - 1)',
          mean_err(lambda f: v2.predict_points(band2, m2[K], f)) / e2 - 1, 0.05)
    check('same field: K = 2 mean NMSE within 20% of point sources at the true positions (ratio - 1)', e2 / ref - 1, 0.2)
    ext = v2.extend(band2, m2[2], 2)
    check('extension of the two points: mean NMSE within 20% of the true-position reference (ratio - 1)',
          mean_err(lambda f: v2.predict_regions(band2, ext, f)) / ref - 1, 0.2)
    gs = v2.gsmf_fit(band2)
    check('group-sparse multi-frequency region kernel: mean NMSE within a factor 2 of the true-position reference (ratio - 1)',
          mean_err(lambda f: v2.predict_regions(band2, gs, f)) / ref - 1, 1.0)
    # remainder family: a plane-wave field must keep the sinc family
    d = np.array([0.3, -0.5, 0.81]); d /= np.linalg.norm(d)
    Yp = np.stack([np.exp(1j * 2 * np.pi * f / C0 * x @ d) + 1e-3 * (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x))) for f in freqs], 1)
    bp = v2.Band(x, Yp, q, freqs, C0)
    fam = v2.choose_family(bp)
    check('remainder family for a noisy plane wave is sinc (0 = sinc)', 0 if fam == 'sinc' else 1, 0)
    # a field drawn from a smooth Gaussian process (not a Helmholtz solution) must choose a Gaussian family
    Gx = v2.gauss_k(x, x, 0.6) + 1e-8 * np.eye(len(x))
    L = np.linalg.cholesky(Gx)
    Yg = np.stack([L @ ((rng.normal(size=len(x)) + 1j * rng.normal(size=len(x))) / np.sqrt(2)) + 1e-3 * rng.normal(size=len(x)) for _ in freqs], 1)
    bg = v2.Band(x, Yg, q, np.array([125.0, 130.0, 135.0]), C0)
    fam = v2.choose_family(bg)
    check('remainder family for a smooth Gaussian-process field at 125 Hz is Gaussian or sum (0 = yes)', 0 if fam in ('gauss', 'sum') else 1, 0)
    # sparse Bayesian learning with the common domain reaches a source 0.6 m outside the region at 1 kHz
    src3 = [np.array([1.8, 0.6, 0.5])]
    Y3, T3 = band_field(x, q, [1000.0], src3, [1.0], noise=1e-4, rng=rng)
    p3 = E.Problem(x, Y3[:, 0], q, 1000.0, C0)
    with v2._common_domain_dictionary():
        nodes, _ = E.sbl_dictionary(p3)
    check('common-domain sparse dictionary covers a source 0.6 m beside the region at 1 kHz (nearest node, m)',
          float(np.min(np.linalg.norm(nodes - src3[0], axis=1))), 0.25)
    json.dump(dict(passed=all(o['passed'] for o in out), n=len(out), tests=out, wall_seconds=time.time() - t0),
              open(sys.argv[1], 'w'), indent=1)
    print('all passed' if all(o['passed'] for o in out) else 'SOME FAILED', round(time.time() - t0, 1), 's')


if __name__ == '__main__':
    main()
