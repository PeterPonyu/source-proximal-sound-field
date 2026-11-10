"""WP2 use 1 (PLAN.md): the recorded fitted balls of the proposed estimator (posthoc/review3/refit/rows.jsonl, every
registered TB-S2 and TB-M2 problem), predicted with (a) the registered 33-node cubature, (b) the existing series with its
fine-cubature fallback near or inside a ball, (c) the exact predictor of exact_ball.py. Gates: (a) reproduces the recorded
j_cv_ext, (b) the recorded frozen_exact. Node stress: targets placed on every cubature node of every ball and 1 mm from it.

    OPENBLAS_NUM_THREADS=1 nice -n 10 python3 -B posthoc/upgrade-wp2/stress.py OUT.jsonl --workers 18
"""
from __future__ import annotations
import argparse
import json
import sys
import traceback
import warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve

HERE = Path(__file__).resolve().parent
A = HERE.parents[1]
for p in (HERE, A / 'common', A / 'V2-dev', A / 'posthoc' / 'review3'):
    sys.path.insert(0, str(p))
import exact_ball as XB  # noqa: E402
import estimators as E  # noqa: E402
import cubature_diag as CD  # noqa: E402
import refit as RF  # noqa: E402


def predict(prob, cen, rad, lw, rho, q, kind):
    """Posterior mean at q for balls (cen, rad, lw) with ridge rho; kind in {'cubature', 'series', 'exact'}."""
    C = prob.H + (rho + E.JITTER) * np.eye(prob.M)
    Cq = E.sinc_k(prob.k, q, prob.x).astype(complex)
    for j in range(len(cen)):
        if kind == 'cubature':
            Fm = E.region_factor(prob.x, cen[j], rad[j], prob.k); Fq = E.region_factor(q, cen[j], rad[j], prob.k)
            K = Fm @ Fm.conj().T; Kq = Fq @ Fm.conj().T
        else:
            K = CD.series_K(prob.x, cen[j], rad[j], prob.k)
            Kq = CD.cross_K(q, prob.x, cen[j], rad[j], prob.k) if kind == 'series' else XB.cross_K(q, prob.x, cen[j], rad[j], prob.k)
        s = np.exp(lw[j]) / float(np.mean(np.real(np.diag(K))))
        C = C + s * K; Cq = Cq + s * Kq
    return Cq @ cho_solve(cho_factor(C, lower=True), prob.y)


def one(line):
    warnings.simplefilter('ignore')
    row = json.loads(line)
    key = row['key']
    try:
        st = row['proposed']
        if not st.get('balls'):
            return dict(key=key, K=st['K_cv'], skipped='K=0')
        band, T = RF.build(key)
        b = st['balls']
        cen, rad, lw, R = np.array(b['centres']), np.array(b['radii']), np.array(b['lw']), np.array(b['ridge'])
        out = dict(key=key, K=st['K_cv'], radii=rad.tolist(), nmse={})
        q = band.q
        dmin = np.min([np.linalg.norm(q - c, axis=1) - r for c, r in zip(cen, rad)], axis=0)
        out['targets_inside'] = int(np.sum(dmin < 0)); out['targets_within_2cm'] = int(np.sum(dmin < 0.02))
        nodes = np.vstack([c + r * E.BALL for c, r in zip(cen, rad)])
        out['target_to_node_min_m'] = float(np.min(np.linalg.norm(q[:, None, :] - nodes[None, :, :], axis=2)))
        for kind in ('cubature', 'series', 'exact'):
            out['nmse'][kind] = [E.nmse(predict(band.probs[f], cen, rad, lw[f], R[f], q, kind), T[:, f]) for f in range(band.F)]
        out['gate_cubature'] = float(np.max(np.abs(np.array(out['nmse']['cubature']) / np.array(st['j_cv_ext']) - 1)))
        if st.get('frozen_exact'):
            out['gate_series'] = float(np.max(np.abs(np.array(out['nmse']['series']) / np.array(st['frozen_exact']) - 1)))
        # node stress at the highest bin: targets on the nodes and 1 mm away (random direction)
        f = band.F - 1; p = band.probs[f]
        rng = np.random.default_rng(0)
        off = rng.standard_normal(nodes.shape); off /= np.linalg.norm(off, axis=1)[:, None]
        on, near = nodes + 1e-7 * off, nodes + 1e-3 * off
        typical = np.median(np.abs(predict(p, cen, rad, lw[f], R[f], q, 'exact')))
        st_out = {}
        for kind in ('cubature', 'exact'):
            a = np.abs(predict(p, cen, rad, lw[f], R[f], on, kind)); b_ = np.abs(predict(p, cen, rad, lw[f], R[f], near, kind))
            st_out[kind] = dict(max_on_node_over_typical=float(a.max() / typical), max_near_over_typical=float(b_.max() / typical))
        st_out['exact_on_vs_near_max_rel'] = float(np.max(np.abs(predict(p, cen, rad, lw[f], R[f], on, 'exact') - predict(p, cen, rad, lw[f], R[f], near, 'exact')))
                                                   / typical)
        out['node_stress'] = st_out
        return out
    except Exception:
        return dict(key=key, error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--workers', type=int, default=18)
    ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    lines = open(A / 'posthoc/review3/refit/rows.jsonl').read().splitlines()
    if a.limit:
        lines = lines[::max(1, len(lines) // a.limit)][:a.limit]
    with Pool(a.workers, initializer=RF.init) as pool, open(a.out, 'w') as fh:
        for n, r in enumerate(pool.imap_unordered(one, lines, chunksize=4), 1):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if n % 200 == 0 or n == len(lines):
                print(n, len(lines), flush=True)


if __name__ == '__main__':
    main()
