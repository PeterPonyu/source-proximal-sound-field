"""Known-answer checks of the author-recipe comparators (common/prox.py). Run before registration and before any
registered output:  python3 -B prox_checks.py OUT.json   (exit code 0 and "passed": true when every check passes).

1. Leave-one-out identity: the closed-form residuals equal explicit leave-one-out refits.
2. Weight gradient: analytic against central finite differences (stage 1, two bins).
3. Region gradients: weights, centres and radii against central finite differences (stage 2, two bins).
4. Proximal operator: one bin equals max(0, z - step tau); with three bins no feasible perturbation lowers the
   proximal objective.
5. Free-field point source at a dictionary node: the weight concentrates near the source and the error is far below
   the sinc kernel (250, 500, 1000 Hz); the case between nodes is recorded without a threshold.
6. Sparsity rises with tau where the fixed-step iteration converges (250 Hz); 500 Hz is recorded without a threshold.
7. The group-sparse variant with one bin reproduces the per-bin variant.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import estimators as E  # noqa: E402
import prox as PX  # noqa: E402
import v2  # noqa: E402

C0 = 343.0


def region_nodes():
    g = [np.arange(n) * 0.15 for n in (9, 9, 8)]
    return np.stack(np.meshgrid(*g, indexing='ij'), -1).reshape(-1, 3)


def free_field_problem(f=500.0, M=48, seed=7, src=None, snr_db=40.0):
    rng = np.random.default_rng(seed)
    nodes = region_nodes()
    mic = np.sort(rng.choice(len(nodes), M, replace=False)); tgt = np.setdiff1d(np.arange(len(nodes)), mic)
    src = np.array([0.6, 0.6, -0.4]) if src is None else np.asarray(src, float)
    k = 2 * np.pi * f / C0
    p = lambda pts: (np.exp(1j * k * np.linalg.norm(pts - src, axis=1)) / np.linalg.norm(pts - src, axis=1))
    y, t = p(nodes[mic]), p(nodes[tgt])
    s = np.sqrt(np.mean(np.abs(y) ** 2))
    noise = (rng.standard_normal(M) + 1j * rng.standard_normal(M)) / np.sqrt(2) * s * 10 ** (-snr_db / 20)
    box = (nodes.min(0), nodes.max(0))
    return E.Problem(nodes[mic], y + noise, nodes[tgt], f, C0, box), t, src


def check_loo():
    rng = np.random.default_rng(1)
    M = 15
    X = rng.standard_normal((M, 4)) + 1j * rng.standard_normal((M, 4))
    K = X @ X.conj().T + 0.5 * np.eye(M)
    y = rng.standard_normal(M) + 1j * rng.standard_normal(M)
    rho = 0.1
    A = K + rho * np.eye(M)
    B = np.linalg.inv(A)
    e_closed = (B @ y) / np.real(np.diag(B))
    e_refit = np.empty(M, complex)
    for i in range(M):
        o = np.setdiff1d(np.arange(M), [i])
        e_refit[i] = y[i] - K[i, o] @ np.linalg.solve(K[np.ix_(o, o)] + rho * np.eye(M - 1), y[o])
    L, _ = PX.loo(A, y)
    rel = float(np.max(np.abs(e_closed - e_refit)) / np.max(np.abs(e_refit)))
    return dict(name='leave-one-out identity', max_rel_diff=rel, loss_matches=bool(abs(L - PX._ls(M) * np.sum(np.abs(e_refit) ** 2)) < 1e-10 * L),
                passed=bool(rel < 1e-10 and abs(L - PX._ls(M) * np.sum(np.abs(e_refit) ** 2)) < 1e-10 * L))


def two_bin_setup(M=20, J=4, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1.0, (M, 3))
    ks = [2 * np.pi * f / C0 for f in (400.0, 520.0)]
    cen = np.array([[0.5, 0.5, -0.4], [1.4, 0.3, 0.5], [0.2, 1.5, 0.6], [0.6, -0.5, 1.2]])[:J]
    rad = np.array([0.05, 0.08, 0.12, 0.06])[:J]
    Hr = [E.sinc_k(k, x, x) + PX.RIDGE * np.eye(M) for k in ks]
    ys = []
    for k in ks:
        g = E.green(x, np.array([[0.45, 0.55, -0.35]]), k)[:, 0] + 0.1 * (rng.standard_normal(M) + 1j * rng.standard_normal(M))
        ys.append(g / np.sqrt(np.mean(np.abs(g) ** 2)))
    W = rng.uniform(0.2, 2.0, (len(ks), J))
    return x, ks, Hr, ys, cen, rad, W


def total_loss(x, ks, Hr, ys, cen, rad, W):
    Lt = 0.0
    for f, k in enumerate(ks):
        A = Hr[f].copy()
        Fx = PX.factors(x, cen, rad, k)
        Ks, _ = PX.gram_stack(Fx)
        A = A + np.tensordot(W[f], Ks, axes=1)
        Lt += PX.loo(A, ys[f])[0]
    return Lt


def check_weight_grad():
    x, ks, Hr, ys, cen, rad, W = two_bin_setup()
    M = len(x)
    g = np.empty_like(W)
    for f, k in enumerate(ks):
        Ks, _ = PX.gram_stack(PX.factors(x, cen, rad, k))
        A = Hr[f] + np.tensordot(W[f], Ks, axes=1)
        _, G = PX.loo(A, ys[f])
        g[f] = (2.0 * PX._ls(M)) * np.real(Ks.reshape(len(rad), -1) @ np.ascontiguousarray(G.T).reshape(-1))
    fd = np.empty_like(W); h = 1e-6
    for f in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp, Wm = W.copy(), W.copy(); Wp[f, j] += h; Wm[f, j] -= h
            fd[f, j] = (total_loss(x, ks, Hr, ys, cen, rad, Wp) - total_loss(x, ks, Hr, ys, cen, rad, Wm)) / (2 * h)
    rel = float(np.max(np.abs(g - fd)) / np.max(np.abs(fd)))
    return dict(name='weight gradient (stage 1) against finite differences', max_rel_err=rel, passed=bool(rel < 1e-6))


def check_region_grad():
    x, ks, Hr, ys, cen, rad, W = two_bin_setup()
    Lt, gW, gc, gr = PX.loss_and_grads(x, ks, Hr, ys, cen, rad, W)
    out = {}
    h = 1e-6
    fdW = np.empty_like(W)
    for f in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp, Wm = W.copy(), W.copy(); Wp[f, j] += h; Wm[f, j] -= h
            fdW[f, j] = (total_loss(x, ks, Hr, ys, cen, rad, Wp) - total_loss(x, ks, Hr, ys, cen, rad, Wm)) / (2 * h)
    fdc = np.empty_like(cen)
    for j in range(cen.shape[0]):
        for c in range(3):
            cp, cm = cen.copy(), cen.copy(); cp[j, c] += h; cm[j, c] -= h
            fdc[j, c] = (total_loss(x, ks, Hr, ys, cp, rad, W) - total_loss(x, ks, Hr, ys, cm, rad, W)) / (2 * h)
    fdr = np.empty_like(rad)
    for j in range(len(rad)):
        rp, rm = rad.copy(), rad.copy(); rp[j] += h; rm[j] -= h
        fdr[j] = (total_loss(x, ks, Hr, ys, cen, rp, W) - total_loss(x, ks, Hr, ys, cen, rm, W)) / (2 * h)
    for name, a, b in (('weights', gW, fdW), ('centres', gc, fdc), ('radii', gr, fdr)):
        out[name] = float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
    loss_ok = abs(Lt - total_loss(x, ks, Hr, ys, cen, rad, W)) < 1e-12 * Lt
    vec = PX.region_terms(x, cen, rad, ks[0])
    same = []
    for j in range(len(rad)):
        _, (Fx, dFc, dFa), nrm, dnc, dna = E._region_parts(x, cen[j], rad[j], ks[0], x)
        for a, b in ((vec[0][:, j], Fx), (vec[1][:, j], dFc), (vec[2][:, j], dFa), (vec[3][j], nrm), (vec[4][j], dnc), (vec[5][j], dna)):
            same.append(float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / max(float(np.max(np.abs(b))), 1e-300)))
    out['vectorised_terms_vs_registered'] = max(same)
    return dict(name='region gradients (stage 2) against finite differences', max_rel_err=out, loss_consistent=bool(loss_ok),
                passed=bool(loss_ok and max(out.values()) < 1e-5))


def check_prox():
    rng = np.random.default_rng(5)
    z = rng.normal(0, 0.01, (1, 200))
    one = PX.group_prox(z, PX.ETA)
    ok1 = bool(np.allclose(one, np.maximum(0, z - PX.ETA * PX.TAU), rtol=0, atol=1e-15))
    Z = rng.normal(0.0005, 0.002, (3, 60))
    P = PX.group_prox(Z, PX.ETA)
    thr = PX.ETA * PX.TAU
    obj = lambda W: 0.5 * np.sum((W - Z) ** 2, axis=0) + thr * np.linalg.norm(W, axis=0)
    base = obj(P)
    worst = 0.0
    for _ in range(2000):
        Wp = np.maximum(P + rng.normal(0, 1e-4, P.shape), 0.0)
        worst = max(worst, float(np.max(base - obj(Wp))))
    feasible = bool(np.all(P >= 0))
    return dict(name='proximal operator', one_bin_equals_soft_threshold=ok1, max_improvement_by_perturbation=worst,
                zero_columns=int(np.sum(np.all(P == 0, axis=0))), passed=bool(ok1 and feasible and worst <= 1e-15))


def _on_grid_source(f):
    """The dictionary node nearest to (0.6, 0.6, -0.4), moved by (1, -1, 1) cm: a source the dictionary resolves."""
    p0, _, _ = free_field_problem(f=f)
    lo, hi = p0.search_box()
    cen0, _ = PX.dictionary(p0.x, lo, hi, p0.lam)
    return cen0[np.argmin(np.linalg.norm(cen0 - np.array([0.6, 0.6, -0.4]), axis=1))] + np.array([0.01, -0.01, 0.01])


def _ff_case(f, src):
    p, t, src = free_field_problem(f=f, src=src)
    t0 = time.process_time()
    r = PX.mo26_prox(p)
    secs = time.process_time() - t0
    s = E.sinc_fit(p, 'evidence')
    info = r['info']
    cen = np.array(info['centres']); w = np.array(info['weights'])[0] if info['regions'] else np.zeros(0)
    d = np.linalg.norm(cen - src, axis=1) if len(cen) else np.zeros(0)
    return dict(frequency=f, source=src.tolist(), nmse_db=dict(mo26_prox=10 * np.log10(E.nmse(r['pred'], t)), sinc=10 * np.log10(E.nmse(s['pred'], t))),
                heaviest_region_distance_m=float(d[np.argmax(w)]) if len(w) else None, quarter_wavelength_m=p.lam / 4,
                weight_fraction_within_half_wavelength=float(w[d < 0.5 * p.lam].sum() / w.sum()) if w.sum() > 0 else 0.0,
                regions=info['regions'], after_stage1=info['after_stage1'], dictionary=info['dictionary'], process_seconds=round(secs, 2))


def check_free_field():
    """Known answer: a free-field monopole 1.7 cm from a dictionary node (48 microphones, 40 dB SNR) at 250, 500 and
    1000 Hz. Pass: error at least 10 dB below the sinc kernel, heaviest region within a quarter wavelength of the
    source, at least 80 % of the weight within half a wavelength, at every frequency. Descriptive (no threshold): the
    same with the source at (0.6, 0.6, -0.4), between dictionary nodes."""
    cases = [_ff_case(f, _on_grid_source(f)) for f in (250.0, 500.0, 1000.0)]
    ok = [c['nmse_db']['mo26_prox'] < c['nmse_db']['sinc'] - 10 and c['heaviest_region_distance_m'] is not None
          and c['heaviest_region_distance_m'] < c['quarter_wavelength_m'] and c['weight_fraction_within_half_wavelength'] >= 0.8 for c in cases]
    off = [_ff_case(f, np.array([0.6, 0.6, -0.4])) for f in (250.0, 500.0, 1000.0)]
    return dict(name='free-field point source at a dictionary node (250, 500, 1000 Hz)', cases=cases, passed_each=ok,
                descriptive_source_between_nodes=off, passed=bool(all(ok)))


def check_sparsity():
    """Stage 1 (the module's own step rule) on the 250 Hz free-field problem: the number of non-zero weights after
    ITERS steps must not grow with tau and must be smaller at 1e-1 than at 1e-3. Descriptive: the same at 500 Hz."""
    def support(f):
        p, t, src = free_field_problem(f=f)
        lo, hi = p.search_box()
        cen0, rad0 = PX.dictionary(p.x, lo, hi, p.lam)
        Ks = [PX.gram_stack(PX.factors(p.x, cen0, rad0, p.k))[0]]
        y = p.y / np.sqrt(np.mean(np.abs(p.y) ** 2)); Hr = [p.H + PX.RIDGE * np.eye(p.M)]
        res = {}
        tau0 = PX.TAU
        try:
            for tau in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
                PX.TAU = tau
                rec = []
                W = PX._stage_weights(Hr, [y], Ks, record=rec)
                res[f'{tau:g}'] = dict(nonzero=int(np.sum(W > 0)), loss_last=rec[-1], loss_min=min(rec))
        finally:
            PX.TAU = tau0
        return res
    main = support(250.0)
    v = [main[k]['nonzero'] for k in ('0.001', '0.003', '0.01', '0.03', '0.1')]
    ok = all(a >= b for a, b in zip(v, v[1:])) and v[0] > v[-1]
    return dict(name='sparsity rises with tau (stage 1, 250 Hz free-field problem)', support_250Hz=main,
                descriptive_support_500Hz=support(500.0), passed=bool(ok))


def check_one_bin():
    p, t, src = free_field_problem(f=250.0, M=24, seed=11)
    a = PX.mo26_prox(p)
    band = v2.Band(p.x, p.y[:, None], p.q, [p.f], p.c, p.box)
    b = PX.gsmf_prox(band)
    rel = float(np.max(np.abs(a['pred'] - b['preds'][0])) / np.max(np.abs(a['pred'])))
    return dict(name='group-sparse variant with one bin equals the per-bin variant', max_rel_diff=rel,
                regions=(a['info']['regions'], b['info']['regions']), passed=bool(rel < 1e-9))


def main(out):
    checks = []
    for fn in (check_loo, check_weight_grad, check_region_grad, check_prox, check_free_field, check_sparsity, check_one_bin):
        try:
            checks.append(fn())
        except Exception as exc:  # recorded, never hidden
            checks.append(dict(name=fn.__name__, error=repr(exc), passed=False))
    res = dict(checks=checks, passed=all(c['passed'] for c in checks))
    Path(out).write_text(json.dumps(res, indent=1, default=float))
    for c in checks:
        print(('PASS ' if c['passed'] else 'FAIL ') + c['name'])
    return 0 if res['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
