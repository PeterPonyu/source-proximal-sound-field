"""Known-answer checks of the author-recipe comparators (common/prox2.py). Run before registration and before any
registered output:  python3 -B prox2_checks.py OUT.json [--fast]   (exit 0 and "passed": true when every check passes;
--fast skips the two slow behavioural checks, 6 and 7).

1. The spherical-wave series (the paper's eq. 13, order NMAX) is converged and equals a fine quadrature of eq. 12
   for points 5-30 cm outside the ball.
2. The NLML weight gradient (the paper's eq. 21) equals central finite differences (two bins).
3. The refinement gradients (weights, centres, radii; analytic series derivatives) equal central finite differences.
4. The proximal operator: one bin equals max(0, z - thr); with three bins no feasible perturbation lowers the objective.
5. The group proximal step with one bin equals the per-bin l1 step.
6. Free-field monopole at a dictionary node (6 bins around 500 Hz, 48 microphones, 40 dB SNR): both variants at least
   10 dB below the sinc kernel, and the heaviest region within a quarter wavelength of the source.
7. Where the fixed step converges (zeta 1e-3 .. 1), the number of non-zero regions does not grow with zeta and stays
   sparse (group variant, three free-field sources); zeta = 100 and 1000 are recorded without a threshold.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import estimators as E  # noqa: E402
import prox2 as P2  # noqa: E402
import v2  # noqa: E402
import mo2025 as MO  # noqa: E402

C0 = 343.0


def region_nodes():
    g = [np.arange(n) * 0.15 for n in (9, 9, 8)]
    return np.stack(np.meshgrid(*g, indexing='ij'), -1).reshape(-1, 3)


def free_field_band(src, fc=500.0, M=48, seed=7, snr_db=40.0):
    rng = np.random.default_rng(seed)
    nodes = region_nodes()
    mic = np.sort(rng.choice(len(nodes), M, replace=False)); tgt = np.setdiff1d(np.arange(len(nodes)), mic)
    freqs = fc * 2.0 ** ((np.arange(6) + 0.5) / 6 - 0.5)
    Y, T = [], []
    for f in freqs:
        k = 2 * np.pi * f / C0
        p = lambda pts: np.exp(1j * k * np.linalg.norm(pts - src, axis=1)) / np.linalg.norm(pts - src, axis=1)
        y = p(nodes[mic]); s = np.sqrt(np.mean(np.abs(y) ** 2))
        Y.append(y + (rng.standard_normal(M) + 1j * rng.standard_normal(M)) / np.sqrt(2) * s * 10 ** (-snr_db / 20))
        T.append(p(nodes[tgt]))
    box = (nodes.min(0), nodes.max(0))
    return v2.Band(nodes[mic], np.stack(Y, 1), nodes[tgt], freqs, C0, box), np.stack(T, 1)


def check_series():
    """The series (order NMAX) against itself at order 80 and against a fine quadrature (40 x 3000 nodes), for points
    5-30 cm outside the ball, where the series converges slowest."""
    rng = np.random.default_rng(4)
    out = {}
    for f, rho in ((125.0, 0.37), (250.0, 0.25), (1000.0, 0.25), (1400.0, 0.25)):
        k = 2 * np.pi * f / C0
        c = np.zeros(3)
        dirs = rng.standard_normal((10, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        pts = dirs * rng.uniform(rho + 0.05, rho + 0.3, 10)[:, None]
        K = P2.series_region_kernel(pts, c, rho, k)
        K80 = P2.series_kernel(pts, pts, c, rho, k, nmax=80, n_s=400)
        nodes, w = MO.ball_quadrature(rho, k, sigma_ratio=P2.SIGMA_RATIO, n_r=40, n_a=3000)
        F = MO.green4pi(pts, nodes, k) * np.sqrt(w)
        rel = lambda A: float(np.max(np.abs(A - K80)) / np.max(np.abs(K80)))
        out[f'{int(f)}Hz_rho{rho}'] = dict(series_vs_order80=rel(K), fine_quadrature_vs_order80=rel(F @ F.conj().T))
    ok = all(v['series_vs_order80'] < 1e-9 and v['fine_quadrature_vs_order80'] < 5e-5 for v in out.values())
    return dict(name='series kernel (eq. 13) converged and equal to a fine quadrature of eq. 12', max_rel_diff=out, passed=bool(ok))


def _mc_kernel(pts, c, rho, k, n=400000, seed=0):
    """Monte Carlo of eq. 12: samples from the Gaussian density (sigma = rho/3) restricted to the ball, weighted by the
    density's mass inside the ball (the kernels here use the untruncated normalisation over the ball)."""
    rng = np.random.default_rng(seed)
    sig = P2.SIGMA_RATIO * rho
    s = rng.normal(0, sig, (n, 3))
    inside = np.linalg.norm(s, axis=1) <= rho
    mass = inside.mean()
    s = s[inside] + c
    G = MO.green4pi(pts, s, k)
    return (G @ G.conj().T) / len(s) * mass


def check_series_independent():
    """The series against independent integrals of eq. 12: a dense Monte Carlo at large k rho (relative error of the
    Monte Carlo about 0.2 %), and the 2025 reimplementation's quadrature (common/mo2025.py) at small k rho."""
    rng = np.random.default_rng(6)
    out = {}
    for f, rho, ref in ((1400.0, 0.37, 'monte_carlo'), (1000.0, 0.25, 'monte_carlo'), (125.0, 0.25, 'mo2025_quadrature'), (250.0, 0.2, 'mo2025_quadrature')):
        k = 2 * np.pi * f / C0
        c = np.zeros(3)
        dirs = rng.standard_normal((8, 3)); dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        pts = dirs * rng.uniform(rho + 0.05, rho + 0.6, 8)[:, None]
        K = P2.series_region_kernel(pts, c, rho, k)
        if ref == 'monte_carlo':
            R = _mc_kernel(pts, c, rho, k)
        else:
            R = MO.kernel(pts, pts, [(c, rho)], k)
        out[f'{int(f)}Hz_rho{rho}_krho{k * rho:.2f}_{ref}'] = float(np.linalg.norm(K - R) / np.linalg.norm(R))
    return dict(name='series kernel against a Monte Carlo integral (large k rho) and the 2025 quadrature (small k rho)',
                rel_frobenius=out, passed=bool(max(out.values()) < 1e-2))


def check_target_cubature():
    """Targets inside or near a ball use the dense cubature FINE_QUAD; its cross-kernel to microphones outside the
    ball against a Monte Carlo integral, at each band's largest k rho of the registered dictionaries."""
    rng = np.random.default_rng(8)
    out = {}
    for f, rho in ((180.0, 0.37), (350.0, 0.25), (700.0, 0.23), (1400.0, 0.23)):
        k = 2 * np.pi * f / C0
        c = np.zeros(3)
        dq = rng.standard_normal((6, 3)); dq /= np.linalg.norm(dq, axis=1)[:, None]
        q = dq * rng.uniform(0.2 * rho, rho + 0.015, 6)[:, None]
        dx = rng.standard_normal((8, 3)); dx /= np.linalg.norm(dx, axis=1)[:, None]
        x = dx * rng.uniform(rho + 0.05, rho + 0.6, 8)[:, None]
        K = P2.cross_kernel(q, x, c, rho, k, P2.xi(rho, k))
        rng2 = np.random.default_rng(0)
        sig = P2.SIGMA_RATIO * rho
        s = rng2.normal(0, sig, (600000, 3)); inside = np.linalg.norm(s, axis=1) <= rho
        mass = inside.mean(); s = s[inside]
        R = (MO.green4pi(q, s, k) @ MO.green4pi(x, s, k).conj().T) / len(s) * mass
        out[f'{int(f)}Hz_rho{rho}_krho{k * rho:.2f}'] = float(np.linalg.norm(K - R) / np.linalg.norm(R))
    return dict(name='dense cubature for targets inside a ball against a Monte Carlo integral (about 1 %)',
                rel_frobenius=out, passed=bool(max(out.values()) < 2e-2))


def two_bin_setup(M=16, J=3, seed=3, equal_radii=False):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1.0, (M, 3))
    ks = [2 * np.pi * f / C0 for f in (400.0, 520.0)]
    cen = np.array([[0.5, 0.5, -0.5], [1.6, 0.3, 0.5], [0.2, 1.7, 0.6]])[:J]
    rad = np.full(J, 0.2) if equal_radii else np.array([0.15, 0.2, 0.12])[:J]
    Bases = [0.06 * np.eye(M) for _ in ks]
    ys = []
    for k in ks:
        g = E.green(x, np.array([[0.45, 0.55, -0.45]]), k)[:, 0] + 0.1 * (rng.standard_normal(M) + 1j * rng.standard_normal(M))
        ys.append(g / np.sqrt(np.mean(np.abs(g) ** 2)))
    cs = [50.0, 60.0]
    W = rng.uniform(0.2, 2.0, (len(ks), J))
    return x, ks, cs, Bases, ys, cen, rad, W


def total_loss(x, ks, cs, Bases, ys, cen, rad, W):
    Lt = 0.0
    for f, k in enumerate(ks):
        A = Bases[f].astype(complex)
        for j in range(len(rad)):
            A = A + cs[f] * W[f, j] * P2.series_region_kernel(x, cen[j], rad[j], k)
        Lt += P2.nlml(A, ys[f])[0]
    return Lt


def check_weight_grad():
    x, ks, cs, Bases, ys, cen, rad, W = two_bin_setup(equal_radii=True)
    Ks = [cs[f] * P2.series_stacks(x, cen, P2.xi(rad[0], k), k) for f, k in enumerate(ks)]
    flat = [K.reshape(len(rad), -1) for K in Ks]
    _, g = P2.weights_loss(Bases, ys, Ks, flat, W)
    fd = np.empty_like(W); h = 1e-6
    for f in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp, Wm = W.copy(), W.copy(); Wp[f, j] += h; Wm[f, j] -= h
            fd[f, j] = (total_loss(x, ks, cs, Bases, ys, cen, rad, Wp) - total_loss(x, ks, cs, Bases, ys, cen, rad, Wm)) / (2 * h)
    rel = float(np.max(np.abs(g - fd)) / np.max(np.abs(fd)))
    same = float(np.max(np.abs(Ks[0][1] / cs[0] - P2.series_region_kernel(x, cen[1], rad[1], ks[0]))))
    return dict(name='NLML weight gradient (eq. 21) against finite differences', max_rel_err=rel, stack_vs_single=same,
                passed=bool(rel < 1e-6 and same < 1e-12))


def check_refine_grad():
    x, ks, cs, Bases, ys, cen, rad, W = two_bin_setup()
    Lt, gW, gc, gr = P2.refine_loss_grads(x, ks, cs, Bases, ys, cen, rad, W)
    h = 1e-6
    fdW = np.empty_like(W)
    for f in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp, Wm = W.copy(), W.copy(); Wp[f, j] += h; Wm[f, j] -= h
            fdW[f, j] = (total_loss(x, ks, cs, Bases, ys, cen, rad, Wp) - total_loss(x, ks, cs, Bases, ys, cen, rad, Wm)) / (2 * h)
    fdc = np.empty_like(cen)
    for j in range(len(rad)):
        for c in range(3):
            cp, cm = cen.copy(), cen.copy(); cp[j, c] += h; cm[j, c] -= h
            fdc[j, c] = (total_loss(x, ks, cs, Bases, ys, cp, rad, W) - total_loss(x, ks, cs, Bases, ys, cm, rad, W)) / (2 * h)
    fdr = np.empty_like(rad)
    for j in range(len(rad)):
        rp, rm = rad.copy(), rad.copy(); rp[j] += h; rm[j] -= h
        fdr[j] = (total_loss(x, ks, cs, Bases, ys, cen, rp, W) - total_loss(x, ks, cs, Bases, ys, cen, rm, W)) / (2 * h)
    out = {n: float(np.max(np.abs(a - b)) / np.max(np.abs(b))) for n, a, b in (('weights', gW, fdW), ('centres', gc, fdc), ('radii', gr, fdr))}
    ok = abs(Lt - total_loss(x, ks, cs, Bases, ys, cen, rad, W)) < 1e-10 * abs(Lt)
    return dict(name='refinement gradients (series, analytic) against finite differences', max_rel_err=out, loss_consistent=bool(ok),
                passed=bool(ok and max(out.values()) < 1e-5))


def paper_units():
    """Both papers' data units from their stated set-ups: mean microphone power P and starting-kernel diagonal K0."""
    th = np.arange(16) * 2 * np.pi / 16
    mics = np.c_[np.cos(th), np.sin(th), np.zeros(16)]
    srcs = np.array([[0.37, 0, 0], [-0.42, -0.4, 0]])
    r = np.linalg.norm(mics[:, None, :] - srcs[None], axis=2)
    ic_power = float(np.mean(np.sum(1 / (16 * np.pi ** 2 * r ** 2), axis=1)))
    g = np.arange(-0.8, 0.8 + 1e-9, 0.1)
    X, Y = np.meshgrid(g, g)
    pts = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    pts = pts[np.linalg.norm(pts, axis=1) <= 0.8 + 1e-9]
    k = 2 * np.pi * 1000.0 / 340
    ic_k0 = float(np.mean(np.real(np.einsum('smm->sm', P2.series_stacks(mics, pts, P2.xi(0.1 / np.sqrt(2), k), k)))))
    mj = MO.fib(36)                                   # 36 points on the unit sphere (their t-design: 36 microphones)
    src = np.random.default_rng(0).uniform(-0.2, 0.2, (20000, 3)) + np.array([0.2, 0, 0])
    rj = np.linalg.norm(src[:, None, :] - mj[None], axis=2)
    ja_power = float(np.mean(1 / (16 * np.pi ** 2 * rj ** 2)))
    g = np.arange(-0.8, 0.8 + 1e-9, 0.2)
    Xj = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
    Xj = Xj[np.linalg.norm(Xj, axis=1) <= 0.8 + 1e-9]
    ja_k0 = float(np.mean(np.real(np.einsum('smm->sm', P2.series_stacks(mj, Xj, P2.xi(np.sqrt(3) / 2 * 0.2, k), k)))))
    return dict(icassp=dict(power=ic_power, k0=ic_k0, S=int(len(pts))), jasa=dict(power=ja_power, k0=ja_k0, S=int(len(Xj))))


def check_paper_units():
    """The constants of prox2.py against both papers' stated set-ups (ICASSP: S = 197 at 0.1 m as it reports)."""
    u = paper_units()
    ok = (abs(u['icassp']['power'] - P2.ICASSP_POWER) < 5e-5 and abs(u['icassp']['k0'] - P2.ICASSP_K0) < 5e-5
          and u['icassp']['S'] == 197 and abs(u['jasa']['power'] - P2.JASA_POWER) < 5e-6 and abs(u['jasa']['k0'] - P2.JASA_K0) < 5e-6)
    return dict(name="both papers' units recomputed from their set-ups", units=u,
                transferred=dict(gsmf=dict(sigma2=P2.GSMF['sigma2'], k0_rel=P2.GSMF['k0_rel']),
                                 mo26=dict(sigma2=P2.MO26['sigma2'], k0_rel=P2.MO26['k0_rel'])), passed=bool(ok))


def check_prox():
    rng = np.random.default_rng(5)
    z = rng.normal(0, 0.01, (1, 200)); thr = 3e-3
    ok1 = bool(np.allclose(P2.group_prox(z, thr), np.maximum(0, z - thr), rtol=0, atol=1e-15))
    Z = rng.normal(0.0005, 0.002, (3, 60))
    P = P2.group_prox(Z, thr)
    obj = lambda W: 0.5 * np.sum((W - Z) ** 2, axis=0) + thr * np.linalg.norm(W, axis=0)
    worst = max(float(np.max(obj(P) - obj(np.maximum(P + rng.normal(0, 1e-4, P.shape), 0.0)))) for _ in range(2000))
    return dict(name='proximal operator (eq. 24)', one_bin_equals_soft_threshold=ok1, max_improvement_by_perturbation=worst,
                passed=bool(ok1 and np.all(P >= 0) and worst <= 1e-15))


def _on_node_source():
    band, _ = free_field_band(np.array([0.6, 0.6, -0.4]))
    cen, rad, rho, delta = P2.dictionary(band.x, band.box[0] - P2.DICT_MARGIN, band.box[1] + P2.DICT_MARGIN, P2.GSMF['rho_factor'])
    return cen[np.argmin(np.linalg.norm(cen - np.array([0.6, 0.6, -0.4]), axis=1))] + np.array([0.01, -0.01, 0.01])


def check_mic_clearance():
    """No microphone inside any ball (the series kernel's validity): on a MeshRIR-like synthetic planar array (21 x 21
    grid at 0.05 m, 48 microphones, a source 0.6 m from the plane, 250 Hz band), for both variants' dictionaries at the
    start and for the per-bin variant after its refinement; every clearance must be at least MIC_MARGIN."""
    rng = np.random.default_rng(12)
    g = np.arange(21) * 0.05
    nodes = np.stack(np.meshgrid(g, g, [0.0], indexing='ij'), -1).reshape(-1, 3)
    mic = np.sort(rng.choice(len(nodes), 48, replace=False)); tgt = np.setdiff1d(np.arange(len(nodes)), mic)
    src = np.array([0.5, 0.4, 0.6])
    freqs = 250.0 * 2.0 ** ((np.arange(6) + 0.5) / 6 - 0.5)
    Y = []
    for f in freqs:
        k = 2 * np.pi * f / C0
        y = np.exp(1j * k * np.linalg.norm(nodes[mic] - src, axis=1)) / np.linalg.norm(nodes[mic] - src, axis=1)
        Y.append(y + (rng.standard_normal(48) + 1j * rng.standard_normal(48)) * 1e-3 * np.sqrt(np.mean(np.abs(y) ** 2)))
    band = v2.Band(nodes[mic], np.stack(Y, 1), nodes[tgt], freqs, C0, (nodes.min(0), nodes.max(0)))
    out = {}
    for name, cfg in (('gsmf', P2.GSMF), ('mo26', P2.MO26)):
        prep = P2.Prepared(band, cfg)
        out[name] = dict(S=int(len(prep.cen)), shrunk=int(np.sum(prep.rad < prep.rho - 1e-12)), rho0=prep.rho,
                         min_radius=float(prep.rad.min()), start_clearance=P2._min_clear(prep.x, prep.cen, prep.rad))
        if name == 'mo26':
            r = P2.mo26_prox(prep, 2)
            out[name].update(after_refinement_clearance=r['info']['min_mic_clearance_m'], iterations=r['info']['iterations'])
    vals = [v['start_clearance'] for v in out.values()] + [out['mo26'].get('after_refinement_clearance') or P2.MIC_MARGIN]
    return dict(name='no microphone inside any ball (start and after refinement, planar array)', details=out,
                passed=bool(min(vals) >= P2.MIC_MARGIN - 1e-9 and out['mo26']['shrunk'] > 0))


def check_one_bin():
    """The group proximal step with one bin is the per-bin l1 step: gsmf_prox's stage on a one-bin band equals the
    stage-1 weights computed bin by bin with the same settings."""
    band, _ = free_field_band(_on_node_source(), fc=250.0)
    b1 = v2.Band(band.x, np.stack([band.probs[2].y], 1), band.q, [band.freqs[2]], band.c, band.box)
    prep = P2.Prepared(b1, P2.GSMF)
    c = P2.GSMF
    Wg = P2.stage_weights(prep.Bases, prep.ys, prep.Ks, c['zeta'], c['eta'], c['iters'])
    Wm = np.vstack([P2.stage_weights([prep.Bases[0]], [prep.ys[0]], [prep.Ks[0]], c['zeta'], c['eta'], c['iters'])])
    rel = float(np.max(np.abs(Wg - Wm)) / max(float(np.max(np.abs(Wm))), 1e-300))
    return dict(name='group step with one bin equals the per-bin l1 step', max_rel_diff=rel, nonzero=int(np.sum(Wm > 0)),
                passed=bool(rel == 0.0))


def check_free_field():
    src = _on_node_source()
    band, T = free_field_band(src)
    t0 = time.process_time()
    pg = P2.Prepared(band, P2.GSMF)
    rec = []
    g = P2.gsmf_prox(pg, rec)
    Wg = P2.stage_weights(pg.Bases, pg.ys, pg.Ks, P2.GSMF['zeta'], P2.GSMF['eta'], P2.GSMF['iters'])
    pm = P2.Prepared(band, P2.MO26)
    mo = [P2.mo26_prox(pm, f) for f in range(band.F)]
    secs = time.process_time() - t0
    db = lambda preds: 10 * np.log10(np.mean([E.nmse(p, T[:, f]) for f, p in enumerate(preds)]))
    e_g, e_m = db(g['preds']), db([r['pred'] for r in mo])
    e_s = db([E.sinc_fit(p, 'evidence')['pred'] for p in band.probs])
    dist = float(np.linalg.norm(pg.cen[int(np.argmax(np.linalg.norm(Wg, axis=0)))] - src))
    lam = C0 / 500.0
    return dict(name='free-field monopole at a dictionary node (6 bins around 500 Hz, 48 microphones)',
                nmse_db=dict(gsmf_prox=e_g, mo26_prox=e_m, sinc=e_s), heaviest_region_distance_m=dist,
                quarter_wavelength_m=lam / 4, gsmf_regions=g['info']['regions'], dictionary=g['info']['dictionary'],
                delta=pg.delta, rho0_gsmf=pg.rho, rho0_mo26=pm.rho, mo26_regions=[r['info']['regions'] for r in mo],
                mo26_iterations=[r['info']['iterations'] for r in mo], gsmf_iterations=len(rec), process_seconds=round(secs, 1),
                passed=bool(e_g < e_s - 10 and e_m < e_s - 10 and dist < lam / 4))


def three_source_band():
    rng = np.random.default_rng(9)
    nodes = region_nodes(); M = 48
    mic = np.sort(rng.choice(len(nodes), M, replace=False)); tgt = np.setdiff1d(np.arange(len(nodes)), mic)
    srcs = [np.array([0.6, 0.6, -0.4]), np.array([1.6, 0.2, 0.5]), np.array([-0.5, 1.0, 0.9])]
    freqs = 500 * 2.0 ** ((np.arange(6) + 0.5) / 6 - 0.5)
    Y = []
    for f in freqs:
        k = 2 * np.pi * f / C0
        y = sum(a * np.exp(1j * k * np.linalg.norm(nodes[mic] - s, axis=1)) / np.linalg.norm(nodes[mic] - s, axis=1)
                for a, s in zip((1.0, 0.7, 0.5), srcs))
        Y.append(y + (rng.standard_normal(M) + 1j * rng.standard_normal(M)) / np.sqrt(2) * np.sqrt(np.mean(np.abs(y) ** 2)) * 1e-2)
    return v2.Band(nodes[mic], np.stack(Y, 1), nodes[tgt], freqs, C0, (nodes.min(0), nodes.max(0)))


def check_sparsity():
    """Group variant on three free-field sources. Where the fixed step converges (zeta in 1e-3 .. 1), the number of
    non-zero regions does not grow with zeta and stays sparse (at most 10 regions for 3 sources): the evidence term
    already prunes, which is the insensitivity to zeta the JASA paper reports. zeta = 100 and 1000 are recorded without a
    threshold (there the step's threshold eta*zeta is so large that 2,000 fixed steps do not settle)."""
    prep = P2.Prepared(three_source_band(), P2.GSMF)
    c = P2.GSMF
    counts = {}
    for z in ZETA_SPARSITY + ZETA_RECORDED:
        W = P2.stage_weights(prep.Bases, prep.ys, prep.Ks, z, c['eta'], c['iters'])
        counts[f'{z:g}'] = int(np.sum(np.any(W > 0, axis=0)))
    v = [counts[f'{z:g}'] for z in ZETA_SPARSITY]
    return dict(name='non-zero regions do not grow with zeta where the step converges, and stay sparse (three sources)',
                nonzero_regions=counts, passed=bool(all(a >= b for a, b in zip(v, v[1:])) and max(v) <= 10))


ZETA_SPARSITY = (1e-3, 1e-1, 1.0)
ZETA_RECORDED = (100.0, 1000.0)


CHECKS = ('check_paper_units', 'check_series', 'check_series_independent', 'check_target_cubature', 'check_weight_grad',
          'check_refine_grad', 'check_prox', 'check_mic_clearance', 'check_one_bin', 'check_free_field', 'check_sparsity')


def _run(name):
    import warnings
    warnings.simplefilter('ignore')
    try:
        return globals()[name]()
    except Exception:
        import traceback
        return dict(name=name, error=traceback.format_exc()[-1500:], passed=False)


SLOW = ('check_free_field', 'check_sparsity')     # behavioural checks: on a rented box they are not rerun (--fast);
                                                  # they run on the laptop at the registration commit


def main(out, fast=False):
    """Every check runs in its own process (one BLAS thread each), so the suite takes as long as its slowest check.
    fast: the correctness checks only (SLOW skipped)."""
    from multiprocessing import Pool
    names = [c for c in CHECKS if not (fast and c in SLOW)]
    with Pool(len(names)) as pool:
        checks = pool.map(_run, names, chunksize=1)
    res = dict(checks=checks, fast=fast, passed=all(c['passed'] for c in checks))
    Path(out).write_text(json.dumps(res, indent=1, default=float))
    for c in checks:
        print(('PASS ' if c['passed'] else 'FAIL ') + c['name'])
    return 0 if res['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1], fast='--fast' in sys.argv[2:]))
