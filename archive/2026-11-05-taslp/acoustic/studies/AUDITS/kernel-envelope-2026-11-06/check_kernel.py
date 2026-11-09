"""Envelope check of the exact uniform-ball kernel (PLAN.md; post hoc, kernel only, synthetic geometry).

C1  series orders 40/60/80 in 50-digit arithmetic, microphone-microphone block and the series branch of the
    target-microphone block (targets 2.1 cm outside the ball), against the registered double-precision order 40;
C2  the registered inner cubature branch (24 x 1,200) against a target-centred reference quadrature, and the review's
    48 x 4,800 centre-based cubature (descriptive);
C3  analytic against finite-difference gradients of the exact-kernel objective near the soft microphone guard.

The kernel code of the exact-kernel follow-ups is imported unchanged from V2-dev/cubature_diag.py.

    OPENBLAS_NUM_THREADS=1 python3 -B studies/AUDITS/kernel-envelope-2026-11-06/check_kernel.py --workers 4
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
import mpmath as mp
import numpy as np

HERE = Path(__file__).resolve().parent
S = HERE.parents[2]
sys.path.insert(0, str(S / 'common')); sys.path.insert(0, str(S / 'V2-dev'))
import cubature_diag as CD  # noqa: E402
import estimators as E  # noqa: E402
import mo2025 as MO  # noqa: E402

TOL = dict(C1=1e-6, C2=1e-2, C3=1e-6, C2_reference_self=1e-4)       # PLAN.md
RADII = (0.005, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
KS = (1.786, 3.111, 6.250, 12.50, 24.97)
CLEAR = (0.045, 0.049, 0.05, 0.051, 0.06, 0.1, 0.3)
NTOP = 80
mp.mp.dps = 50


# ------------------------------------------------------------------ geometry

def unit(rng, n):
    v = rng.standard_normal((n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


def geometry(iR, ik):
    R = RADII[iR]
    rng = np.random.default_rng(np.random.SeedSequence([20261106, iR, ik]))
    dirs, rad = [], []
    for c in CLEAR:
        for d in unit(rng, 2):
            dirs.append(d); rad.append(R + c)
    d0 = dirs[0]
    for _ in range(2):                                   # two more at 0.045 m within 10 degrees of the first
        u = rng.standard_normal(3); u -= (u @ d0) * d0; u /= np.linalg.norm(u)
        th = np.deg2rad(rng.uniform(2.0, 10.0))
        dirs.append(np.cos(th) * d0 + np.sin(th) * u); rad.append(R + 0.045)
    x = np.array(dirs) * np.array(rad)[:, None]
    tq = []
    for r, n in ((R + 0.021, 2), (R + 0.019, 2), (R + 0.005, 2), (0.99 * R, 2), (0.9 * R, 2), (0.5 * R, 2), (0.0, 1)):
        for d in unit(rng, n):
            tq.append((r, d * r))
    return R, x, tq


# ------------------------------------------------------------------ C1: the series in 50-digit arithmetic

def sph(n, x, kind):
    f = mp.besselj if kind == 'j' else mp.bessely
    return mp.sqrt(mp.pi / (2 * x)) * f(n + mp.mpf(1) / 2, x)


def mp_vectors(radii, k, R):
    """V[n, i] = sqrt(|a_n|) h_n(k r_i) for n = 0..NTOP (complex128) and the signs of a_n."""
    k = mp.mpf(k); x = k * mp.mpf(R)
    jn = [mp.cos(x) / x] + [sph(n, x, 'j') for n in range(NTOP + 2)]          # j_{-1} .. j_{NTOP+1}
    a = [mp.mpf(3) / 2 * k ** 2 * (2 * n + 1) * (jn[n + 1] ** 2 - jn[n] * jn[n + 2]) for n in range(NTOP + 1)]
    sgn = np.array([1.0 if v >= 0 else -1.0 for v in a])
    V = np.empty((NTOP + 1, len(radii)), complex)
    for i, r in enumerate(radii):
        kr = k * mp.mpf(r)
        for n in range(NTOP + 1):
            h = mp.mpc(sph(n, kr, 'j'), sph(n, kr, 'y'))
            v = mp.sqrt(abs(a[n])) * h
            V[n, i] = complex(v)
    return V, sgn


def legendre(u, N):
    P = [np.ones_like(u), u]
    for n in range(1, N):
        P.append(((2 * n + 1) * u * P[n] - n * P[n - 1]) / (n + 1))
    return P


def series_orders(Vr, Vc, sgn, U, orders=(40, 60, 80)):
    P = legendre(U, NTOP)
    out, acc = {}, np.zeros(U.shape, complex)
    for n in range(NTOP + 1):
        acc = acc + sgn[n] * np.outer(Vr[n], Vc[n].conj()) * P[n]
        if n in orders:
            out[n] = acc.copy()
    return out


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ------------------------------------------------------------------ C2: target-centred reference quadrature

def ref_row(q, x, R, k, n_rho, n_ang, n_phi):
    """K(q, x_j) = (1/|B|) int_B G(q, s) conj(G(x_j, s)) ds, G = exp(ikd)/d, ball centred at the origin."""
    D = float(np.linalg.norm(q))
    phi = 2 * np.pi * np.arange(n_phi) / n_phi; wphi = 2 * np.pi / n_phi
    gx, gw = np.polynomial.legendre.leggauss(n_rho)
    if D < R:                                            # inside: full sphere of directions about q
        mu, wmu = np.polynomial.legendre.leggauss(n_ang)
        st = np.sqrt(1 - mu ** 2)
        om = np.stack([np.outer(st, np.cos(phi)), np.outer(st, np.sin(phi)), np.outer(mu, np.ones(n_phi))], -1).reshape(-1, 3)
        wom = np.outer(wmu, np.full(n_phi, wphi)).reshape(-1)
        b = om @ q
        lo = np.zeros(len(om)); hi = -b + np.sqrt(b ** 2 - (D ** 2 - R ** 2))
    else:                                                # outside: the cap of directions that hit the ball
        e = -q / D
        t1 = np.cross(e, [1.0, 0, 0] if abs(e[0]) < 0.9 else [0, 1.0, 0]); t1 /= np.linalg.norm(t1); t2 = np.cross(e, t1)
        psi, wpsi = np.polynomial.legendre.leggauss(n_ang)
        psi = 0.25 * np.pi * (psi + 1); wpsi = 0.25 * np.pi * wpsi
        sth = (R / D) * np.sin(psi); cth = np.sqrt(1 - sth ** 2)
        dth = (R / D) * np.cos(psi) / cth                # d theta / d psi
        om = (cth[:, None, None] * e + sth[:, None, None] * (np.cos(phi)[None, :, None] * t1 + np.sin(phi)[None, :, None] * t2)).reshape(-1, 3)
        wom = np.outer(wpsi * dth * sth, np.full(n_phi, wphi)).reshape(-1)
        mid = np.repeat(D * cth, n_phi); half = np.repeat(R * np.cos(psi), n_phi)
        lo, hi = mid - half, mid + half
    rho = 0.5 * (hi - lo)[:, None] * (gx[None, :] + 1) + lo[:, None]
    wr = 0.5 * (hi - lo)[:, None] * gw[None, :]
    s = q[None, None, :] + rho[:, :, None] * om[:, None, :]
    vol = 4 * np.pi * R ** 3 / 3
    base = (np.exp(1j * k * rho) * rho * wr * wom[:, None] / vol).reshape(-1)       # G(q,s) rho^2 = exp(ik rho) rho
    sp = s.reshape(-1, 3)
    out = np.empty(len(x), complex)
    for j, xj in enumerate(x):
        d = np.linalg.norm(sp - xj, axis=1)
        out[j] = np.sum(base * np.exp(-1j * k * d) / d)
    measure = float(np.sum((rho ** 2 * wr * wom[:, None])))                          # equals |B| if exact
    return out, measure / vol - 1.0


def centred_cubature(q, x, R, k, n_r, n_a):
    nodes, w = MO.ball_quadrature(R, k, n_r=n_r, n_a=n_a, density='uniform')
    Fq = E.green(q, nodes, k) * np.sqrt(w)
    Fx = E.green(x, nodes, k) * np.sqrt(w)
    return Fq @ Fx.conj().T


# ------------------------------------------------------------------ one (k, R) pair: C1 and C2

def pair(args):
    iR, ik = args
    k = KS[ik]
    R, x, tq = geometry(iR, ik)
    c = np.zeros(3)
    out = dict(R=R, k=k, kR=k * R)
    # C1: microphone block and the series branch of the target block
    rx = np.linalg.norm(x, axis=1); dx = x / rx[:, None]
    far = [(r, q) for r, q in tq if r > R + 0.02]
    rq = np.array([r for r, _ in far]); qf = np.array([q for _, q in far]); dq = qf / rq[:, None]
    V, sgn = mp_vectors(list(rx) + list(rq), k, R)
    Vx, Vq = V[:, :len(rx)], V[:, len(rx):]
    Kxx = series_orders(Vx, Vx, sgn, np.clip(dx @ dx.T, -1, 1))
    Kqx = series_orders(Vq, Vx, sgn, np.clip(dq @ dx.T, -1, 1))
    reg_xx = CD.series_K(x, c, R, k)
    reg_qx = CD.cross_K(qf, x, c, R, k)
    out['C1'] = dict(
        mic_80_40=rel(Kxx[40], Kxx[80]), mic_80_60=rel(Kxx[60], Kxx[80]), mic_reg40_vs_mp40=rel(reg_xx, Kxx[40]),
        tgt_80_40=rel(Kqx[40], Kqx[80]), tgt_80_60=rel(Kqx[60], Kqx[80]), tgt_reg40_vs_mp40=rel(reg_qx, Kqx[40]),
        a_negative=int(np.sum(sgn < 0)))
    out['C1']['pass'] = bool(max(out['C1']['mic_80_40'], out['C1']['tgt_80_40']) <= TOL['C1'])
    # C2: the inner cubature branch
    near = [(r, q) for r, q in tq if r <= R + 0.02]
    qn = np.array([q for _, q in near])
    reg = CD.cross_K(qn, x, c, R, k)
    fr = [ref_row(q, x, R, k, 64, 64, 128) for q in qn]
    fine = np.array([v for v, _ in fr]); meas = max(abs(m) for _, m in fr)
    coarse = np.array([ref_row(q, x, R, k, 48, 48, 96)[0] for q in qn])
    cen = centred_cubature(qn, x, R, k, 48, 4800)
    per_target = [dict(r_over_R=float(r / R), rel=rel(reg[i:i + 1], fine[i:i + 1]), rel_centred_48x4800=rel(cen[i:i + 1], fine[i:i + 1]))
                  for i, (r, _) in enumerate(near)]
    out['C2'] = dict(reg_vs_ref=rel(reg, fine), centred_48x4800_vs_ref=rel(cen, fine), ref_self=rel(coarse, fine),
                     ref_measure_err=meas, per_target=per_target)
    out['C2']['ref_ok'] = bool(out['C2']['ref_self'] <= TOL['C2_reference_self'])
    out['C2']['pass'] = bool(out['C2']['reg_vs_ref'] <= TOL['C2'])
    return out


# ------------------------------------------------------------------ C3: gradients near the guard

def gradient_case(R, k, clear, J):
    rng = np.random.default_rng(np.random.SeedSequence([20261107, int(R * 1000), int(k * 100), int(clear * 1000), J]))
    cens = [np.zeros(3), np.array([1.6, 0.3, -0.2])][:J]
    rads = [R, 0.3][:J]
    xs = []
    for cj, rj in zip(cens, rads):
        for c in (clear, clear, 0.08, 0.15, 0.3, 0.6):
            xs.append(cj + unit(rng, 1)[0] * (rj + c))
    x = np.array(xs)
    y = rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x))
    q = x[:2] + 0.01
    prob = E.Problem(x, y, q, k * 343.0 / (2 * np.pi), 343.0)
    t = np.r_[np.array(cens).ravel(), np.log(rads), np.log([2.0, 0.7][:J]), np.log(1e-2)]
    f0, g = CD.mo26_objective_series(t, prob, J)
    fd = np.empty_like(t); h = 1e-6
    for i in range(len(t)):
        tp, tm = t.copy(), t.copy(); tp[i] += h; tm[i] -= h
        fd[i] = (CD.mo26_objective_series(tp, prob, J)[0] - CD.mo26_objective_series(tm, prob, J)[0]) / (2 * h)
    dist = min(float(np.min(np.linalg.norm(x - cj, axis=1) - rj)) for cj, rj in zip(cens, rads))
    err = float(np.max(np.abs(g - fd)) / np.max(np.abs(fd)))
    return dict(R=R, k=k, kR=k * R, clearance=clear, balls=J, min_clearance=dist, penalty_active=bool(dist < E.GUARD),
                objective=float(f0), rel_err=err, pass_=bool(err <= TOL['C3']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    pairs = [(iR, ik) for iR in range(len(RADII)) for ik in range(len(KS))]
    with Pool(a.workers) as pool:
        res = pool.map(pair, pairs, chunksize=1)
    grads = [gradient_case(R, k, cl, J) for (R, k) in ((0.5, 24.97), (0.005, 1.786)) for cl in (0.049, 0.051) for J in (1, 2)]
    summary = dict(
        tolerances=TOL, pairs=len(res), gradient_cases=len(grads),
        C1_pass=all(r['C1']['pass'] for r in res), C2_pass=all(r['C2']['pass'] for r in res),
        C2_reference_ok=all(r['C2']['ref_ok'] for r in res), C3_pass=all(g['pass_'] for g in grads),
        C1_max_mic_80_40=max(r['C1']['mic_80_40'] for r in res), C1_max_tgt_80_40=max(r['C1']['tgt_80_40'] for r in res),
        C1_max_80_60=max(max(r['C1']['mic_80_60'], r['C1']['tgt_80_60']) for r in res),
        C1_max_reg40_vs_mp40=max(max(r['C1']['mic_reg40_vs_mp40'], r['C1']['tgt_reg40_vs_mp40']) for r in res),
        C2_max_reg_vs_ref=max(r['C2']['reg_vs_ref'] for r in res),
        C2_max_centred_48x4800_vs_ref=max(r['C2']['centred_48x4800_vs_ref'] for r in res),
        C2_max_ref_self=max(r['C2']['ref_self'] for r in res), C3_max_rel_err=max(g['rel_err'] for g in grads),
        seconds=round(time.time() - t0, 1))
    summary['all_pass'] = bool(summary['C1_pass'] and summary['C2_pass'] and summary['C2_reference_ok'] and summary['C3_pass'])
    (HERE / 'kernel-envelope.json').write_text(json.dumps(dict(summary=summary, pairs=res, gradients=grads), indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
