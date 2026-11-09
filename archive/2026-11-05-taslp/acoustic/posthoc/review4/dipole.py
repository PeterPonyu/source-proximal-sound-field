"""Point-plus-dipole baseline on the anechoic grid (posthoc/review4/PLAN.md, part B). Post hoc, not registered.

`two_stage_dipole`: at the arrival-time centre of `two_stage_point` (unchanged), four rank-one terms, the normalized
monopole and its three normalized Cartesian source derivatives, each with a per-bin log weight, plus the per-bin log
ridge and the homogeneous sinc remainder of `oracle_fit`. Per-bin evidence ascent with the bounds, menu and optimizer
of `oracle_fit`; with the centre fixed and every parameter per bin, the band evidence separates into these per-bin
fits. No order selection and no scan.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
        python3 -B posthoc/review4/dipole.py check
    ... python3 -B posthoc/review4/dipole.py run OUT --workers N [--limit N]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
sys.path.insert(0, str(S / 'common')); sys.path.insert(0, str(S / 'studies/TB-M3-MIRACLE'))
import estimators as E  # noqa: E402
import run_m3 as RM  # noqa: E402

EXTRACT = S / 'studies/TB-M3-MIRACLE/extract'


def columns(pts, centre, k):
    """(P, 4): free-field Green function of a source at `centre` and its derivatives in the source coordinates."""
    diff = np.asarray(centre)[None, :] - pts                       # s - r
    d = np.linalg.norm(diff, axis=1)
    g = np.exp(1j * k * d) / d
    dg = (g * (1j * k - 1 / d) / d)[:, None] * diff
    return np.column_stack((g, dg))


def _parts(prob, centre, dipole):
    A = columns(prob.x, centre, prob.k)
    if not dipole:
        A = A[:, :1]
    nu = np.sqrt(np.mean(np.abs(A) ** 2, axis=0))
    return A / nu[None, :], nu


def objective(t, prob, b):
    J = b.shape[1]
    w, rho = np.exp(t[:J]), float(np.exp(t[J]))
    C = prob.H + (b * w[None, :]) @ b.conj().T + (rho + E.JITTER) * np.eye(prob.M)
    try:
        fac = cho_factor(C, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return 1e30, np.zeros_like(t)
    alpha = cho_solve(fac, prob.y)
    Q = float(np.vdot(prob.y, alpha).real)
    ev = -prob.M * np.log(Q) - 2 * float(np.sum(np.log(np.diag(fac[0]).real)))
    Cb = cho_solve(fac, b)
    ba = b.conj().T @ alpha
    g_w = w * (prob.M * np.abs(ba) ** 2 / Q - np.real(np.sum(b.conj() * Cb, axis=0)))
    trinv = float(np.real(np.trace(cho_solve(fac, np.eye(prob.M)))))
    g_rho = rho * (prob.M * float(np.vdot(alpha, alpha).real) / Q - trinv)
    return -ev, -np.r_[g_w, g_rho]


def fit(prob, centre, dipole=True, maxiter=200):
    b, nu = _parts(prob, centre, dipole)
    J = b.shape[1]
    starts = [np.r_[np.full(J, np.log(wn)), np.log(1e-2)] for wn in E.W_MENU]
    if dipole:                                                     # also start from the point-only optimum
        p = fit(prob, centre, dipole=False, maxiter=maxiter)
        starts.append(np.r_[np.log(p['w']), np.full(3, E.LOGW[0]), np.log(p['rho'])])
    best = None
    for t0 in starts:
        r = minimize(objective, t0, args=(prob, b), jac=True, method='L-BFGS-B',
                     bounds=[E.LOGW] * J + [E.LOGRHO], options=dict(maxiter=maxiter))
        if best is None or r.fun < best.fun:
            best = r
    w, rho = np.exp(best.x[:J]), float(np.exp(best.x[J]))
    C = prob.H + (b * w[None, :]) @ b.conj().T + (rho + E.JITTER) * np.eye(prob.M)
    alpha = cho_solve(cho_factor(C, lower=True), prob.y)
    bq = columns(prob.q, centre, prob.k)[:, :J] / nu[None, :]
    pred = prob.Hq @ alpha + bq @ (w * (b.conj().T @ alpha))
    return dict(w=w, rho=rho, evidence=-float(best.fun), pred=pred)


# ------------------------------------------------------------------ known-answer checks

def check():
    out = {}
    # 1. a pure dipole field on a synthetic grid is recovered (plan: relative error < 1e-6)
    rng = np.random.default_rng(7)
    g = np.stack(np.meshgrid(*[np.linspace(0, 1, 9)] * 2, indexing='ij'), -1).reshape(-1, 2)
    pts = np.c_[g, np.zeros(len(g))]
    centre = np.array([0.5, 0.5, -0.6])
    res = []
    for f in (250.0, 500.0, 1000.0):
        k = 2 * np.pi * f / 343.0
        p_true = columns(pts, centre, k)[:, 1:] @ np.array([1.0, -0.5j, 0.3])
        mic = np.sort(rng.choice(len(pts), 48, replace=False)); tgt = np.setdiff1d(np.arange(len(pts)), mic)
        prob = E.Problem(pts[mic], p_true[mic], pts[tgt], f, 343.0)
        r = fit(prob, centre, dipole=True)
        res.append(float(np.linalg.norm(r['pred'] - p_true[tgt]) / np.linalg.norm(p_true[tgt])))
    out['pure_dipole_rel_error'] = res
    out['pure_dipole_pass_1e-6'] = bool(max(res) < 1e-6)
    # 2. point-only mode equals oracle_fit at the arrival-time centre on 10 measured problems (plan: 1e-8)
    RM.init(str(EXTRACT), ['A1', 'R2'])
    T = RM.tasks(['A1', 'R2'])[::150][:10]
    dev, ev = [], []
    for sc, spk, fc, M, d, seed in T:
        band, Tt, ref, ts = RM.build(sc, spk, fc, M, seed)
        c2 = np.asarray(ts['centre'])
        for prob in band.probs[:2]:
            a = fit(prob, c2, dipole=False)
            o = E.oracle_fit(prob, c2[None, :])
            dev.append(float(np.linalg.norm(a['pred'] - o['pred']) / np.linalg.norm(o['pred'])))
            ev.append(abs(a['evidence'] - o['evidence']))
    out['point_only_vs_oracle_fit_rel_pred'] = dict(max=max(dev), median=float(np.median(dev)), n=len(dev))
    out['point_only_vs_oracle_fit_evidence_abs_diff_max'] = max(ev)
    out['point_only_pass_1e-8'] = bool(max(dev) < 1e-8)
    return out


# ------------------------------------------------------------------ run

def run_task(task):
    t0 = time.process_time()
    try:
        sc, spk, fc, M, d, seed = task
        band, T, ref, ts = RM.build(sc, spk, fc, M, seed)
        if not ts:
            return dict(kind='miracle', scenario=sc, source=spk, band=fc, M=M, draw=d, seed=seed, two_stage_ok=False)
        c2 = np.asarray(ts['centre'])
        pt, dp, wd = [], [], []
        for f, prob in enumerate(band.probs):
            a = fit(prob, c2, dipole=False); b = fit(prob, c2, dipole=True)
            pt.append(E.nmse(a['pred'], T[:, f])); dp.append(E.nmse(b['pred'], T[:, f]))
            wd.append([float(v) for v in np.log10(b['w'])] + [float(np.log10(b['rho'])), b['evidence'] - a['evidence']])
        return dict(kind='miracle', scenario=sc, source=spk, band=fc, M=M, draw=d, seed=seed, two_stage_ok=True,
                    nmse=dict(two_stage_point_refit=pt, two_stage_dipole=dp), info=dict(log10_w_rho_dev=wd),
                    process_seconds=round(time.process_time() - t0, 1))
    except Exception:
        return dict(task=list(task), error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('check', 'run')); ap.add_argument('out', nargs='?')
    ap.add_argument('--workers', type=int, default=1); ap.add_argument('--limit', type=int, default=None)
    a = ap.parse_args()
    if a.mode == 'check':
        r = check()
        (HERE / 'dipole-known-answer.json').write_text(json.dumps(r, indent=1))
        print(json.dumps(r, indent=1)); return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    T = RM.tasks(['A1', 'R2'])
    T = sorted(T, key=lambda t: -t[3])
    if a.limit:
        T = T[:a.limit]
    commit = subprocess.run(['git', '-C', str(S), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() or None
    (out / 'started.json').write_text(json.dumps(dict(
        tasks=len(T), workers=a.workers, commit=commit, started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        machine=dict(node=platform.node(), cpu=platform.processor(), numpy=np.__version__),
        extracts={s: hashlib.sha256((EXTRACT / f'miracle-{s}.npz').read_bytes()).hexdigest() for s in ('A1', 'R2')},
        inputs={n: hashlib.sha256((S / n).read_bytes()).hexdigest() for n in
                ('common/estimators.py', 'common/v2.py', 'common/problems_v2.py', 'studies/TB-M3-MIRACLE/run_m3.py',
                 'posthoc/review4/dipole.py', 'posthoc/review4/PLAN.md')}), indent=1))
    t0 = time.time(); n_err = 0
    with Pool(a.workers, initializer=RM.init, initargs=(str(EXTRACT), ['A1', 'R2'])) as pool, open(out / 'rows.jsonl', 'w') as fh:
        for n, row in enumerate(pool.imap_unordered(run_task, T), 1):
            n_err += 'error' in row
            fh.write(json.dumps(row, default=float) + '\n'); fh.flush()
            if n % 100 == 0 or n == len(T):
                print(f'{n}/{len(T)} tasks, {time.time() - t0:.0f} s, errors {n_err}', flush=True)
    (out / 'receipt.json').write_text(json.dumps(dict(finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'), tasks=len(T), errors=n_err,
                                                       wall_seconds=round(time.time() - t0),
                                                       rows_sha256=hashlib.sha256((out / 'rows.jsonl').read_bytes()).hexdigest()), indent=1))
    print('done', n_err, 'errors')


if __name__ == '__main__':
    main()
