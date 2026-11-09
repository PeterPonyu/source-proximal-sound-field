"""Monte Carlo check of the bound on synthetic data drawn from the model itself (no study data).

A ball of radius 0.5 m with 48 random microphones, a source 1.1 m from the centre, data drawn from
y = a g(s) + e, e ~ CN(0, sigma^2 (K + rho I)); s estimated by maximizing the profiled likelihood (amplitude and
sigma^2 profiled, rho known) from the true position. The empirical mean squared error should approach tr J^{-1}
at high power ratio, for one bin and for six bins sharing the coordinates.
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).parent))
import crb

C = 343.0


def geometry(seed, M=48, a=0.5):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(M, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
    x = v * a * rng.random(M)[:, None] ** (1 / 3)
    s = np.array([1.1, 0.0, 0.0])
    return x, s


def fit(x, Ys, ks, rho, s0):
    Ls = [cholesky(crb.sinc_kernel(x, k) + rho * np.eye(len(x)), lower=True) for k in ks]

    def nll(s):
        tot = 0.0
        for y, k, L in zip(Ys, ks, Ls):
            G, _ = crb.green(x, s, k)
            u = solve_triangular(L, G, lower=True); z = solve_triangular(L, y, lower=True)
            a = (u.conj() @ z) / np.real(u.conj() @ u)
            tot += len(x) * np.log(np.real(np.vdot(z - a * u, z - a * u)))
        return tot
    return minimize(nll, s0, method='Nelder-Mead', options=dict(xatol=1e-7, fatol=1e-10, maxiter=4000)).x


def main():
    out = {}
    x, s = geometry(1)
    fc = 500.0
    freqs = fc * 2.0 ** ((np.arange(6) + 0.5) / 6 - 0.5)
    rho, eta = 1e-2, 10.0
    for label, fs in (('one bin', freqs[2:3]), ('six bins', freqs)):
        ks = 2 * np.pi * fs / C
        J = sum(crb.info(x, s, k, eta, rho) for k in ks)
        rng = np.random.default_rng(7)
        errs = []
        for t in range(300):
            Ys = []
            for k in ks:
                G, _ = crb.green(x, s, k)
                nu = np.sqrt(np.mean(np.abs(G) ** 2))
                L = cholesky(crb.sinc_kernel(x, k) + rho * np.eye(len(x)), lower=True)
                e = L @ ((rng.normal(size=len(x)) + 1j * rng.normal(size=len(x))) / np.sqrt(2))
                a = np.sqrt(eta) / nu * np.exp(1j * rng.uniform(0, 2 * np.pi))     # sigma^2 = 1
                Ys.append(a * G + e)
            errs.append(fit(x, Ys, ks, rho, s) - s)
        errs = np.array(errs)
        out[label] = dict(bound_rms_m=crb.rms_bound(J), empirical_rms_m=float(np.sqrt(np.mean(np.sum(errs ** 2, 1)))),
                          trials=len(errs))
    out['ratio_bound'] = out['six bins']['bound_rms_m'] / out['one bin']['bound_rms_m']
    print(json.dumps(out, indent=1))
    (Path(__file__).parent / 'check-bound.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
