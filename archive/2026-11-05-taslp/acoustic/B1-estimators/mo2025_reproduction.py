"""Reproduction check of Matsuda & Otani (2025), Sec. V A, 2500 Hz: reported NMSE -21.7 dB (proposed Gs, Opt),
+7.8 dB (Gs, Init), -18.7 dB (uniform, Opt). Development seeds 400300-400309 (noise and evaluation points).

Deviations forced by missing details, stated in advance: a 64-point Fibonacci sphere replaces their 64-point
spherical t-design; sigma = R/3 (their earlier figures); 10 noise/evaluation draws are averaged in dB because
the paper reports one unnamed draw. Tolerance: within 3 dB of each reported value.
Run: OPENBLAS_NUM_THREADS=1 python3 -B mo2025_reproduction.py OUT.json
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import mo2025 as MO  # noqa: E402

c, f = 340.0, 2500.0
k = 2 * np.pi * f / c
sph = lambda r, th, ph: np.array([r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)])
src = [sph(0.72, np.pi / 2, 0.0), sph(0.57, np.pi / 2, 2 * np.pi / 3)]
amp = [1.0, 0.5j]
prior = [(sph(0.6, np.pi / 2, np.pi / 24), 0.2), (sph(0.55, 35 * np.pi / 72, 7 * np.pi / 12), 0.2)]
x = MO.fib(64) * 1.0
field = lambda p: sum(a * MO.green4pi(p, s[None, :], k)[:, 0] for s, a in zip(src, amp))
rows = []
for seed in range(400300, 400310):
    rng = np.random.default_rng(seed)
    u = field(x)
    sy2 = float(np.mean(np.abs(u) ** 2))
    s2 = sy2 / (10 ** (20 / 10) + 1)            # noise variance for SNR 20 dB on the noisy-signal power convention
    y = u + np.sqrt(s2 / 2) * (rng.normal(size=64) + 1j * rng.normal(size=64))
    s2_est = float(np.mean(np.abs(y) ** 2)) / (10 ** (20 / 10) + 1)
    v = rng.normal(size=(4000, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
    q = (v * rng.random((4000, 1)) ** (1 / 3))[:500]
    t = field(q)
    nm = lambda p: 10 * np.log10(float(np.sum(np.abs(p - t) ** 2) / np.sum(np.abs(t) ** 2)))
    init = nm(MO.predict(x, y, q, k, prior, s2_est))
    opt_regions = MO.fit_grid(x, y, k, prior, s2_est)
    opt = nm(MO.predict(x, y, q, k, opt_regions, s2_est))
    row = dict(seed=seed, gs_init_db=init, gs_opt_db=opt, opt_regions=[(np.round(c_, 3).tolist(), R) for c_, R in opt_regions])
    rows.append(row); print(row, flush=True)
gi = float(np.mean([r['gs_init_db'] for r in rows])); go = float(np.mean([r['gs_opt_db'] for r in rows]))
res = dict(reported=dict(gs_opt_db=-21.7, gs_init_db=7.8), reproduced_mean_db=dict(gs_opt_db=go, gs_init_db=gi),
           per_draw=rows, tolerance_db=3.0, passed_opt=bool(abs(go + 21.7) <= 3.0), passed_init=bool(abs(gi - 7.8) <= 3.0),
           deviations=['64-point Fibonacci sphere instead of the spherical t-design', 'sigma = R/3 assumed',
                       'mean over 10 noise and evaluation draws', 'kernel integral by quadrature instead of the series'])
Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
print('reproduced Opt %.1f dB (reported -21.7), Init %.1f dB (reported +7.8)' % (go, gi))
