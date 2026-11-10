"""E2 follow-up (labelled): the worst E2 cases were targets outside the ball, where the exact formula is the existing
series (E4, 2e-16). Test whether the deviation is the reference's angular resolution: rerun those cases with 5,000,
20,000 and 80,000 directions, and inside-ball cases likewise."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exact_ball as XB
d = json.load(open('checks.json'))
cases = sorted(d['E2_cases'], key=lambda r: -r['rel'])[:4] + [r for r in d['E2_cases'] if 0 < r['rq_over_R'] < 1][:3]
rng = np.random.default_rng(5); out = []
for r in cases:
    k = 2 * np.pi * r['f'] / 343.0; R = r['R']; c = np.zeros(3)
    dd = rng.standard_normal((6, 3)); dd /= np.linalg.norm(dd, axis=1)[:, None]
    x = dd * (R + rng.uniform(0.06, 0.6, 6))[:, None]
    dq = rng.standard_normal(3); dq /= np.linalg.norm(dq); q = c + r['rq_over_R'] * R * dq
    ex = XB.cross_K(q[None], x, c, R, k)[0]
    errs = {nd: float(np.linalg.norm(ex - XB.reference_cross_K(q, x, c, R, k, n_dir=nd)) / np.linalg.norm(ex)) for nd in (5000, 20000, 80000)}
    out.append(dict(f=r['f'], R=R, rq_over_R=r['rq_over_R'], rel_by_directions=errs))
    print(out[-1], flush=True)
Path('checks-e2-refine.json').write_text(json.dumps(out, indent=1))
