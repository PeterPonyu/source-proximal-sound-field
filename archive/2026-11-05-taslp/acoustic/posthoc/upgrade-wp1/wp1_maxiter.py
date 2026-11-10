"""WP1 follow-up (post hoc, PLAN.md addendum): baseline K = 1 refinement with the iteration cap raised 200 -> 5000,
everything else unchanged, to separate the iteration cap from the bound start."""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wp1  # noqa: E402

def one(seed):
    band, T, src = wp1.make_band(seed)
    lo, hi = band.search_box()
    nodes = wp1.E.grid_nodes(lo, hi, 0.25 * band.lam_mid(), max_nodes=wp1.v2.SCAN_NODES)
    R0 = np.array([wp1.v2.remainder_rho(p) for p in band.probs])
    cands = wp1.v2.band_scan(band, nodes, np.zeros((0, 3)), np.zeros((band.F, 0)), R0, 3, 0.5 * band.lam_mid())
    runs = [wp1.refine(band, np.concatenate((n, np.log(np.maximum(w, 1e-4)), np.log(R0))), maxiter=5000) for n, w, _ in cands]
    r, th = min(runs, key=lambda z: z[0].fun)
    m = wp1.model_of(band, th)
    return dict(seed=seed, nit=int(r.nit), msg=str(r.message), fun=float(r.fun), db=wp1.errs_db(band, m, T),
                log10w=np.log10(m['W'][:, 0]).tolist())

if __name__ == '__main__':
    with Pool(10, initializer=wp1.RP.init) as pool, open(sys.argv[1], 'w') as fh:
        for row in pool.imap_unordered(one, list(wp1.RP.SEEDS)):
            fh.write(json.dumps(row) + '\n')
