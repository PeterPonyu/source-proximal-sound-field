"""RAF interior-emitter check: per pose group (position 1 cm, quaternion 0.02), is the emitter inside the hull of
its own receivers, and is the neighbourhood (0.5 m ball inside the hull) sampled at <= lambda/2 (covering radius <= lambda/4)?"""
import os
from pathlib import Path
import json, sys
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.cluster.hierarchy import fcluster, linkage
D = str(Path(os.environ.get('ACOUSTIC_DATA', 'data')) / 'raf') + '/'
out = {}
for room, pre in (('emptyroom', 'emptyroom-EmptyRoom'), ('furnishedroom', 'furnishedroom-FurnishedRoom')):
    tx = np.loadtxt(D + pre + '_metadata_all_tx_pos.txt', delimiter=',')
    rx = np.loadtxt(D + pre + '_metadata_all_rx_pos.txt', delimiter=',')[:len(tx)]
    q = tx[:, :4] * np.sign(tx[:, 3:4] + 1e-12)
    u, inv = np.unique(np.round(np.hstack([tx[:, 4:] / 0.01, q / 0.02]), 0), axis=0, return_inverse=True)
    g = fcluster(linkage(u, 'single'), 1.5, 'distance')[inv]
    rng = np.random.default_rng(0)
    rows = []
    for c in range(1, g.max() + 1):
        sel = g == c; cen = tx[sel, 4:].mean(0); r = rx[sel]
        lr = fcluster(linkage(r, 'single'), 0.02, 'distance'); r = np.array([r[lr == k].mean(0) for k in range(1, lr.max() + 1)])
        row = dict(n_rx=len(r), inside=False)
        if len(r) >= 24:
            h = Delaunay(r); row['inside'] = bool(h.find_simplex(cen) >= 0)
            if row['inside']:
                v = rng.normal(size=(4000, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
                ball = cen + v * 0.5 * rng.random((4000, 1)) ** (1 / 3); ball = ball[h.find_simplex(ball) >= 0]
                row['covering_radius_m'] = float(cKDTree(r).query(ball)[0].max())
        rows.append(row)
    ins = [r for r in rows if r['inside']]
    cov = sorted(r['covering_radius_m'] for r in ins)
    out[room] = dict(rows=len(tx), pose_groups=len(rows), receivers_per_group_median=float(np.median([r['n_rx'] for r in rows])),
                     inside_own_hull=len(ins), best_covering_radius_m=cov[0] if cov else None,
                     qualifying_250Hz=sum(c <= 343 / 250 / 4 for c in cov), qualifying_125Hz=sum(c <= 343 / 125 / 4 for c in cov))
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[1], 'w'), indent=1)
