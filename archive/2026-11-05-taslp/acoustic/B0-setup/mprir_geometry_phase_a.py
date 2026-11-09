"""MP-RIR geometry check: is any loudspeaker inside the hull of the measured microphones?"""
import io, json, pickle, sys, zipfile
import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree

ALLOWED = {('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
           ('numpy', 'ndarray'), ('numpy', 'dtype'), ('numpy.core.multiarray', 'scalar'),
           ('numpy._core.multiarray', 'scalar'), ('builtins', 'dict'), ('collections', 'OrderedDict')}

class Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'blocked {module}.{name}')

def load_npz_restricted(path):
    out = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            raw = z.read(n)
            try:
                arr = np.lib.format.read_array(io.BytesIO(raw), allow_pickle=False)
            except ValueError:  # object array: parse header, then restricted unpickle
                f = io.BytesIO(raw); ver = np.lib.format.read_magic(f)
                np.lib.format._read_array_header(f, ver)
                arr = Restricted(f).load()
            out[n[:-4]] = arr
    return out

mxyz = np.load(sys.argv[1], allow_pickle=False)
setup = load_npz_restricted(sys.argv[2])
pts = mxyz.reshape(-1, 3)
print('mic grid shape', mxyz.shape, 'points', len(pts))
print('extent min', pts.min(0).round(3), 'max', pts.max(0).round(3))
xy = mxyz[:, 0, :2]; tree = cKDTree(xy); d, _ = tree.query(xy, k=2)
print('xy nearest-neighbour spacing: median %.3f m, max %.3f m' % (np.median(d[:, 1]), d[:, 1].max()))
z = np.sort(np.unique(mxyz[0, :, 2].round(4))); print('z levels', z, 'dz', np.diff(z).round(3))
for k, v in setup.items():
    if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
        v = v.item()
    print('setup', k, type(v).__name__, v if not isinstance(v, np.ndarray) or v.size < 20 else v.shape)
spk = setup['coord_speaker_center']
spk = spk.item() if isinstance(spk, np.ndarray) else spk
hull3 = Delaunay(pts); hull2 = Delaunay(xy); tree3 = cKDTree(pts)
for f in (250., 500., 1000.):
    print('lambda/2 at %d Hz = %.3f m' % (f, 343 / f / 2))
rows = []
for name, c in spk.items():
    c = np.asarray(c, float)
    inside3 = bool(hull3.find_simplex(c) >= 0); inside2 = bool(hull2.find_simplex(c[:2]) >= 0)
    dist, _ = tree3.query(c, k=24)
    within = [int((np.linalg.norm(pts - c, axis=1) <= r).sum()) for r in (0.5, 1.0)]
    rows.append(dict(speaker=name, coord=c.round(3).tolist(), inside_3d_hull=inside3, inside_xy_hull=inside2,
                     nearest_mic_m=round(float(dist[0]), 3), dist_24th_mic_m=round(float(dist[-1]), 3),
                     mics_within_0p5m=within[0], mics_within_1m=within[1]))
for r in rows: print(json.dumps(r))
