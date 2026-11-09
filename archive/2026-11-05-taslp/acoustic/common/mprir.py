"""MP-RIR measured-arm preprocessing (geometry and transfer functions only; no reconstruction outcome here).

Dataset: Multi-Purpose Room Impulse Response dataset, Zenodo record 11148712 (CC BY 4.0). Grid: 1074 columns x 8
heights at 0.15 m, z = 0.70-1.75 m; impulse responses float32, 100096 samples at 48 kHz. Setup.npz is read only
through a restricted unpickler; .npy files are memory-mapped with allow_pickle=False.

Convention: the estimators use exp(+i k r)/r, i.e. time dependence exp(-i omega t). numpy's FFT uses exp(-i omega t)
inside the transform, which maps a delay to exp(-i k r); the transfer function is therefore the complex conjugate of
numpy's rfft bin.
"""
from __future__ import annotations
import io
import pickle
import zipfile
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import hilbert, butter, sosfiltfilt

DATA = Path('/home/zeyufu/Desktop/data/external/acoustic-geometry-20261104/mp-rir')
FS = 48000
N = 100096
ALLOWED = {('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'), ('numpy', 'ndarray'),
           ('numpy', 'dtype'), ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar'),
           ('builtins', 'dict'), ('collections', 'OrderedDict')}


class Restricted(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'blocked {module}.{name}')


def load_setup(path=DATA / 'Setup.npz'):
    out = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            raw = z.read(n)
            try:
                arr = np.lib.format.read_array(io.BytesIO(raw), allow_pickle=False)
            except ValueError:
                f = io.BytesIO(raw); ver = np.lib.format.read_magic(f)
                np.lib.format._read_array_header(f, ver)
                arr = Restricted(f).load()
            if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
                arr = arr.item()
            out[n[:-4]] = arr
    return out


def geometry():
    return np.load(DATA / 'Mxyz.npy', allow_pickle=False)          # (1074, 8, 3)


def rirs(source, path=None):
    return np.load(path or (DATA / f'{source}_Mrir.npy'), mmap_mode='r', allow_pickle=False)   # (1074, 8, N)


def arrival_times(h, t_system, window_ms=25.0, band=(500.0, 8000.0), up=8):
    """Direct-sound arrival per impulse response: peak of the Hilbert envelope of the band-passed response in the
    window [t_system, t_system + window_ms], refined by 8x Fourier upsampling. Returns seconds after t_system."""
    w = int(window_ms * 1e-3 * FS)
    seg = np.asarray(h[..., int(t_system):int(t_system) + w + 512], float)
    sos = butter(4, band, btype='band', fs=FS, output='sos')
    seg = sosfiltfilt(sos, seg, axis=-1)[..., :w]
    from scipy.signal import resample
    env = np.abs(hilbert(resample(seg, w * up, axis=-1), axis=-1))
    first = np.argmax(env > 0.5 * env.max(axis=-1, keepdims=True), axis=-1)   # earliest crossing of half the peak
    return first / (FS * up)


def fit_centre(xyz, t, s0, c0=343.0):
    """Robust (soft L1) fit of t_m = delta + |x_m - s| / c for the acoustic centre s, the sound speed c and a constant
    onset offset delta (the arrival detector fires a fixed time before the true arrival)."""
    def res(p):
        return p[4] + np.linalg.norm(xyz - p[:3], axis=1) / p[3] - t
    r = least_squares(res, np.r_[s0, c0, 0.0], loss='soft_l1', f_scale=2e-5)
    rr = res(r.x)
    J = r.jac
    cov = np.linalg.pinv(J.T @ J) * float(np.mean(rr ** 2))
    return dict(centre=r.x[:3], c=float(r.x[3]), offset_s=float(r.x[4]), residual_rms_s=float(np.sqrt(np.mean(rr ** 2))),
                residual_rms_mm=float(np.sqrt(np.mean(rr ** 2)) * r.x[3] * 1e3), n=len(t),
                centre_sd_m=np.sqrt(np.diag(cov)[:3]).tolist(), c_sd=float(np.sqrt(cov[3, 3])))


def box_columns(xy_cols, centre_xy, n=9, spacing=0.15, mode='centred'):
    """Indices (row-major n x n) of a complete n x n block of grid columns.

    mode 'centred': the block whose centre is nearest to centre_xy (source under the grid).
    mode 'adjacent': among complete blocks that do not contain centre_xy in their footprint, the one whose footprint
    is nearest to centre_xy (source beside the grid)."""
    key = {tuple(np.round(p / spacing).astype(int)): i for i, p in enumerate(xy_cols)}
    ij = np.array(list(key))
    best = None
    for i0, j0 in ij:
        idx = [key.get((i0 + a, j0 + b)) for a in range(n) for b in range(n)]
        if any(v is None for v in idx):
            continue
        pts = xy_cols[idx]
        lo, hi = pts.min(0), pts.max(0)
        if mode == 'centred':
            score = float(np.linalg.norm((lo + hi) / 2 - centre_xy))
        else:
            if np.all(centre_xy >= lo) and np.all(centre_xy <= hi):
                continue
            score = float(np.linalg.norm(np.maximum(np.maximum(lo - centre_xy, centre_xy - hi), 0)))
            score += 1e-3 * float(np.linalg.norm((lo + hi) / 2 - centre_xy))   # tie-break towards the facing block
        if best is None or score < best[0]:
            best = (score, idx)
    return np.array(best[1])


def frequency_bins(centre, per_band=12):
    """Bins log-spaced strictly inside the octave band [centre / sqrt 2, centre * sqrt 2], rounded to FFT bins."""
    f = np.geomspace(centre / np.sqrt(2), centre * np.sqrt(2), per_band + 2)[1:-1]
    b = np.unique(np.round(f * N / FS).astype(int))
    return b, b * FS / N


def transfer(h_block, bins):
    """Transfer functions (exp(+i k r) convention) of impulse responses (..., N) at the given rfft bins."""
    X = np.fft.rfft(np.asarray(h_block, np.float64), axis=-1)
    return np.conj(X[..., bins])


def schroeder_t30(h, band, t_system):
    sos = butter(4, [band / np.sqrt(2), band * np.sqrt(2)], btype='band', fs=FS, output='sos')
    x = sosfiltfilt(sos, np.asarray(h, float))[int(t_system):]
    e = np.cumsum(x[::-1] ** 2)[::-1]
    L = 10 * np.log10(e / e[0] + 1e-30)
    i5, i35 = np.argmax(L <= -5), np.argmax(L <= -35)
    t = np.arange(len(L)) / FS
    slope = np.polyfit(t[i5:i35], L[i5:i35], 1)[0]
    return float(-60 / slope)
