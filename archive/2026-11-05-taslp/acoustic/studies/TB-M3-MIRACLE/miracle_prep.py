"""TB-M3-MIRACLE extract (PROTOCOL.md): geometry check, then transfer functions and arrival times. No estimator.

python3 -I -B miracle_prep.py SCENARIO H5 MD5 OUT_DIR
Reads the MIRACLE scenario file read-only, checks its MD5, infers the array layout from the dimensions (a 4,096-long
loudspeaker-position axis, a 64-long microphone axis, the rest time) and refuses anything else. Writes OUT_DIR/
miracle-<SCENARIO>.npz with the arrays of the registered measured extract (pos_mic: the 441 grid positions; pos_src: the
64 array microphones; H: (64, 441, 36) transfer functions in the exp(+ikr) convention, as the conjugated DTFT at 12
log-spaced frequencies strictly inside each octave at 250, 500 and 1000 Hz; arrival_s: (64, 441); freqs; band; c0) and a
receipt JSON with the qualification, which must pass before any transfer function is computed.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, hilbert, resample, sosfiltfilt

FS = 32000.0
BANDS = (250.0, 500.0, 1000.0)
PER_BAND = 12
SUB = np.arange(1, 64, 3)                     # 21 indices per axis: 1, 4, ..., 61
SOURCES = np.sort(np.random.default_rng(414600).choice(64, 32, replace=False))


def freqs_of(centre):
    return np.geomspace(centre / np.sqrt(2), centre * np.sqrt(2), PER_BAND + 2)[1:-1]


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def arrival_times(h, band=(500.0, 8000.0), up=8):
    """h: (..., nt). Earliest crossing of half the peak of the Hilbert envelope of the band-passed response."""
    nt = h.shape[-1]
    seg = sosfiltfilt(butter(4, band, btype='band', fs=FS, output='sos'), np.asarray(h, float), axis=-1)
    env = np.abs(hilbert(resample(seg, nt * up, axis=-1), axis=-1))
    first = np.argmax(env > 0.5 * env.max(axis=-1, keepdims=True), axis=-1)
    return first / up / FS


def as_points(a, n):
    a = np.asarray(a, float)
    if a.shape == (n, 3):
        return a
    if a.shape == (3, n):
        return a.T
    raise SystemExit(f'unexpected position array shape {a.shape} for {n} points')


def main():
    import h5py
    scen, h5p, want_md5, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    got = md5(h5p)
    if got != want_md5:
        raise SystemExit(f'MD5 mismatch: {got} != {want_md5}')
    rec = dict(scenario=scen, file=h5p.name, md5=got, sources=SOURCES.tolist())
    with h5py.File(h5p, 'r') as f:
        names = []
        f.visit(names.append)
        rec['hdf5_items'] = {n: (list(f[n].shape) if isinstance(f[n], h5py.Dataset) else 'group') for n in names}
        src = as_points(f['data/location/source'][()], 4096)            # loudspeaker grid = this study's positions
        rcv = as_points(f['data/location/receiver'][()], 64)            # array microphones = this study's sources
        c0 = float(np.asarray(f['metadata/c0'][()]).ravel()[0])
        ir = f['data/impulse_response']
        shp = ir.shape
        if len(shp) != 3 or 4096 not in shp or 64 not in shp:
            raise SystemExit(f'unexpected impulse-response shape {shp}')
        ax_pos = shp.index(4096); ax_mic = [i for i in range(3) if shp[i] == 64 and i != ax_pos][0]
        ax_t = [i for i in range(3) if i not in (ax_pos, ax_mic)][0]
        rec['layout'] = dict(shape=list(shp), position_axis=ax_pos, microphone_axis=ax_mic, time_axis=ax_t)
        # the 64 x 64 grid order: sort positions by the two in-plane coordinates
        span = src.max(0) - src.min(0)
        normal = int(np.argmin(span)); inplane = [i for i in range(3) if i != normal]
        ux = np.unique(np.round(src[:, inplane[0]], 4)); uy = np.unique(np.round(src[:, inplane[1]], 4))
        ok_grid = len(ux) == 64 and len(uy) == 64
        dx = float(np.median(np.diff(ux))) if ok_grid else float('nan'); dy = float(np.median(np.diff(uy))) if ok_grid else float('nan')
        idx = {}
        for i, p in enumerate(src):
            idx[(int(np.argmin(np.abs(ux - p[inplane[0]]))), int(np.argmin(np.abs(uy - p[inplane[1]]))))] = i
        sel = np.array([idx[(a, b)] for a in SUB for b in SUB])
        pos = src[sel]
        lo, hi = pos.min(0), pos.max(0); o = 0.5 * (lo + hi); a = 0.5 * float(np.linalg.norm(hi - lo))
        da = np.linalg.norm(rcv[SOURCES] - o, axis=1) / a
        qual = dict(grid_64x64=bool(ok_grid), spacing_m=[dx, dy], spacing_ok=bool(ok_grid and abs(dx - 0.0233) < 1e-3 and abs(dy - 0.0233) < 1e-3),
                    planar_std_mm=float(1e3 * np.std(src[:, normal])), planar_ok=bool(1e3 * np.std(src[:, normal]) < 1.0),
                    region_a_m=a, d_over_a=[float(da.min()), float(da.max())], plane_to_array_m=float(abs(np.mean(src[:, normal]) - np.mean(rcv[:, normal]))))
        qual['passed'] = bool(qual['grid_64x64'] and qual['spacing_ok'] and qual['planar_ok'] and (scen != 'A1' or qual['d_over_a'][1] < 1.40))
        rec['qualification'] = qual
        (out / f'miracle-{scen}-receipt.json').write_text(json.dumps(rec, indent=1))
        if not qual['passed']:
            raise SystemExit('qualification failed: ' + json.dumps(qual))
        fr = np.concatenate([freqs_of(c) for c in BANDS]); band = np.concatenate([np.full(PER_BAND, c) for c in BANDS])
        nt = shp[ax_t]
        E = np.exp(-2j * np.pi * fr[None, :] * np.arange(nt)[:, None] / FS)      # (nt, nf)
        H = np.zeros((64, len(sel), len(fr)), complex); T = np.zeros((64, len(sel)))
        for m, i in enumerate(sel):
            h = np.moveaxis(np.asarray(ir[(slice(None),) * ax_pos + (int(i),)], float), [a - (a > ax_pos) for a in (ax_mic, ax_t)], [0, 1])
            assert h.shape == (64, nt)
            H[:, m, :] = np.conj(h @ E)
            T[:, m] = arrival_times(h)
    np.savez(out / f'miracle-{scen}.npz', pos_mic=pos, pos_src=rcv, H=H, arrival_s=T, freqs=fr, band=band, c0=np.array(c0),
             sources=SOURCES, grid_index=sel)
    rec['extract_sha256'] = hashlib.sha256((out / f'miracle-{scen}.npz').read_bytes()).hexdigest()
    d = np.linalg.norm(rcv[:, None, :] - pos[None], axis=2)
    lat = np.median(T - d / c0, axis=1)
    rec.update(c0=c0, n_time=int(nt), latency_s_median_per_source=lat.tolist(),
               arrival_residual_ms_median=float(np.median(np.abs(T - d / c0 - lat[:, None])) * 1e3))
    (out / f'miracle-{scen}-receipt.json').write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k not in ('hdf5_items', 'latency_s_median_per_source')}, indent=1))


if __name__ == '__main__':
    main()
