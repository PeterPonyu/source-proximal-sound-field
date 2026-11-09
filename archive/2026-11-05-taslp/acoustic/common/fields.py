"""Forward models for the TASLP study: steady-state single-frequency pressure in a shoebox room.

Image-source sum with locally reacting walls (angle-dependent plane-wave reflection coefficient per image path),
standard air absorption, monopole / monopole-plus-dipole directivity, coherent multi-driver and independent
multi-source excitation, and the region grid that mirrors the measured grid (0.15 m spacing).
Free-field Green convention: exp(i k r) / r (no 4*pi; it cancels on normalisation).
"""
from __future__ import annotations
import numpy as np

C0 = 343.0
# ISO 9613-1 pure-tone air attenuation at 20 degC, 50 % relative humidity, 101.325 kPa (dB/km).
AIR_DB_PER_KM = {125.0: 0.44, 250.0: 1.31, 500.0: 2.73, 1000.0: 4.66}
SPACING = 0.15


def air_np_per_m(frequency):
    """Interpolated (log-log) ISO 9613-1 attenuation in nepers per metre."""
    f = np.array(sorted(AIR_DB_PER_KM)); a = np.array([AIR_DB_PER_KM[v] for v in f])
    db_km = float(np.exp(np.interp(np.log(frequency), np.log(f), np.log(a))))
    return db_km / 1000.0 / (20.0 / np.log(10.0))


def zeta_from_r0(r0):
    """Normalised specific impedance of a locally reacting wall with normal-incidence reflection coefficient r0."""
    r0 = complex(r0)
    return (1 + r0) / (1 - r0)


def reflection(zeta, cos_theta):
    return (zeta * cos_theta - 1) / (zeta * cos_theta + 1)


def wall_counts(x0, length, nmax):
    """Image positions with the number of reflections on the low (0) and high (length) wall of one axis.

    Image 2mL + x0 (even): |m| reflections on each wall. Image 2mL - x0 (odd): for m >= 1 it has m on the high wall
    and m-1 on the low wall; for m <= 0 it has |m|+1 on the low wall and |m| on the high wall.
    """
    m = np.arange(-(nmax // 2 + 1), nmax // 2 + 2)
    pe, le, he = 2 * m * length + x0, np.abs(m), np.abs(m)
    po = 2 * m * length - x0
    lo = np.where(m >= 1, m - 1, np.abs(m) + 1)
    ho = np.where(m >= 1, m, np.abs(m))
    pos = np.concatenate((pe, po)); low = np.concatenate((le, lo)); high = np.concatenate((he, ho))
    flip = np.concatenate((np.ones_like(m), -np.ones_like(m)))
    keep = (low + high) <= nmax
    return pos[keep], low[keep], high[keep], flip[keep]


def image_pressure(points, source, frequency, room, zeta, nmax, beta=0.0, axis=None, c=C0, air=True, chunk=4096, rconst=None):
    """Pressure of one driver at `points` (P, 3).

    zeta: dict with keys 'x0','x1','y0','y1','z0','z1' (normalised impedance of each wall) or one complex value.
    beta: dipole fraction; driver = (1 - beta) monopole + beta first-order dipole with far field cos(theta) on `axis`.
    rconst: if given, an angle-independent reflection coefficient on every wall (used only by the known-answer tests).
    """
    k = 2 * np.pi * frequency / c
    alpha = air_np_per_m(frequency) if air else 0.0
    if rconst is None and not isinstance(zeta, dict):
        zeta = {w: complex(zeta) for w in ('x0', 'x1', 'y0', 'y1', 'z0', 'z1')}
    pts = np.asarray(points, float)
    ax = [wall_counts(source[a], room[a], nmax) for a in range(3)]
    px, lx, hx, fx = ax[0]; py, ly, hy, fy = ax[1]; pz, lz, hz, fz = ax[2]
    cx, cy, cz = lx + hx, ly + hy, lz + hz
    out = np.zeros(len(pts), complex)
    for ix in range(len(px)):
        for iy in range(len(py)):
            left = nmax - cx[ix] - cy[iy]
            if left < 0:
                continue
            sel = np.where(cz <= left)[0]
            nodes = np.column_stack((np.full(len(sel), px[ix]), np.full(len(sel), py[iy]), pz[sel]))
            for lo in range(0, len(pts), chunk):
                p = pts[lo:lo + chunk]
                diff = p[:, None, :] - nodes[None, :, :]
                d = np.linalg.norm(diff, axis=2)
                if np.any(d <= 1e-9):
                    raise ValueError('Evaluation point coincides with an image source')
                cosx, cosy, cosz = np.abs(diff[..., 0]) / d, np.abs(diff[..., 1]) / d, np.abs(diff[..., 2]) / d
                if rconst is not None:
                    w = complex(rconst) ** (cx[ix] + cy[iy] + cz[sel])[None, :] * np.ones((len(p), 1))
                else:
                    w = (reflection(zeta['x0'], cosx) ** lx[ix] * reflection(zeta['x1'], cosx) ** hx[ix]
                         * reflection(zeta['y0'], cosy) ** ly[iy] * reflection(zeta['y1'], cosy) ** hy[iy]
                         * reflection(zeta['z0'], cosz) ** lz[sel][None, :] * reflection(zeta['z1'], cosz) ** hz[sel][None, :])
                g = np.exp((1j * k - alpha) * d) / d
                term = (1 - beta) * g
                if beta:
                    assert axis is not None
                    n_img = np.column_stack((np.full(len(sel), axis[0] * fx[ix]), np.full(len(sel), axis[1] * fy[iy]), axis[2] * fz[sel]))
                    cos = np.einsum('pnj,nj->pn', diff, n_img) / d
                    term = term + beta * (1 - 1 / (1j * k * d)) * g * cos
                out[lo:lo + chunk] += (w * term).sum(axis=1)
    return out


def direct_pressure(points, source, frequency, beta=0.0, axis=None, c=C0, air=True):
    """Free-field direct term of one driver (the zero-reflection image)."""
    k = 2 * np.pi * frequency / c
    alpha = air_np_per_m(frequency) if air else 0.0
    diff = np.asarray(points, float) - np.asarray(source, float)[None, :]
    d = np.linalg.norm(diff, axis=1)
    g = np.exp((1j * k - alpha) * d) / d
    out = (1 - beta) * g
    if beta:
        out = out + beta * (1 - 1 / (1j * k * d)) * g * (diff @ np.asarray(axis, float)) / d
    return out


def region_grid(origin, shape=(9, 9, 8), spacing=SPACING):
    """Grid nodes of a box region (the measured grid geometry): origin is the lowest corner."""
    axes = [origin[a] + spacing * np.arange(shape[a]) for a in range(3)]
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)


def eyring_t60(room, r0):
    """Eyring reverberation time with random-incidence absorption approximated by 1 - |r0|^2 on every wall."""
    lx, ly, lz = room
    v = lx * ly * lz; s = 2 * (lx * ly + lx * lz + ly * lz)
    a = 1 - abs(r0) ** 2
    return float(0.161 * v / (-s * np.log(1 - a)))


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack((np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)))
