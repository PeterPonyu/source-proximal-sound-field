"""Band problems for the registered version-2 studies (TB-S2 simulation, TB-M2 measured).

Simulation: `sim.build` at each of 6 bins log-spaced strictly inside the octave of the cell, with the band's dipole
fraction, so geometry, microphones and directivity axis are identical across the bins and every bin has its own noise.
Measured MeshRIR S32-M441: the odd-indexed 6 of the 12 preprocessed bins of each band; microphones drawn from the 441
grid points, the rest are targets. Measured S4/S7/S8 held-out bins (5 per band), labelled as the same recordings.
Measured sound speed: from the training microphones' arrival times (microphone-only two-stage fit), with the
temperature value as fallback if the fit fails or leaves [320, 370] m/s.
"""
from __future__ import annotations
import numpy as np
import estimators as E  # noqa: F401
import mprir as P
import sim
import v2

C_TEMP_MESHRIR = 341.51277000396715


def band_freqs(fc, n=6):
    return fc * 2.0 ** ((np.arange(n) + 0.5) / n - 0.5)


def sim_band(cell_name, seed):
    cell = dict(sim.cells()[cell_name])
    fc = float(cell['f'])
    beta = sim.BETA_REALISTIC[fc] if cell['beta'] == 'realistic' else float(cell['beta'])
    Y, T, fd0 = [], [], None
    for f in band_freqs(fc):
        fd = sim.build(dict(cell, f=float(f), beta=beta), seed)
        if fd0 is None:
            fd0 = fd
        assert np.allclose(fd['x'], fd0['x']) and np.allclose(fd['q'], fd0['q'])
        Y.append(fd['y']); T.append(fd['truth'])
    band = v2.Band(fd0['x'], np.stack(Y, 1), fd0['q'], band_freqs(fc), c=fd0['c_assumed'], box=fd0['box'])
    return band, np.stack(T, 1), fd0['sources'], fd0['primary']


def two_stage_centre(x, t, box):
    """Robust fit of centre, sound speed and onset lead to the training microphones' arrival times (multi-start)."""
    lo, hi = box
    c = (lo + hi) / 2
    starts = [c] + [c + d for d in np.vstack((np.eye(3), -np.eye(3)))]
    best = None
    for s0 in starts:
        try:
            r = P.fit_centre(x, t, s0)
        except Exception:
            continue
        if 300 < r['c'] < 380 and (best is None or r['residual_rms_s'] < best['residual_rms_s']):
            best = r
    return best


def _measured_band(pos, H, arrival, sel_bins, freqs, M, rng, box, c_fallback):
    mic = np.sort(rng.choice(len(pos), M, replace=False)); tgt = np.setdiff1d(np.arange(len(pos)), mic)
    ts = two_stage_centre(pos[mic], arrival[mic], box)
    c = ts['c'] if ts and 320 <= ts['c'] <= 370 else c_fallback
    Y, T = H[mic][:, sel_bins], H[tgt][:, sel_bins]
    sc = np.sqrt(np.mean(np.abs(Y) ** 2, axis=0))
    band = v2.Band(pos[mic], Y / sc, pos[tgt], freqs[sel_bins], c=c, box=box)
    return band, T / sc, ts


def meshrir_band(prep, spk, fc, M, seed):
    """prep: arrays of B3-measured/meshrir-prep/meshrir-prep.npz."""
    inband = np.where(prep['band'] == fc)[0]
    sel = inband[1::2]                                           # odd-indexed 6 of the 12 bins
    rng = np.random.default_rng(np.random.SeedSequence([seed, 9311]))
    pos = prep['pos_mic']
    box = (pos.min(0), pos.max(0))
    band, T, ts = _measured_band(pos, prep['H'][spk], prep['arrival_s'][spk], sel, prep['freqs'], M, rng, box, C_TEMP_MESHRIR)
    return band, T, prep['pos_src'][spk][None, :], ts


def heldout_band(prep, meta, spk, fc, M, seed):
    """prep: arrays of B3-measured/heldout-prep/heldout.npz; meta: its JSON."""
    sel = np.where(prep['band'] == fc)[0]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 9312]))
    m = meta[spk]
    box = (np.asarray(m['box_lo']), np.asarray(m['box_hi']))
    band, T, ts = _measured_band(prep[f'{spk}_positions'], prep[f'{spk}_H'], prep[f'{spk}_arrival_s'], sel, prep['freqs'], M, rng,
                                 box, 343.0)
    return band, T, np.asarray(m['acoustic_centre_all_mics'])[None, :], ts
