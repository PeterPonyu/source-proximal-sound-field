"""TB-M1 preprocessing (geometry and transfer functions; computes no reconstruction outcome).

For each loudspeaker S4, S7, S8: the 9 x 9 x 8 box of grid points next to it, direct-path arrival times at every box
microphone, the acoustic centre and sound speed fitted from them, octave-band T30 of the room, and the transfer
functions (exp(+i k r) convention) at the registered frequency bins. Impulse responses are read from the verified
full file when its MD5 matches Zenodo, otherwise from the downloaded byte ranges (chunk files); the SHA-256 of every
extracted box is recorded so a later extraction from the verified file can be compared byte for byte.

python3 -B measured_prep.py OUT_DIR
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'common'))
import mprir as P  # noqa: E402

SOURCES = {'S4': 'centred', 'S7': 'adjacent', 'S8': 'adjacent'}
MD5 = dict(S4='a705e3f028cfab4303511c52f75749ef', S7='f33f021d2550711d1371b6d65c5abbfb', S8='25b297c591162d71059f53b6e955057c')
BANDS = (125.0, 250.0, 500.0, 1000.0)
PER_BAND = 12
COL = 8 * P.N * 4
CH = 16 * 1024 * 1024
HDR = 128


def full_ok(s):
    f = P.DATA / f'{s}_Mrir.npy'
    if not f.exists() or f.stat().st_size != 3440099456:
        return False
    marker = P.DATA / f'{s}_Mrir.md5ok'
    return marker.exists() and marker.read_text().strip() == MD5[s]


def read_columns(s, cols):
    """(len(cols), 8, N) float32 impulse responses of grid columns, from the verified file or from chunk files."""
    if full_ok(s):
        mm = P.rirs(s)
        return np.ascontiguousarray(mm[np.asarray(cols)]), 'verified full file'
    d = P.DATA / f'.{s}_chunks'
    out = np.empty((len(cols), 8, P.N), np.float32)
    for n, i in enumerate(cols):
        a = HDR + int(i) * COL; b = a + COL
        buf = bytearray()
        for c in range(a // CH, (b - 1) // CH + 1):
            data = (d / f'{c:05d}').read_bytes()
            lo = max(a, c * CH) - c * CH; hi = min(b, (c + 1) * CH) - c * CH
            buf += data[lo:hi]
        out[n] = np.frombuffer(bytes(buf), '<f4').reshape(8, P.N)
    return out, 'downloaded byte ranges (full-file MD5 pending)'


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    st = P.load_setup(); g = P.geometry(); xy = g[:, 0, :2]
    t_sys = int(st['T_system']) + int(st['T_guard'])   # impulse responses start after the system delay plus the guard interval
    rec, arrays = {}, {}
    bins = {f: P.frequency_bins(f, PER_BAND) for f in BANDS}
    allbins = np.concatenate([b for b, _ in bins.values()]); allfreq = np.concatenate([fr for _, fr in bins.values()])
    for s, mode in SOURCES.items():
        base = np.asarray(st['coord_speaker_center'][s], float)
        cols = P.box_columns(xy, base[:2], mode=mode)
        h, src = read_columns(s, cols)
        pos = g[cols].reshape(-1, 3)
        hh = h.reshape(-1, P.N)
        t = P.arrival_times(hh, t_sys)
        fit = P.fit_centre(pos, t, base + np.array([0, 0, 0.3]))
        t30 = {int(f): float(np.median([P.schroeder_t30(hh[i], f, t_sys) for i in range(0, len(hh), 27)])) for f in BANDS}
        H = P.transfer(hh, allbins)                                   # (648, nbins)
        # phase-convention check on the 1 kHz band: the transfer function rotated by exp(-i k r) should be more
        # coherent across microphones than when rotated by exp(+i k r) (direct field dominant near the source)
        r = np.linalg.norm(pos - fit['centre'], axis=1)
        sel = allfreq > 700
        k = 2 * np.pi * allfreq[sel] / fit['c']
        coh_plus = float(np.mean(np.abs(np.mean(np.exp(1j * np.angle(H[:, sel] * np.exp(-1j * k[None, :] * r[:, None]))), axis=0))))
        coh_minus = float(np.mean(np.abs(np.mean(np.exp(1j * np.angle(H[:, sel] * np.exp(+1j * k[None, :] * r[:, None]))), axis=0))))
        dmin = float(np.min(np.linalg.norm(pos - fit['centre'], axis=1)))
        lo, hi = pos.min(0), pos.max(0)
        rec[s] = dict(mode=mode, base_centre=base.tolist(), columns=[int(c) for c in cols], box_lo=lo.tolist(), box_hi=hi.tolist(),
                      acoustic_centre=fit['centre'].tolist(), sound_speed=fit['c'], arrival_residual_rms_mm=fit['residual_rms_mm'],
                      onset_offset_ms=1e3 * fit['offset_s'], centre_sd_m=fit['centre_sd_m'], sound_speed_sd=fit['c_sd'],
                      centre_minus_base=(fit['centre'] - base).tolist(), nearest_mic_to_centre_m=dmin,
                      centre_height_below_box_m=float(lo[2] - fit['centre'][2]), t30_s=t30,
                      phase_coherence_plus_convention=coh_plus, phase_coherence_minus_convention=coh_minus,
                      impulse_source=src, box_sha256=hashlib.sha256(h.tobytes()).hexdigest())
        arrays[f'{s}_positions'] = pos
        arrays[f'{s}_H'] = H.astype(np.complex128)
        print(s, json.dumps({k: v for k, v in rec[s].items() if k not in ('columns',)}, default=float), flush=True)
    arrays['bins'] = allbins; arrays['freqs'] = allfreq
    np.savez(out / 'measured-prep.npz', **arrays)
    rec['bins_per_band'] = {int(f): [float(v) for v in fr] for f, (_, fr) in bins.items()}
    rec['npz_sha256'] = hashlib.sha256((out / 'measured-prep.npz').read_bytes()).hexdigest()
    (out / 'measured-prep.json').write_text(json.dumps(rec, indent=1, default=float))


if __name__ == '__main__':
    main()
