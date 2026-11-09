"""MeshRIR S32-M441 preprocessing for a registered measured study (geometry and transfer functions only; no estimator).

python3 -I -B meshrir_prep.py ZIP OUT_DIR [--allow-missing]

Input: `S32-M441_npy.zip` of MeshRIR v1.2 (Zenodo 10852693, MD5 9818fc66b36513590e7abd071243d8e9), read in place with
zipfile; arrays are loaded with allow_pickle=False. For each of the 32 loudspeakers and the 441 grid microphones it
writes:
- transfer functions (exp(+i k r) convention: complex conjugate of the real FFT of the full 32,768-sample response) at
  12 bins log-spaced strictly inside each octave band centred at 125, 250, 500 and 1000 Hz (FFT resolution 1.46 Hz);
- the direct-sound arrival time at every microphone: earliest crossing of half the peak of the Hilbert envelope of the
  0.5-8 kHz band-passed response in the 25 ms after sample 5,400 (the measurement latency is about 5,675 samples, read
  from two responses before registration), with 8x Fourier upsampling, as for MP-RIR;
- the sound speed from the dataset's average temperature (331.3 sqrt(1 + T / 273.15) m/s), as a thermometer would give.
The SHA-256 of the extract is printed and stored; a registered receipt must record it. --allow-missing is for the
laptop dry run on a partial copy only (missing microphones are skipped and the extract is marked partial).
"""
from __future__ import annotations
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
from scipy.signal import butter, hilbert, resample, sosfiltfilt

FS, NIR, T0 = 48000, 32768, 5400
BANDS = (125.0, 250.0, 500.0, 1000.0)
PER_BAND = 12
PREFIX = 'S32-M441_npy/'


def bins_of(centre):
    f = np.geomspace(centre / np.sqrt(2), centre * np.sqrt(2), PER_BAND + 2)[1:-1]
    b = np.unique(np.round(f * NIR / FS).astype(int))
    return b


def arrival_times(h, window_ms=25.0, band=(500.0, 8000.0), up=8):
    w = int(window_ms * 1e-3 * FS)
    seg = np.asarray(h[..., T0:T0 + w + 512], float)
    seg = sosfiltfilt(butter(4, band, btype='band', fs=FS, output='sos'), seg, axis=-1)[..., :w]
    env = np.abs(hilbert(resample(seg, w * up, axis=-1), axis=-1))
    first = np.argmax(env > 0.5 * env.max(axis=-1, keepdims=True), axis=-1)
    return (T0 + first / up) / FS


def load_member(z, name):
    return np.load(io.BytesIO(z.read(PREFIX + name)), allow_pickle=False)


def main():
    zpath, out = Path(sys.argv[1]), Path(sys.argv[2]); allow_missing = '--allow-missing' in sys.argv
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        names = set(z.namelist())
        pos_mic, pos_src = load_member(z, 'pos_mic.npy'), load_member(z, 'pos_src.npy')
        meta = json.loads(z.read(PREFIX + 'data.json'))
        assert pos_mic.shape == (441, 3) and pos_src.shape == (32, 3)
        allb = np.concatenate([bins_of(f) for f in BANDS])
        band = np.concatenate([np.full(len(bins_of(f)), f) for f in BANDS])
        H = np.full((32, 441, len(allb)), np.nan + 0j)
        T = np.full((32, 441), np.nan)
        present = []
        for m in range(441):
            nm = f'ir_{m}.npy'
            if PREFIX + nm not in names:
                if allow_missing:
                    continue
                raise FileNotFoundError(nm)
            h = load_member(z, nm)
            assert h.shape == (32, NIR) and h.dtype == np.float64
            H[:, m, :] = np.conj(np.fft.rfft(h, axis=-1)[:, allb])
            T[:, m] = arrival_times(h)
            present.append(m)
    temp = float(meta['average temperature'])
    c = 331.3 * np.sqrt(1 + temp / 273.15)
    np.savez(out / 'meshrir-prep.npz', pos_mic=pos_mic, pos_src=pos_src, H=H, arrival_s=T, bins=allb,
             freqs=allb * FS / NIR, band=band, present=np.array(present))
    sha = hashlib.sha256((out / 'meshrir-prep.npz').read_bytes()).hexdigest()
    d = np.linalg.norm(pos_src[:, None, :] - pos_mic[None, :, :], axis=2)
    lat = np.nanmedian(T - d / c, axis=1)
    rec = dict(source='MeshRIR v1.2 S32-M441_npy.zip', microphones_present=len(present), partial=len(present) < 441,
               average_temperature_c=temp, sound_speed_from_temperature=float(c), bins_per_band={int(f): bins_of(f).tolist() for f in BANDS},
               latency_s_median_per_source=lat.tolist(), arrival_residual_ms_median=float(np.nanmedian(np.abs(T - d / c - lat[:, None])) * 1e3),
               nearest_mic_m_per_source=d.min(axis=1).tolist(), extract_sha256=sha)
    (out / 'meshrir-prep.json').write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k not in ('latency_s_median_per_source', 'nearest_mic_m_per_source')}))


if __name__ == '__main__':
    main()
