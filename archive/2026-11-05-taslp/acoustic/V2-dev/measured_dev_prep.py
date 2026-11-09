"""Version-2 development data from the S4/S7/S8 boxes of TB-M1 (V2-DESIGN.md): transfer functions at the
development bins only (log-midpoints 1, 3, ..., 11 between consecutive TB-M1 bins of each band) and per-microphone
direct-sound arrival times for the microphone-only two-stage baseline. The held-out midpoints 2, 4, ..., 10 are
listed but not computed.

python3 -B measured_dev_prep.py OUT_DIR
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common')); sys.path.insert(0, str(HERE.parent))
import mprir as P  # noqa: E402
import measured_prep as MP  # noqa: E402

BANDS = (125.0, 250.0, 500.0, 1000.0)


def split_bins(centre):
    b1, _ = P.frequency_bins(centre, 12)                      # the TB-M1 bins (spent)
    f1 = b1 * P.FS / P.N
    mids = np.sqrt(f1[:-1] * f1[1:])                          # 11 log-midpoints
    mb = np.round(mids * P.N / P.FS).astype(int)
    assert len(set(mb) & set(b1)) == 0 and len(np.unique(mb)) == len(mb)
    return b1, mb[0::2], mb[1::2]                             # spent, development (6), held out (5)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    tbm1 = json.loads((HERE.parent / 'B3-measured/prep/measured-prep.json').read_text())
    st = P.load_setup(); g = P.geometry()
    t_sys = int(st['T_system']) + int(st['T_guard'])
    split = {int(f): dict(zip(('tbm1', 'development', 'held_out'), (v.tolist() for v in split_bins(f)))) for f in BANDS}
    dev = np.concatenate([np.array(split[int(f)]['development']) for f in BANDS])
    arrays, rec = {}, dict(bins=split)
    for s in ('S4', 'S7', 'S8'):
        cols = np.array(tbm1[s]['columns'])
        h, src = MP.read_columns(s, cols)
        sha = hashlib.sha256(h.tobytes()).hexdigest()
        assert sha == tbm1[s]['box_sha256'], 'box differs from the TB-M1 box'
        hh = h.reshape(-1, P.N)
        arrays[f'{s}_positions'] = g[cols].reshape(-1, 3)
        arrays[f'{s}_H'] = P.transfer(hh, dev).astype(np.complex128)
        arrays[f'{s}_arrival_s'] = P.arrival_times(hh, t_sys)
        rec[s] = dict(box_sha256=sha, impulse_source=src, box_lo=tbm1[s]['box_lo'], box_hi=tbm1[s]['box_hi'],
                      acoustic_centre_all_mics=tbm1[s]['acoustic_centre'], sound_speed_all_mics=tbm1[s]['sound_speed'])
        print(s, src, flush=True)
    arrays['bins'] = dev; arrays['freqs'] = dev * P.FS / P.N
    arrays['band'] = np.concatenate([np.full(len(split[int(f)]['development']), f) for f in BANDS])
    np.savez(out / 'measured-dev.npz', **arrays)
    (out / 'measured-dev.json').write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
