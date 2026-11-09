"""Held-out MP-RIR bins for a registered version-2 measured analysis (geometry and transfer functions only).

python3 -B heldout_prep.py OUT_DIR
The S4/S7/S8 boxes of TB-M1 at the held-out bins fixed in V2-DESIGN.md before any version-2 output: log-midpoints
2, 4, 6, 8 and 10 between consecutive TB-M1 bins of each band (5 per band). These recordings were seen in TB-M1 and in
version-2 development, so results on them are labelled as the same recordings, not independent evidence. Box bytes:
the verified full file if present, otherwise the downloaded byte ranges, whose boxes were shown byte-identical to the
MD5-verified Zenodo files on 2026-11-05 (B3-measured/integrity.json). Per-microphone arrival times are stored for the
microphone-only two-stage baseline and the sound speed from training microphones.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common')); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / 'V2-dev'))
import mprir as P  # noqa: E402
import measured_prep as MP  # noqa: E402
from measured_dev_prep import split_bins, BANDS  # noqa: E402


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    tbm1 = json.loads((HERE / 'prep/measured-prep.json').read_text())
    integ = json.loads((HERE / 'integrity.json').read_text())
    st = P.load_setup(); g = P.geometry()
    t_sys = int(st['T_system']) + int(st['T_guard'])
    split = {int(f): dict(zip(('tbm1', 'development', 'held_out'), (v.tolist() for v in split_bins(f)))) for f in BANDS}
    held = np.concatenate([np.array(split[int(f)]['held_out']) for f in BANDS])
    arrays, rec = {}, dict(bins=split)
    for s in ('S4', 'S7', 'S8'):
        cols = np.array(tbm1[s]['columns'])
        h, src = MP.read_columns(s, cols)
        sha = hashlib.sha256(h.tobytes()).hexdigest()
        assert sha == tbm1[s]['box_sha256'] == integ[s]['registered_box_sha256'] and integ[s]['status'] == 'verified'
        hh = h.reshape(-1, P.N)
        arrays[f'{s}_positions'] = g[cols].reshape(-1, 3)
        arrays[f'{s}_H'] = P.transfer(hh, held).astype(np.complex128)
        arrays[f'{s}_arrival_s'] = P.arrival_times(hh, t_sys)
        rec[s] = dict(box_sha256=sha, impulse_source=src, box_lo=tbm1[s]['box_lo'], box_hi=tbm1[s]['box_hi'],
                      acoustic_centre_all_mics=tbm1[s]['acoustic_centre'], sound_speed_all_mics=tbm1[s]['sound_speed'])
    arrays['bins'] = held; arrays['freqs'] = held * P.FS / P.N
    arrays['band'] = np.concatenate([np.full(len(split[int(f)]['held_out']), f) for f in BANDS])
    np.savez(out / 'heldout.npz', **arrays)
    rec['npz_sha256'] = hashlib.sha256((out / 'heldout.npz').read_bytes()).hexdigest()
    (out / 'heldout.json').write_text(json.dumps(rec, indent=1))
    print('held-out bins', held.tolist(), rec['npz_sha256'])


if __name__ == '__main__':
    main()
