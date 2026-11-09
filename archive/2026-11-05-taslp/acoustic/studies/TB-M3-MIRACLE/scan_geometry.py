"""Scan-grid spacing of the TB-M3-MIRACLE problems, from geometry only (post hoc; no estimate or outcome is read).

python3 -B scan_geometry.py EXTRACT_DIR OUT_JSON

Added 2026-11-07 after an independent review noted that the manuscript's statement of where the scan's node cap binds
covered only the simulation and the room grid. Uses posthoc/geometry/geometry.py's spacing() with the extract's grid
positions, its nominal sound speed and the registered bins of each band (the odd-indexed six of twelve, as run_m3.build).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
sys.path.insert(0, str(S / 'common')); sys.path.insert(0, str(S / 'posthoc' / 'geometry'))
import geometry as GEO  # noqa: E402


def main():
    ext, out = Path(sys.argv[1]), Path(sys.argv[2])
    res = dict(status='geometry only; no estimate or outcome is read')
    for sc in ('A1', 'R2'):
        z = np.load(ext / f'miracle-{sc}.npz', allow_pickle=False)
        pos, c0 = z['pos_mic'], float(z['c0'])
        for fc in (250.0, 500.0, 1000.0):
            inband = np.where(z['band'] == fc)[0][1::2]
            res[f'{sc}-{int(fc)}'] = GEO.spacing(pos[:24], pos[24:], z['freqs'][inband], c0, (pos.min(0), pos.max(0)))
    out.write_text(json.dumps(res, indent=1) + '\n')
    for k, v in res.items():
        print(k, v)


if __name__ == '__main__':
    main()
