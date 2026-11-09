"""Source proximity and scan-grid spacing of the registered studies, from geometry only. No estimate or outcome is read.

Proximity of a source s to the region, with o the centre of the region's bounding box and a the radius of the
smallest ball about o that holds every region point:
  d/a        the distance of s from o over a; below one, s lies inside that ball, where the expansion of its direct
             field about o does not converge at every region point;
  clearance  the distance of s from the nearest region point (microphone or target node), in metres and in
             wavelengths at the band centre.
Simulation: the registered field builder `sim.build` is imported unchanged. Its two field functions are replaced in
memory by a constant, so only the registered geometry draws are computed; three problems are checked against the
unpatched builder. Measured: the loudspeaker and grid positions of the registered extract.
Scan grid: the registered rule, `estimators.grid_nodes` on `v2.Band.search_box`, spacing a quarter of the band-centre
wavelength coarsened by factors of 1.1 until at most `v2.SCAN_NODES` nodes remain.

    python3 -B posthoc/geometry/geometry.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
sys.path.insert(0, str(S / 'common'))
import estimators as E  # noqa: E402
import fields as F  # noqa: E402
import problems_v2 as PB  # noqa: E402
import sim  # noqa: E402
import v2  # noqa: E402

SEEDS = json.loads((S / 'SEEDS.json').read_text())['declared']['TB-S2_simulation_fields']
CENTRES = (125.0, 250.0, 500.0, 1000.0)


def geometry_only(cell, seed):
    real = F.image_pressure, F.direct_pressure
    F.image_pressure = lambda pts, *a, **k: np.ones(len(pts), complex)
    F.direct_pressure = lambda pts, *a, **k: np.zeros(len(pts), complex)
    try:
        return sim.build(dict(cell), seed)
    finally:
        F.image_pressure, F.direct_pressure = real


def proximity(src, pts, centre, a, lam):
    d = float(np.linalg.norm(src - centre))
    clear = float(np.min(np.linalg.norm(pts - src, axis=1)))
    return dict(d_over_a=d / a, clearance_m=clear, clearance_wavelengths=clear / lam)


def spacing(x, q, freqs, c, box):
    band = v2.Band(x, np.ones((len(x), len(freqs)), complex), q, np.asarray(freqs), c=c, box=box)
    lo, hi = band.search_box()
    nodes = E.grid_nodes(lo, hi, 0.25 * band.lam_mid(), max_nodes=v2.SCAN_NODES)
    steps = [float(np.diff(np.unique(nodes[:, i])).max()) for i in range(3)]
    return dict(nodes=int(len(nodes)), step_m=max(steps), step_wavelengths=max(steps) / band.lam_mid(),
                quarter_wavelength_m=0.25 * band.lam_mid(), cap_binds=bool(max(steps) > 0.25 * band.lam_mid() + 1e-12))


def span(v):
    v = np.asarray(v, float)
    return dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max()))


def main():
    cells = sim.cells()
    check = []
    for name, seed in (('base', SEEDS[0]), ('A1-beside0.3', SEEDS[0] + 1), ('FF-07', SEEDS[0] + 2)):
        g, r = geometry_only(cells[name], seed), sim.build(dict(cells[name]), seed)
        check.append(dict(cell=name, seed=seed, identical=bool(all(np.array_equal(g[k], r[k]) for k in
                                                                  ('x', 'q', 'sources', 'primary', 'origin')))))
    assert all(c['identical'] for c in check), check
    out = dict(status='geometry only; no estimate or outcome is read', check_against_unpatched_builder=check,
               simulation={}, measured={}, scan_grid={})
    for name, cell in cells.items():
        lam = F.C0 / float(cell['f'])
        per = []
        for seed in range(SEEDS[0], SEEDS[1] + 1):
            g = geometry_only(cell, seed)
            pts = np.vstack((g['x'], g['q']))
            nodes = F.region_grid(g['origin'], sim.SHAPE)
            centre = g['origin'] + sim.EXTENT / 2
            a = float(np.max(np.linalg.norm(nodes - centre, axis=1)))
            per.append(dict(a=a, inside=bool(np.all(g['primary'] >= g['box'][0]) and np.all(g['primary'] <= g['box'][1])),
                            **proximity(g['primary'], pts, centre, a, lam)))
        out['simulation'][name] = dict(f=float(cell['f']), place=cell['place'], a_m=per[0]['a'],
                                       inside=bool(any(p['inside'] for p in per)),
                                       **{k: span([p[k] for p in per]) for k in ('d_over_a', 'clearance_m', 'clearance_wavelengths')})
    prep = np.load(S / 'B3-measured/meshrir-prep/meshrir-prep.npz')
    pos, src = prep['pos_mic'], prep['pos_src']
    centre = (pos.min(0) + pos.max(0)) / 2
    a = float(np.max(np.linalg.norm(pos - centre, axis=1)))
    for fc in CENTRES:
        lam = PB.C_TEMP_MESHRIR / fc
        per = [proximity(s, pos, centre, a, lam) for s in src]
        out['measured'][f'{int(fc)}'] = dict(a_m=a, **{k: span([p[k] for p in per]) for k in ('d_over_a', 'clearance_m', 'clearance_wavelengths')})
    g = geometry_only(cells['base'], SEEDS[0])
    for fc in CENTRES:
        out['scan_grid'][f'simulation-{int(fc)}'] = spacing(g['x'], g['q'], PB.band_freqs(fc), F.C0, g['box'])
        out['scan_grid'][f'measured-{int(fc)}'] = spacing(pos[:24], pos[24:], PB.band_freqs(fc), PB.C_TEMP_MESHRIR,
                                                          (pos.min(0), pos.max(0)))
    (HERE / 'geometry.json').write_text(json.dumps(out, indent=1) + '\n')
    for k in ('base', 'A1-in', 'A1-below0.1', 'A1-below0.6', 'A1-beside0.3', 'A1-beside0.6', 'A1-beside1.2'):
        v = out['simulation'][k]
        print(k, v['f'], 'd/a', v['d_over_a'], 'clear', v['clearance_m'], 'wl', v['clearance_wavelengths'])
    for k, v in out['measured'].items():
        print('measured', k, 'd/a', v['d_over_a'], 'clear', v['clearance_m'], 'wl', v['clearance_wavelengths'])
    for k, v in out['scan_grid'].items():
        print(k, v)


if __name__ == '__main__':
    main()
