"""Descriptive development comparison (not a kill rule): every candidate assembly of version 2 against every competitor,
by regime. Cell value = median over tasks of the paired difference in dB (negative favours the candidate); a regime
value = median over its cells; also the number of cells where the candidate is better / worse by more than 0.1 dB.

python3 -B describe_dev.py DEV_DIR
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_dev import load, cell_of, db, GENERIC

CANDIDATES = ('v1', 'ind_sw', 'j_K1', 'j_sw', 'j_la', 'j_cv', 'j_sw_ext', 'j_la_ext', 'j_cv_ext', 'fj_cv', 'fj_cv_ext')
COMPETITORS = ('mo26_cap5', 'hybrid_v1', 'gsmf', 'best_generic', 'v1')


def main():
    d = Path(sys.argv[1]); rows, _ = load(d)
    E = defaultdict(list)
    for r in rows:
        e = {m: db(v) for m, v in r['nmse'].items()}
        e['best_generic'] = min(e[m] for m in GENERIC)
        E[cell_of(r)].append(e)
    cells = sorted(E); sim = [c for c in cells if not c.startswith('meas')]; meas = [c for c in cells if c.startswith('meas')]
    reg = {'simulated': sim, 'measured': meas,
           'measured 125-250 Hz': [c for c in meas if '-125-' in c or '-250-' in c],
           'measured 500-1000 Hz': [c for c in meas if '-500-' in c or '-1000-' in c],
           'sparse (24-48 mics)': [c for c in cells if any(t in c for t in ('M24', 'M48'))],
           'rich (192 mics or 40 dB)': [c for c in cells if 'M192' in c or c == 'A9-snr40'],
           'inside or beside (sim)': [c for c in sim if c.startswith('A1')]}
    med = lambda c, a, b: float(np.median([t[a] - t[b] for t in E[c]]))
    out = {}
    for a in CANDIDATES:
        out[a] = {}
        for g, cs in reg.items():
            out[a][g] = {b: dict(median_db=round(float(np.median([med(c, a, b) for c in cs])), 2),
                                 better=sum(med(c, a, b) < -0.1 for c in cs), worse=sum(med(c, a, b) > 0.1 for c in cs), cells=len(cs))
                         for b in COMPETITORS if b != a}
            out[a][g]['abs_median_db'] = round(float(np.median([np.median([t[a] for t in E[c]]) for c in cs])), 2)
    ext = {s: {c: round(med(c, f'j_{s}_ext', f'j_{s}'), 2) for c in cells} for s in ('sw', 'la', 'cv')}
    (d / 'describe.json').write_text(json.dumps(dict(by_candidate=out, extension_vs_points=ext), indent=1))
    for a in CANDIDATES:
        print(f'\n== {a}')
        for g in reg:
            v = out[a][g]
            print(f'  {g:26s} abs {v["abs_median_db"]:7.2f} | ' + ' | '.join(f'{b}: {v[b]["median_db"]:+.2f} ({v[b]["better"]}/{v[b]["worse"]} of {v[b]["cells"]})' for b in COMPETITORS if b != a))
    for s, v in ext.items():
        print(f'extension at {s}: worst {max(v.values()):+.2f} best {min(v.values()):+.2f} dB; cells worse >0.1: {[c for c, x in v.items() if x > 0.1]}')


if __name__ == '__main__':
    main()
