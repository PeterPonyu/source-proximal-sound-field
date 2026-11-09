"""Timing probe for the exact-kernel rerun (development MeshRIR problems only; timings, never reported as results).

Process seconds of the registered comparators with the exact uniform-ball series kernel (V2-dev/cubature_diag.py,
`exact_kernel`) and with the registered 33-node cubature: mo26_cap5 (v2.mo26_cap per bin) and gsmf (v2.gsmf_fit) at
M = 24 and 48, and gsmf alone at M = 96 and 192, on development loudspeakers and seeds.

OPENBLAS_NUM_THREADS=1 nice -n 10 python3 -B exact_sizing.py OUT_DIR
"""
import contextlib, json, sys, time, warnings
from multiprocessing import Pool
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'common')); sys.path.insert(0, str(HERE))
import v2  # noqa: E402
import run_dev as RD  # noqa: E402
import prox_dev as PD  # noqa: E402
import cubature_diag as CD  # noqa: E402

LANES = [[('S7', 500.0, 24, ('mo26_cap5', 'gsmf')), ('S7', 500.0, 48, ('mo26_cap5', 'gsmf')), ('S7', 500.0, 96, ('gsmf',)),
          ('S4', 1000.0, 96, ('gsmf',))],
         [('S7', 500.0, 192, ('gsmf',)), ('S4', 1000.0, 192, ('gsmf',))]]


def seed_of(spk, M):
    for t in PD.tasks('full'):
        if t[0] == 'meas' and t[1] == spk and t[3] == M and t[4] == 0:
            return t[5]


def lane(items):
    warnings.simplefilter('ignore')
    RD.init()
    out = []
    for spk, fc, M, methods in items:
        band, *_ = RD.meas_band(spk, fc, M, seed_of(spk, M))
        row = dict(source=spk, band=fc, M=M, seconds={})
        for tag, ctx in (('cubature', contextlib.nullcontext), ('series', CD.exact_kernel)):
            with ctx():
                for m in methods:
                    a = time.process_time()
                    if m == 'mo26_cap5':
                        [v2.mo26_cap(p) for p in band.probs]
                    else:
                        v2.gsmf_fit(band)
                    row['seconds'][f'{m}_{tag}'] = round(time.process_time() - a, 1)
        print(json.dumps(row), flush=True)
        out.append(row)
    return out


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    with Pool(len(LANES)) as pool:
        res = pool.map(lane, LANES)
    (out / 'rows.json').write_text(json.dumps([r for l in res for r in l], indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
