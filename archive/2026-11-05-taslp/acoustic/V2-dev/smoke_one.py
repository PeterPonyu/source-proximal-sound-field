"""Smoke test of the registered runners' one() on development problems (problem builder patched to a development band;
optimiser iterations cut for the prox runners). Checks only that a row is produced without error and serialises."""
import json, sys, time, warnings
A = str(__import__('pathlib').Path(__file__).resolve().parents[1])
sys.path.insert(0, A); sys.path.insert(0, A + '/common'); sys.path.insert(1, A + '/V2-dev')
warnings.simplefilter('ignore')
import run_dev as RD
RD.init()
which = sys.argv[1]
dev = lambda PREP, spk, fc, M, seed: RD.meas_band('S7', 125.0, 24, 400946)
if which == 'exact':
    import run_exact as R
    R.PB.meshrir_band = dev
    row = R.one(('mesh', 'S7', 125.0, 24, 0, 400946))
else:
    import prox2 as PX
    PX.MO26 = dict(PX.MO26, iters=5, refine_max=2); PX.GSMF = dict(PX.GSMF, iters=5)
    if which == 's2':
        import run_prox_s2 as R
        job = ('mesh', 'S7', 125.0, 24, 0, 400946, 'mo26')
    else:
        import run_prox_a1 as R
        job = ('mesh', 'S7', 125.0, 24, 0, 400946, 'gsmf')
    R.PB.meshrir_band = dev
    row = R.one(job)
s = json.dumps(row, default=float)
print(which, 'error' in row, row.get('error', '')[-600:], 'keys', sorted(row), 'nmse keys', sorted(row.get('nmse', {})), 'len', len(s))
