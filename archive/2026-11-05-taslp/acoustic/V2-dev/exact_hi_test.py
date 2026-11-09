"""Development test of run_exact_hi.one (before the TB-M2-EXACT-HI registration): on development MeshRIR problems
with 96 and 192 microphones (problem builder patched to the development band), the row is built without error and
its gsmf cubature twin reproduces the V2-dev/dev1-box row bit for bit; the exact-kernel time is recorded.
python3 -B V2-dev/exact_hi_test.py"""
import json, sys, warnings
from multiprocessing import Pool
A = str(__import__('pathlib').Path(__file__).resolve().parents[1])
sys.path.insert(0, A)
import run_exact_hi as H
sys.path.insert(0, A + '/V2-dev')
import run_dev as RD


def job(key):
    warnings.simplefilter('ignore'); RD.init()
    spk, fc, M, seed = key
    H.PB.meshrir_band = lambda PREP, s_, f_, m_, d_: RD.meas_band(spk, fc, M, seed)
    return key, H.one(('mesh', spk, fc, M, 0, seed))


if __name__ == '__main__':
    dev = {(r['source'], r['band'], r['M_mics'], r['seed']): r for r in map(json.loads, open(A + '/V2-dev/dev1-box/rows.jsonl'))
           if r.get('kind') == 'meas' and 'error' not in r}
    keys = [next(k for k in sorted(dev) if k[2] == M and k[1] == 125.0) for M in (96, 192)]
    with Pool(2) as p:
        for key, row in p.imap_unordered(job, keys):
            ok = 'error' not in row
            print(key, 'row built:', ok, row.get('error', '')[-400:])
            if ok:
                json.dumps(row, default=float)
                print('  nmse keys', sorted(row['nmse']), 'seconds', row['seconds'])
                print('  cubature twin vs dev1-box gsmf:', row['nmse']['gsmf_cubature'] == dev[key]['nmse']['gsmf'], flush=True)
