"""Development test of run_exact.fit (before the TB-M2-EXACT registration): with the cubature it must reproduce the
V2-dev/dev1-box rows, with the exact kernel the cubature diagnostic's series rows, bit for bit (development MeshRIR problems only).
python3 -B V2-dev/exact_fit_test.py"""
import json, sys, warnings
from multiprocessing import Pool
A = str(__import__('pathlib').Path(__file__).resolve().parents[1])
sys.path.insert(0, A)
import run_exact as RX
sys.path.insert(0, A + '/V2-dev')
import run_dev as RD

def job(key):
    warnings.simplefilter('ignore'); RD.init()
    spk, fc, M, seed = key
    band, T, _, _ = RD.meas_band(spk, fc, M, seed)
    return key, RX.fit(band, T)

if __name__ == '__main__':
    diag = {(r['source'], r['band'], r['M'], r['seed']): r for r in map(json.loads, open(A + '/V2-dev/cubature-diag/rows.jsonl')) if r['kind'] == 'meas'}
    dev = {(r['source'], r['band'], r['M'], r['seed']): r for r in map(json.loads, open(A + '/V2-dev/dev1-box/rows.jsonl')) if r.get('kind') == 'meas' and 'error' not in r}
    keys = [k for k in diag if k[1] == 125.0][:2]
    with Pool(2) as p:
        for key, (nmse, info, sec) in p.imap_unordered(job, keys):
            d, g = dev[key], diag[key]
            print(key, sec)
            print('  cubature vs dev1-box:', nmse['mo26_cap5_cubature'] == d['nmse']['mo26_cap5'], nmse['gsmf_cubature'] == d['nmse']['gsmf'])
            print('  exact vs diagnostic series:', nmse['mo26_cap5_exact'] == g['nmse']['series']['mo26_cap5'], nmse['gsmf_exact'] == g['nmse']['series']['gsmf'])
            print('  cubature vs diagnostic cubature:', nmse['mo26_cap5_cubature'] == g['nmse']['cubature']['mo26_cap5'], nmse['gsmf_cubature'] == g['nmse']['cubature']['gsmf'], flush=True)
