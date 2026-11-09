"""Summary of the cubature diagnostic rows (descriptive): radii, k_max R, per-problem dB change (series minus cubature)
for j_cv_ext, mo26_cap5 and gsmf, and the reproduction of the development rows by the cubature runs.
python3 -B cubature_diag_summary.py ROWS.jsonl"""
import json, math, sys
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
err = [r for r in rows if 'error' in r]; rows = [r for r in rows if 'error' not in r]
def key(r): return (r['kind'], r.get('cell'), r.get('seed')) if r['kind'] == 'sim' else (r['kind'], r['source'], r['band'], r['M'], r['draw'], r['seed'])
dev = {key(r): r for r in map(json.loads, open('V2-dev/dev1-box/rows.jsonl')) if 'error' not in r}
db = lambda v: 10 * math.log10(float(np.mean(v))) if all(math.isfinite(x) for x in v) else float('nan')
print('rows', len(rows), 'errors', len(err))
for e in err: print('ERROR', e['task'], e['error'][-400:])
same = {m: [] for m in ('j_cv_ext', 'mo26_cap5', 'gsmf')}
delta = {m: [] for m in same}
kR = []
for r in rows:
    d = dev[key(r)]['nmse']
    for m in same:
        same[m].append(r['nmse']['cubature'][m] == d[m])
        delta[m].append(db(r['nmse']['series'][m]) - db(r['nmse']['cubature'][m]))
    kR.append(r['k_max_R']['cubature'])
    tag = r.get('cell') or f"{r['source']} {int(r['band'])} M{r['M']}"
    print(f"{tag:22s} K_cv {r['K_cv']} k_maxR(cub) {r['k_max_R']['cubature']:.2f} radii j {np.round(r['radii']['cubature']['j_cv_ext'],3)} gsmf {np.round(r['radii']['cubature']['gsmf'],3)};"
          f" dB change j {delta['j_cv_ext'][-1]:+.2f} mo26 {delta['mo26_cap5'][-1]:+.2f} gsmf {delta['gsmf'][-1]:+.2f}")
print('cubature runs reproduce the development rows bit for bit:', {m: f'{sum(v)}/{len(v)}' for m, v in same.items()})
for m, v in delta.items():
    v = np.array(v); print(f'{m}: series minus cubature dB: median {np.median(v):+.3f}, mean {v.mean():+.3f}, min {v.min():+.2f}, max {v.max():+.2f}, |change| > 0.1 dB in {int(np.sum(np.abs(v) > 0.1))}/{len(v)}')
print('k_max R (cubature fits): median', round(float(np.median(kR)), 2), 'max', round(float(max(kR)), 2))
print('--- paired summary, exact minus cubature (dB), 95 % Student-t over the problems')
from scipy import stats
for m, v in delta.items():
    v = np.array(v); n = len(v)
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n)
    print(f'{m}: n {n}, mean {v.mean():+.3f} [{v.mean() - h:+.3f}, {v.mean() + h:+.3f}], median {np.median(v):+.3f};'
          f' rule (interval entirely below -0.3 dB): {"TRIGGERED" if v.mean() + h < -0.3 else "not triggered"}')
