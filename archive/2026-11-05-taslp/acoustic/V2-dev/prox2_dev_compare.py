import json, math, sys, numpy as np
from collections import defaultdict
R = sys.argv[1] if len(sys.argv) > 1 else 'V2-dev/prox-dev/N3/rows.jsonl'
rows = [json.loads(l) for l in open(R)]
err = [r for r in rows if 'error' in r]
rows = [r for r in rows if 'error' not in r]
print('rows', len(rows), 'errors', len(err))
for e in err[:3]: print(e['task'], e['error'][-300:])
def key(r): return (r['kind'], r.get('cell'), r.get('seed')) if r['kind'] == 'sim' else (r['kind'], r['source'], r['band'], r['M'], r['draw'], r['seed'])
dev = {key(r): r for r in map(json.loads, open('V2-dev/dev1-box/rows.jsonl')) if 'error' not in r}
db = lambda v: 10 * math.log10(float(np.mean(v))) if all(math.isfinite(x) for x in v) else float('nan')
probs = sorted({key(r) for r in rows}, key=str)
print('problems', len(probs))
# gsmf grid: mean dB over problems per (sigma2, zeta)
G = defaultdict(dict)
for r in rows:
    if r['variant'] == 'gsmf':
        G[(round(r['sigma2'], 4), r['zeta'])][key(r)] = db(r['nmse'])
print('gsmf grid: (sigma2, zeta): n, mean dB, mean (gsmf_prox - registered gsmf)')
for c in sorted(G):
    v = G[c]; d = [v[k] - db(dev[k]['nmse']['gsmf']) for k in v]
    print('  ', c, len(v), round(np.mean(list(v.values())), 2), round(np.mean(d), 2), round(np.median(d), 2))
best = min((c for c in G if len(G[c]) == max(len(x) for x in G.values())), key=lambda c: np.mean(list(G[c].values())))
print('best gsmf', best)
mo = {key(r): r for r in rows if r['variant'] == 'mo26'}
print('mo26 rows', len(mo))
tab = []
for k in probs:
    d = dev[k]['nmse']
    e = dict(j_cv_ext=db(d['j_cv_ext']), gsmf=db(d['gsmf']), mo26_cap5=db(d['mo26_cap5']), sinc_ev=db(d['sinc_ev']))
    e['gsmf_prox'] = G[best].get(k, float('nan'))
    e['mo26_prox'] = db(mo[k]['nmse']) if k in mo else float('nan')
    tab.append((k, e))
    print(k[:4] if k[0]=='meas' else k, {m: round(v, 1) for m, v in e.items()})
def med(a, b):
    x = [e[a] - e[b] for _, e in tab if math.isfinite(e[a]) and math.isfinite(e[b])]
    return round(float(np.median(x)), 2), round(float(np.mean(x)), 2), len(x), sum(v < 0 for v in x)
for a, b in (('j_cv_ext', 'gsmf_prox'), ('j_cv_ext', 'mo26_prox'), ('gsmf_prox', 'gsmf'), ('mo26_prox', 'mo26_cap5'), ('gsmf_prox', 'sinc_ev'), ('mo26_prox', 'sinc_ev'), ('j_cv_ext', 'gsmf'), ('j_cv_ext', 'mo26_cap5')):
    print(f'{a} - {b}: median, mean, n, count<0 =', med(a, b))
# mo26 iterations and timing
it1 = []; it2 = []; tt = []; reg = []
for k, r in mo.items():
    for b in r['info']:
        it1.append(b['iterations'][0]); it2.append(b['iterations'][1]); reg.append(b['after_stage1'])
    tt.append((r['M_mics'], r['fit_s'], r['prepare_s'], r['S']))
it1 = np.array(it1); it2 = np.array(it2); reg = np.array(reg)
print('mo26 stage-1 iterations: median', np.median(it1), 'min', it1.min(), 'max', it1.max(), 'fraction at 2000:', np.mean(it1 >= 2000), 'fraction stopped early:', np.mean(it1 < 2000))
print('mo26 stage-2 iterations: median', np.median(it2), 'max', it2.max(), '; active regions after stage 1: median', np.median(reg), 'max', reg.max())
for M in (24, 48):
    x = [t for t in tt if t[0] == M]
    print('mo26 M', M, 'fit_s median', np.median([t[1] for t in x]), 'max', max(t[1] for t in x), 'prepare median', np.median([t[2] for t in x]))
g = [r for r in rows if r['variant'] == 'gsmf']
for M in (24, 48):
    x = [r for r in g if r['M_mics'] == M]
    print('gsmf M', M, 'fit_s median', np.median([r['fit_s'] for r in x]), 'prepare median', np.median([r['prepare_s'] for r in x]), 'iters', set(r['info']['iterations'] for r in x))
print('--- spread and power')
from scipy import stats
for a, b in (('j_cv_ext', 'gsmf_prox'), ('j_cv_ext', 'mo26_prox'), ('gsmf_prox', 'gsmf'), ('mo26_prox', 'mo26_cap5')):
    x = np.array([e[a] - e[b] for _, e in tab if math.isfinite(e[a]) and math.isfinite(e[b])])
    meas = np.array([e[a] - e[b] for k, e in tab if k[0] == 'meas' and math.isfinite(e[a]) and math.isfinite(e[b])])
    # per-loudspeaker means (meas only, S4/S7/S8 over 4 bands)
    spk = defaultdict(list)
    for k, e in tab:
        if k[0] == 'meas' and math.isfinite(e[a]) and math.isfinite(e[b]):
            spk[k[1]].append(e[a] - e[b])
    um = [np.mean(v) for v in spk.values()]
    print(f'{a} - {b}: mean {x.mean():+.2f}, per-problem SD {x.std(ddof=1):.2f}; measured per-loudspeaker means {np.round(um,2)} (SD {np.std(um, ddof=1):.2f})')
    for n in (32, 16, 8, 6):
        sd = max(np.std(um, ddof=1), 1.0)
        h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
        print(f'     n={n}: 95% half-width about {h:.2f} dB (SD {sd:.2f}); effect/half-width {abs(x.mean())/h:.1f}')
# stage split for mo26: approx stage-1 time from gsmf on the same problem/sigma? use gsmf fit at same M as proxy
