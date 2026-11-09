import json, sys, math, numpy as np
from collections import defaultdict
SP = sys.argv[1]
def key(r):
    return (r['kind'], r.get('cell'), r.get('seed')) if r['kind'] == 'sim' else (r['kind'], r['source'], r['band'], r['M'], r['draw'], r['seed'])
dev = {key(r): r for r in map(json.loads, open('V2-dev/dev1-box/rows.jsonl')) if 'error' not in r}
db = lambda v: 10 * math.log10(float(np.mean(v)))
for name in sys.argv[2:]:
    rows = [json.loads(l) for l in open(f'{SP}/{name}/rows.jsonl')]
    err = [r for r in rows if 'error' in r]
    ok = [r for r in rows if 'error' not in r and key(r) in dev]
    out = defaultdict(list); tm = defaultdict(list); bad = 0
    for r in ok:
        d = dev[key(r)]['nmse']
        for m in ('mo26_prox', 'gsmf_prox'):
            if not all(math.isfinite(v) for v in r['nmse'][m]):
                bad += 1; continue
        e = {m: db(r['nmse'][m]) for m in ('mo26_prox', 'gsmf_prox')}
        ref = {m: db(d[m]) for m in ('mo26_cap5', 'gsmf', 'j_cv_ext', 'sinc_ev')}
        grp = r['kind']
        for a, b in (('mo26_prox', 'mo26_cap5'), ('gsmf_prox', 'gsmf'), ('mo26_prox', 'sinc_ev'), ('gsmf_prox', 'sinc_ev')):
            out[(grp, a, b)].append(e[a] - ref[b])
        out[(grp, 'j_cv_ext', 'mo26_prox')].append(ref['j_cv_ext'] - e['mo26_prox'])
        out[(grp, 'j_cv_ext', 'gsmf_prox')].append(ref['j_cv_ext'] - e['gsmf_prox'])
        tm[(grp, r['M_mics'])].append(r['timing']['mo26_prox'] + r['timing']['gsmf_prox'])
    print(f'== {name} cfg {json.load(open(f"{SP}/{name}/config.json"))["config"]} rows {len(rows)} errors {len(err)} nonfinite {bad}')
    for k in sorted(out):
        v = np.array(out[k]); print(f'   {k[0]:4s} {k[1]:9s} - {k[2]:9s}: mean {v.mean():+6.2f} dB  median {np.median(v):+6.2f}  n {len(v)}')
    print('   cpu s per problem (both variants):', {f'{g}-M{M}': round(float(np.mean(v)), 1) for (g, M), v in sorted(tm.items())})
