"""Post hoc descriptive comparison for the TB-PROX record (not a registered rule): the faithful mo26-prox and, once
amendment 1 has run, gsmf-prox (studies/TB-PROX/gsmf-a1) against the source-free sinc kernel (`sinc_ev`, registered
TB-M2 rows) on the TB-PROX problems, with the registered region-kernel readings and the primary beside them for scale.

python3 -B studies/TB-PROX/descriptive_sinc.py   ->  studies/TB-PROX/descriptive-sinc.json

Metric, units and intervals as analyze_prox.py (imported unchanged): per problem, 10 log10 of the mean NMSE over the
bins; paired differences in dB, negative favouring the first method; the 32 loudspeakers as units (mean of their 4
bands); pooled 95 % Student-t; per band at the Bonferroni-corrected level 0.05 over the 4 bands. Both row sets must
match the SHA-256 in their receipts.
"""
from __future__ import annotations
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import analyze_prox as AP  # noqa: E402

PAIRS = (('mo26_prox', 'sinc_ev'), ('mo26_cap5', 'sinc_ev'), ('j_cv_ext', 'sinc_ev'))
PAIRS_A1 = (('gsmf_prox', 'sinc_ev'), ('gsmf', 'sinc_ev'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def checked(d):
    rc = json.loads((d / 'run-receipt.json').read_text())
    if sha(d / 'rows.jsonl') != rc['outputs']['rows.jsonl']:
        raise SystemExit(f'rows of {d} do not match their receipt')
    return [json.loads(l) for l in open(d / 'rows.jsonl')]


def main():
    prox = [r for r in checked(ROOT / 'studies/TB-PROX/primary') if 'error' not in r and r['variant'] == 'mo26']
    reg = {AP.key(r): r for r in checked(ROOT / 'studies/TB-M2/primary') if 'error' not in r and r.get('kind') == 'mesh'}
    err = {}
    for r in prox:
        k = AP.key(r); g = reg[k]
        err[k] = dict(mo26_prox=AP.db(r['nmse']['mo26_prox']), **{m: AP.db(g['nmse'][m]) for m in ('sinc_ev', 'mo26_cap5', 'j_cv_ext', 'gsmf')})
    pairs = PAIRS
    a1 = ROOT / 'studies/TB-PROX/gsmf-a1'
    if (a1 / 'run-receipt.json').exists():
        for r in checked(a1):
            if 'error' not in r:
                err[AP.key(r)]['gsmf_prox'] = AP.db(r['nmse']['gsmf_prox'])
        for e in err.values():
            e.setdefault('gsmf_prox', float('nan'))
        pairs = PAIRS + PAIRS_A1
    out = dict(note='post hoc descriptive, not a registered rule', problems=len(err), pooled={}, bands={})
    bands = sorted({k[2] for k in err})
    for a, b in pairs:
        name = f'{a}-vs-{b}'
        unit, cell, left = defaultdict(list), defaultdict(lambda: defaultdict(list)), 0
        for k, e in err.items():
            d = e[a] - e[b]
            if not math.isfinite(d):
                left += 1; continue
            unit[k[1]].append(d); cell[k[2]][k[1]].append(d)
        out['pooled'][name] = dict(AP.interval([np.mean(v) for v in unit.values()], 0.95), problems_left_out=left,
                                   loudspeakers_favouring_first=int(sum(np.mean(v) < 0 for v in unit.values())))
        for c in bands:
            iv = AP.interval([np.mean(v) for v in cell[c].values()], 1 - AP.ALPHA / len(bands))
            iv['cls'] = AP.cls(iv)
            out['bands'].setdefault(str(c), {})[name] = iv
    (HERE / 'descriptive-sinc.json').write_text(json.dumps(out, indent=1, default=float))
    for p, iv in out['pooled'].items():
        print(f"{p}: {iv['mean']:+.2f} dB [{iv['low']:+.2f}, {iv['high']:+.2f}], {iv['units']} loudspeakers, "
              f"{iv['loudspeakers_favouring_first']} favour the first, left out {iv['problems_left_out']}")
    for c, d in out['bands'].items():
        print(c, {p: f"{v['mean']:+.2f} [{v['low']:+.2f}, {v['high']:+.2f}] {v['cls']}" for p, v in d.items()})


if __name__ == '__main__':
    main()
