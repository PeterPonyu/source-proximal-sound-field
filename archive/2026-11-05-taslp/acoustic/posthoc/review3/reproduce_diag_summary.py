"""Summary of reproduction-diagnostic.jsonl against the reading of PLAN-C2-DIAGNOSTIC.md."""
import json
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
rows = [json.loads(l) for l in open(HERE / 'reproduction-diagnostic.jsonl')]
err = [r for r in rows if 'error' in r]
items = []
for r in rows:
    if 'error' in r:
        continue
    if r['task'][0] == 'bin':
        items.append(('bin', r['task'][2], r['final'], r['K']))
    else:
        for i, b in enumerate(r['bins']):
            items.append(('shared', 100.0 * (i + 1), b, r['K']))
def collapsed(b):
    w = b['log10w']
    near = None if not w else (max(w) if b['dist_m'] is None else w[0])
    return (len(w) == 0 or max(w) <= -3.9) and b['ridge'] < 1e-3
out = dict(errors=len(err))
for kind in ('bin', 'shared'):
    it = [x for x in items if x[0] == kind]
    fail = [x for x in it if x[2]['db'] > -10]; ok = [x for x in it if x[2]['db'] <= -10]
    out[kind] = dict(bins=len(it), failing=len(fail), failing_collapsed=sum(collapsed(x[2]) for x in fail),
                     failing_order0=sum(x[3] == 0 for x in fail), succeeding=len(ok), succeeding_collapsed=sum(collapsed(x[2]) for x in ok),
                     failing_by_freq={str(f): sum(x[1] == f for x in fail) for f in sorted({x[1] for x in it})},
                     sinc_cond_median_by_freq={str(f): float(np.median([x[2]['sinc_cond'] for x in it if x[1] == f])) for f in sorted({x[1] for x in it})},
                     shared_point_within_1cm=(sum(1 for x in it if x[2]['dist_m'] is not None and x[2]['dist_m'] < 0.01) if kind == 'shared' else None))
# added 2026-11-07 after an independent review, descriptive only; the fields above are unchanged.
# The rows keep the distance of the nearest fitted point and the weights in fit order, not which point is nearest,
# so the weight fields describe the first fitted point, not necessarily the one at the source.
for kind in ('bin', 'shared'):
    it = [x for x in items if x[0] == kind]
    fail = [x for x in it if x[2]['db'] > -10]
    d = [x[2]['dist_m'] for x in fail if x[2]['dist_m'] is not None]
    out[kind]['failing_with_point'] = len(d)
    out[kind]['failing_min_dist_m'] = float(min(d)) if d else None
    out[kind]['max_dist_m_all_bins'] = float(max(x[2]['dist_m'] for x in it if x[2]['dist_m'] is not None))
    out[kind]['failing_first_weight_within_a_decade_of_bound'] = sum(1 for x in fail if x[2]['log10w'] and x[2]['log10w'][0] <= -3.0)
(HERE / 'reproduction-diagnostic-summary.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
