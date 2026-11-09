"""Registered analysis of TB-PROX-S2 (PROTOCOLS/TB-PROX-S2.md).

python3 -B analyze_prox_s2.py studies/TB-PROX-S2/primary  ->  summary.json in that folder

mo26-prox with the paper's stage-2 stop (`mo26_prox_s2`) comes from the TB-PROX-S2 rows; the primary `j_cv_ext` and
the registered `mo26_cap5` from the registered TB-M2 rows, and mo26-prox with the registered (penalised) stop
(`mo26_prox`) from the TB-PROX rows; both earlier sets must match the SHA-256 in their run receipts.
Metric, units, intervals and classification exactly as in analyze_prox.py (TB-PROX): per problem, 10 log10 of the mean
NMSE over the bins of the band; paired differences in dB (negative favours the first method); a problem with a
non-finite or missing error in either method of a pair is left out of that pair and counted; units are the 32
loudspeakers; pooled 95 % Student-t interval; per-cell (4 bands) descriptive classification at a Bonferroni-corrected
level 0.05 with the 0.1 dB margin.
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
sys.path.insert(0, str(HERE))
import analyze_prox as AP  # noqa: E402

REG = HERE / 'studies/TB-M2/primary'
PROX = HERE / 'studies/TB-PROX/primary'
PAIRS = (('j_cv_ext', 'mo26_prox_s2'),       # (a) the primary against the paper's-stop reading (rule PXS)
         ('mo26_prox_s2', 'mo26_cap5'),      # (b) stronger reading against the registered comparator
         ('mo26_prox_s2', 'mo26_prox'))      # (c) descriptive: the size of the stop deviation in TB-PROX
METHODS = ('j_cv_ext', 'mo26_cap5', 'mo26_prox', 'mo26_prox_s2')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_checked(d, kind='mesh'):
    rc = json.loads((d / 'run-receipt.json').read_text())
    ok = sha(d / 'rows.jsonl') == rc['outputs']['rows.jsonl']
    if not ok:
        raise SystemExit(f'rows of {d} do not match their receipt')
    rows = [json.loads(l) for l in open(d / 'rows.jsonl')]
    return [r for r in rows if 'error' not in r and r.get('kind') == kind], dict(rows_sha256_matches_receipt=ok,
                                                                                  protocol_commit=rc['protocol_commit'], status=rc.get('status'))


def main():
    out = Path(sys.argv[1])
    rows = [json.loads(l) for l in open(out / 'rows.jsonl')]
    errors = [r for r in rows if 'error' in r]
    s2 = {AP.key(r): r for r in rows if 'error' not in r}
    reg_rows, reg_rc = load_checked(REG)
    reg = {AP.key(r): r for r in reg_rows}
    prox_rows, prox_rc = load_checked(PROX)
    prox = {AP.key(r): r for r in prox_rows if 'mo26_prox' in r['nmse']}
    err = defaultdict(dict)
    for k, r in s2.items():
        if k not in reg:
            raise SystemExit(f'no registered row for {k}')
        err[k]['mo26_prox_s2'] = AP.db(r['nmse']['mo26_prox_s2'])
        err[k]['mo26_prox'] = AP.db(prox[k]['nmse']['mo26_prox']) if k in prox else float('nan')
        for m in ('j_cv_ext', 'mo26_cap5'):
            err[k][m] = AP.db(reg[k]['nmse'][m])
    ks = sorted(err)
    unit_of = lambda k: k[1]
    cell_of = lambda k: (k[2], k[3])
    arm = dict(problems=len(ks), nonfinite_or_missing={m: int(sum(not math.isfinite(err[k][m]) for k in ks)) for m in METHODS},
               pooled={}, cells={})
    cells = sorted({cell_of(k) for k in ks}, key=str)
    for a, b in PAIRS:
        name = f'{a}-vs-{b}'
        per_unit, per_cell = defaultdict(list), defaultdict(lambda: defaultdict(list))
        dropped = 0
        for k in ks:
            d = err[k][a] - err[k][b]
            if not math.isfinite(d):
                dropped += 1; continue
            per_unit[unit_of(k)].append(d)
            per_cell[cell_of(k)][unit_of(k)].append(d)
        iv = AP.interval([np.mean(v) for v in per_unit.values()], 0.95)
        iv.update(problems_left_out=dropped)
        arm['pooled'][name] = iv
        lvl = 1 - AP.ALPHA / len(cells)
        for c in cells:
            civ = AP.interval([np.mean(v) for v in per_cell[c].values()], lvl)
            civ['cls'] = AP.cls(civ)
            arm['cells'].setdefault(str(c), {})[name] = civ
    P = arm['pooled']
    pa, pb, pc = (P[f'{a}-vs-{b}'] for a, b in PAIRS)
    arm['rules'] = {'primary_vs_mo26_prox_s2': dict(rule='j_cv_ext pooled superior to mo26_prox_s2 (upper bound < 0)',
                                                    passed=bool(pa['high'] is not None and pa['high'] < 0))}
    arm['stronger_reading'] = {'mo26_prox_s2': ('author recipe stronger' if pb['high'] is not None and pb['high'] < 0 else
                                                'registered stronger' if pb['low'] is not None and pb['low'] > 0 else 'indistinguishable')}
    arm['report_against_prox_s2'] = arm['stronger_reading']['mo26_prox_s2'] == 'author recipe stronger'
    arm['stop_deviation'] = dict(reading='mo26_prox_s2 minus mo26_prox (paper stop minus registered stop), descriptive',
                                 **{k: pc[k] for k in ('mean', 'low', 'high', 'units', 'problems_left_out')})
    arm['stop'] = dict(rule='ask the user before any manuscript work if j_cv_ext shows a pooled loss to mo26_prox_s2 (lower bound > 0)',
                       triggered=bool(pa['low'] is not None and pa['low'] > 0))
    stage2 = [b['iterations'][1] for r in s2.values() for b in r['info']['bins']]
    arm['stage2_lengths'] = dict(bins=len(stage2), median=float(np.median(stage2)) if stage2 else None,
                                 at_cap=int(sum(n >= 101 for n in stage2)))
    res = dict(study='TB-PROX-S2', n_rows=len(rows), n_errors=len(errors), registered_rows=reg_rc, tb_prox_rows=prox_rc,
               arms={'MeshRIR': arm})
    (out / 'summary.json').write_text(json.dumps(res, indent=1, default=float))
    for r, v in arm['rules'].items():
        print(f"MeshRIR {r}: {'PASS' if v['passed'] else 'FAIL'}  {v['rule']}")
    for p, iv in P.items():
        lo = 'nan' if iv['low'] is None else f"{iv['low']:+.2f}"; hi = 'nan' if iv['high'] is None else f"{iv['high']:+.2f}"
        print(f"  MeshRIR {p}: {iv['mean']:+.2f} dB [{lo}, {hi}] units {iv['units']}, left out {iv['problems_left_out']}")
    print(f"  stronger reading: {arm['stronger_reading']}; report against prox_s2: {arm['report_against_prox_s2']}; stop: {arm['stop']['triggered']}")
    print(f"  stage-2 lengths: {arm['stage2_lengths']}")


if __name__ == '__main__':
    main()
