"""Registered analysis of TB-PROX (PROTOCOLS/TB-PROX.md).

python3 -B analyze_prox.py studies/TB-PROX/primary  ->  summary.json in that folder

The author-recipe variants come from the TB-PROX rows (one row per problem and variant, merged here); the primary
`j_cv_ext` and the registered comparators `mo26_cap5` and `gsmf` come from the registered TB-M2 rows, whose SHA-256 must
equal their run receipt.
Per problem, a method's error is 10 log10 of its mean NMSE over the bins of the band; a paired difference is in dB
(negative favours the first method). A problem with a non-finite error in either method of a pair is left out of that
pair and counted.
Units: the loudspeaker (the mean over its problems, 4 bands at M = 24, when none is missing). Pooled comparison: the
mean over units with a 95 % Student-t interval. Per-cell results (band x M, 32 loudspeakers) are descriptive,
with the TB-M2 classification at a Bonferroni-corrected family-wise level 0.05 (win: upper bound < 0 and mean < -0.1 dB;
loss: lower bound > 0 and mean > +0.1 dB; tie otherwise).
"""
from __future__ import annotations
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
DELTA, ALPHA = 0.1, 0.05
REG = {'mesh': HERE / 'studies/TB-M2/primary'}
PAIRS = (('j_cv_ext', 'mo26_prox'), ('j_cv_ext', 'gsmf_prox'),            # (a) the primary against each variant
         ('mo26_prox', 'mo26_cap5'), ('gsmf_prox', 'gsmf'))               # (b) which reading is stronger
ARMS = {'mesh': 'MeshRIR'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def key(r):
    return ('sim', r['cell'], r['seed']) if r['kind'] == 'sim' else ('mesh', r['source'], r['band'], r['M'], r['draw'], r['seed'])


def db(v):
    v = np.asarray(v, float)
    m = float(np.mean(v)) if v.size and np.all(np.isfinite(v)) else float('nan')
    return 10 * math.log10(m) if m > 0 and math.isfinite(m) else float('nan')


def interval(x, level):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return dict(mean=float(np.mean(x)) if n else None, low=None, high=None, units=n)
    m, s = float(np.mean(x)), float(np.std(x, ddof=1))
    h = stats.t.ppf(1 - (1 - level) / 2, n - 1) * s / math.sqrt(n)
    return dict(mean=m, low=m - h, high=m + h, units=n)


def cls(iv):
    if iv['low'] is None:
        return 'undefined'
    if iv['high'] < 0 and iv['mean'] < -DELTA:
        return 'win'
    if iv['low'] > 0 and iv['mean'] > DELTA:
        return 'loss'
    return 'tie'


def main():
    out = Path(sys.argv[1])
    rows = [json.loads(l) for l in open(out / 'rows.jsonl')]
    errors = [r for r in rows if 'error' in r]
    prox = {}
    for r in rows:
        if 'error' not in r:
            prox.setdefault(key(r), dict(r, nmse={}))['nmse'].update(r['nmse'])
    reg, receipts = {}, {}
    for kind, d in REG.items():
        rc = json.loads((d / 'run-receipt.json').read_text())
        ok = sha(d / 'rows.jsonl') == rc['outputs']['rows.jsonl']
        receipts[kind] = dict(rows_sha256_matches_receipt=ok, protocol_commit=rc['protocol_commit'])
        if not ok:
            raise SystemExit(f'registered rows of {d} do not match their receipt')
        for l in open(d / 'rows.jsonl'):
            r = json.loads(l)
            if 'error' not in r and r.get('kind') == kind:
                reg[key(r)] = r
    err = defaultdict(dict)
    for k, r in prox.items():
        if k not in reg:
            raise SystemExit(f'no registered row for {k}')
        for m in ('mo26_prox', 'gsmf_prox'):
            err[k][m] = db(r['nmse'][m]) if m in r['nmse'] else float('nan')
        for m in ('j_cv_ext', 'mo26_cap5', 'gsmf'):
            err[k][m] = db(reg[k]['nmse'][m])
    res = dict(study='TB-PROX', n_rows=len(rows), n_errors=len(errors), registered_rows=receipts, arms={})
    for kind in ('mesh',):
        ks = [k for k in err if k[0] == kind]
        unit_of = (lambda k: k[1]) if kind == 'mesh' else (lambda k: k[2])
        cell_of = (lambda k: (k[2], k[3])) if kind == 'mesh' else (lambda k: k[1])
        arm = dict(problems=len(ks), nonfinite={m: int(sum(not math.isfinite(err[k][m]) for k in ks))
                                                for m in ('j_cv_ext', 'mo26_cap5', 'gsmf', 'mo26_prox', 'gsmf_prox')},
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
            iv = interval([np.mean(v) for v in per_unit.values()], 0.95)
            iv.update(problems_left_out=dropped)
            arm['pooled'][name] = iv
            lvl = 1 - ALPHA / len(cells)
            for c in cells:
                civ = interval([np.mean(v) for v in per_cell[c].values()], lvl)
                civ['cls'] = cls(civ)
                arm['cells'].setdefault(str(c), {})[name] = civ
        P = arm['pooled']
        arm['rules'] = {
            'primary_vs_mo26_prox': dict(rule='j_cv_ext pooled superior to mo26_prox (upper bound < 0)',
                                         passed=bool(P['j_cv_ext-vs-mo26_prox']['high'] is not None and P['j_cv_ext-vs-mo26_prox']['high'] < 0)),
            'primary_vs_gsmf_prox': dict(rule='j_cv_ext pooled superior to gsmf_prox (upper bound < 0)',
                                         passed=bool(P['j_cv_ext-vs-gsmf_prox']['high'] is not None and P['j_cv_ext-vs-gsmf_prox']['high'] < 0))}
        reading = {}
        for v, regd in (('mo26_prox', 'mo26_cap5'), ('gsmf_prox', 'gsmf')):
            iv = P[f'{v}-vs-{regd}']
            reading[v] = ('author recipe stronger' if iv['high'] is not None and iv['high'] < 0 else
                          'registered stronger' if iv['low'] is not None and iv['low'] > 0 else 'indistinguishable')
        arm['stronger_reading'] = reading
        arm['report_against_prox'] = [v for v, s in reading.items() if s == 'author recipe stronger']
        arm['stop'] = dict(rule='ask the user before any manuscript work if j_cv_ext shows a pooled loss to either variant (lower bound > 0)',
                           triggered=bool(any(P[f'j_cv_ext-vs-{v}']['low'] is not None and P[f'j_cv_ext-vs-{v}']['low'] > 0
                                              for v in ('mo26_prox', 'gsmf_prox'))))
        res['arms'][ARMS[kind]] = arm
    (out / 'summary.json').write_text(json.dumps(res, indent=1, default=float))
    for name, arm in res['arms'].items():
        for r, v in arm['rules'].items():
            print(f"{name} {r}: {'PASS' if v['passed'] else 'FAIL'}  {v['rule']}")
        for p, iv in arm['pooled'].items():
            lo = 'nan' if iv['low'] is None else f"{iv['low']:+.2f}"; hi = 'nan' if iv['high'] is None else f"{iv['high']:+.2f}"
            print(f"  {name} {p}: {iv['mean']:+.2f} dB [{lo}, {hi}] units {iv['units']}, left out {iv['problems_left_out']}")
        print(f"  {name} stronger reading: {arm['stronger_reading']}; report against prox: {arm['report_against_prox']}; stop: {arm['stop']['triggered']}")


if __name__ == '__main__':
    main()
