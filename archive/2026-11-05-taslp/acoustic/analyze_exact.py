"""Registered analysis of TB-M2-EXACT (PROTOCOLS/TB-M2-EXACT.md).

python3 -B analyze_exact.py studies/TB-M2-EXACT/primary  ->  summary.json in that folder

The exact-kernel comparators (`mo26_cap5_exact`, `gsmf_exact`) and their in-run cubature twins (`*_cubature`) come
from the TB-M2-EXACT rows; the primary `j_cv_ext` and the registered `mo26_cap5` and `gsmf` from the registered TB-M2
rows, whose SHA-256 must equal their run receipt.
Metric, units, cells and classification are TB-M2's (analyze_v2.py, imported unchanged): per problem, 10 log10 of the
mean NMSE over the bins of the band; paired differences in dB (negative favours the first method); the loudspeaker is
the unit (its two draws averaged); per cell (band x M) a Student-t interval at the family-wise level 0.05 with
Bonferroni over TB-M2's 16 MeshRIR cells (so a cell's level equals TB-M2's), win / loss with the 0.1 dB margin,
non-inferior below +0.2 dB; pooled over the 8 cells here, 95 % Student-t over the 32 loudspeakers. A non-finite error
makes its cell's and the pooled interval undefined, as in TB-M2; a second, finite-only reading leaves such problems out
of that pair and counts them.
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
import analyze_v2 as AV  # noqa: E402

REG = HERE / 'studies/TB-M2/primary'
FAMILY_N = 16                    # TB-M2's MeshRIR cell family (4 bands x 4 microphone counts)
COMPS = ('gsmf', 'mo26_cap5')
PAIRS = [('j_cv_ext', c) for c in COMPS] + [('j_cv_ext', f'{c}_exact') for c in COMPS] + \
        [(f'{c}_exact', c) for c in COMPS] + [(f'{c}_exact', f'{c}_cubature') for c in COMPS]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def finite_cell(rs, a, b, alpha):
    per, out = defaultdict(list), 0
    for r in rs:
        d = r['e'][a] - r['e'][b]
        if math.isfinite(d):
            per[r['source']].append(d)
        else:
            out += 1
    m, lo, hi = AV.tint([np.mean(v) for v in per.values()], alpha)
    return dict(mean=m, low=lo, high=hi, units=len(per), problems_left_out=out, cls=AV.classify(m, lo, hi), non_inferior=bool(hi < AV.NI_MARGIN))


def finite_pooled(by, cells, a, b):
    per, out = defaultdict(list), 0
    for c in cells:
        cu = defaultdict(list)
        for r in by[c]:
            d = r['e'][a] - r['e'][b]
            if math.isfinite(d):
                cu[r['source']].append(d)
            else:
                out += 1
        for u, v in cu.items():
            per[u].append(np.mean(v))
    m, lo, hi = AV.tint([np.mean(v) for v in per.values()], AV.ALPHA)
    return dict(mean=m, low=lo, high=hi, units=len(per), problems_left_out=out)


def defined(iv):
    return all(math.isfinite(iv[k]) for k in ('mean', 'low', 'high'))


def main():
    out = Path(sys.argv[1])
    rows = [json.loads(l) for l in open(out / 'rows.jsonl')]
    errors = [r for r in rows if 'error' in r]
    rc = json.loads((REG / 'run-receipt.json').read_text())
    if sha(REG / 'rows.jsonl') != rc['outputs']['rows.jsonl']:
        raise SystemExit('registered TB-M2 rows do not match their receipt')
    key = lambda r: (r['source'], r['band'], r['M'], r['draw'], r['seed'])
    reg = {key(r): r for r in map(json.loads, open(REG / 'rows.jsonl')) if 'error' not in r and r.get('kind') == 'mesh'}
    merged, repro = [], {c: 0 for c in COMPS}
    for r in rows:
        if 'error' in r:
            continue
        g = reg[key(r)]
        e = {m: AV.db(g['nmse'][m]) for m in ('j_cv_ext',) + COMPS}
        e.update({m: AV.db(v) for m, v in r['nmse'].items()})
        for c in COMPS:
            repro[c] += r['nmse'][f'{c}_cubature'] == g['nmse'][c]
        merged.append(dict(source=r['source'], band=r['band'], M=r['M'], draw=r['draw'], cell_key=f"{int(r['band'])}Hz-M{r['M']}", e=e,
                           ))
    cells = sorted({r['cell_key'] for r in merged}, key=lambda c: (int(c.split('Hz')[0]), int(c.split('M')[1])))
    by = defaultdict(list)
    for r in merged:
        by[r['cell_key']].append(r)
    alpha = AV.ALPHA / FAMILY_N
    table = {}
    for c in cells:
        comp = {}
        for a, b in PAIRS:
            d = AV.cell_diff(by[c], 'mesh', a, b)
            m, lo, hi = AV.tint(list(d.values()), alpha)
            comp[f'{a}-vs-{b}'] = dict(mean=m, low=lo, high=hi, units=len(d), cls=AV.classify(m, lo, hi), non_inferior=bool(hi < AV.NI_MARGIN),
                                       defined=bool(math.isfinite(m) and math.isfinite(lo) and math.isfinite(hi)),
                                       finite_only=finite_cell(by[c], a, b, alpha))
        table[c] = dict(n=len(by[c]), comparisons=comp)
    pooled = {f'{a}-vs-{b}': dict(AV.pooled(merged, 'mesh', cells, a, b), finite_only=finite_pooled(by, cells, a, b)) for a, b in PAIRS}
    rules = {}
    for c in COMPS:
        k_reg, k_ex = f'j_cv_ext-vs-{c}', f'j_cv_ext-vs-{c}_exact'
        reading = {}
        for tag in ('as_registered', 'finite_only'):
            get = (lambda c_, k: table[c_]['comparisons'][k]) if tag == 'as_registered' else (lambda c_, k: table[c_]['comparisons'][k]['finite_only'])
            pk = (lambda k: pooled[k]) if tag == 'as_registered' else (lambda k: pooled[k]['finite_only'])
            reg_cls = {c_: get(c_, k_reg)['cls'] for c_ in cells}
            ex_cls = {c_: get(c_, k_ex)['cls'] for c_ in cells}
            lost_wins = [c_ for c_ in cells if reg_cls[c_] == 'win' and ex_cls[c_] != 'win']
            new_losses = [c_ for c_ in cells if ex_cls[c_] == 'loss']
            p_reg, p_ex = pk(k_reg), pk(k_ex)
            pooled_win_kept = not (defined(p_reg) and p_reg['high'] < 0) or (defined(p_ex) and p_ex['high'] < 0)
            robust = bool(not lost_wins and not new_losses and pooled_win_kept)
            reading[tag] = dict(registered=dict(wins=list(reg_cls.values()).count('win'), ties=list(reg_cls.values()).count('tie'),
                                                losses=list(reg_cls.values()).count('loss')),
                                exact=dict(wins=list(ex_cls.values()).count('win'), ties=list(ex_cls.values()).count('tie'),
                                           losses=list(ex_cls.values()).count('loss')),
                                wins_lost=lost_wins, cells_lost_by_primary=new_losses, pooled_registered=pk(k_reg), pooled_exact=p_ex,
                                robust_to_kernel=robust)
        rules[c] = dict(rule='robust to the kernel: no registered cell win of j_cv_ext becomes a tie or loss with the exact kernel, no cell '
                             'loss to the exact comparator, and a registered pooled win (defined, upper bound < 0) stays a pooled win', **reading)
    stop = any(rules[c][t]['cells_lost_by_primary'] or (defined(rules[c][t]['pooled_exact']) and rules[c][t]['pooled_exact']['low'] > 0)
               for c in COMPS for t in ('as_registered', 'finite_only'))
    kx = {tag: {m: [r['info'][tag]['k_max_R'][m] for r in rows if 'error' not in r] for m in COMPS} for tag in ('cubature', 'exact')}
    res = dict(study='TB-M2-EXACT', n_rows=len(rows), n_errors=len(errors), problems=len(merged), cells=cells, family_n=FAMILY_N,
               registered_rows=dict(rows_sha256_matches_receipt=True, protocol_commit=rc['protocol_commit']),
               cubature_reproduces_registered={c: f'{repro[c]}/{len(merged)}' for c in COMPS},
               rules=rules, stop=dict(rule='ask the user before any manuscript work if j_cv_ext loses a cell or shows a pooled loss to an exact-kernel comparator',
                                      triggered=bool(stop)),
               k_max_R={t: {m: dict(median=float(np.median(v)), max=float(np.max(v))) for m, v in d.items() if v} for t, d in kx.items()},
               pooled=pooled, cells_table=table)
    (out / 'summary.json').write_text(json.dumps(res, indent=1, default=float))
    print('cubature reproduces the registered rows:', res['cubature_reproduces_registered'])
    for c in COMPS:
        for t in ('as_registered', 'finite_only'):
            r = rules[c][t]
            print(f"{c} [{t}]: registered {r['registered']} exact {r['exact']} wins lost {r['wins_lost']} cell losses {r['cells_lost_by_primary']} "
                  f"robust {r['robust_to_kernel']}")
    for k, iv in pooled.items():
        f = iv['finite_only']
        print(f"  pooled {k}: {iv['mean']:+.3f} [{iv['low']:+.3f}, {iv['high']:+.3f}]; finite-only {f['mean']:+.3f} [{f['low']:+.3f}, {f['high']:+.3f}] "
              f"left out {f['problems_left_out']}")
    print('stop:', res['stop']['triggered'])


if __name__ == '__main__':
    main()
