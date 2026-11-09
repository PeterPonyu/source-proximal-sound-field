"""Registered analysis of TB-M2-EXACT-HI (PROTOCOLS/TB-M2-EXACT-HI.md).

python3 -B analyze_exact_hi.py studies/TB-M2-EXACT-HI/primary  ->  summary.json in that folder

Rows: the exact-kernel gsmf (`gsmf_exact`) and its in-run cubature twin (`gsmf_cubature`) from the TB-M2-EXACT-HI rows
(M in {96, 192}) and, for the combined part, from the TB-M2-EXACT rows (M in {24, 48}); `j_cv_ext` and the registered
`gsmf` from the registered TB-M2 rows. All three row sets must match the SHA-256 in their receipts.
Definitions are TB-M2-EXACT's (analyze_exact.py) and TB-M2's (analyze_v2.py), both imported unchanged: per-problem
error in dB; loudspeakers as units (two draws averaged within a cell); per-cell Student-t intervals at the
family-wise level 0.05 with Bonferroni over TB-M2's 16 MeshRIR cells; win / loss with the 0.1 dB margin;
non-inferior below +0.2 dB; pooled: per loudspeaker the mean of its cell means, 95 % Student-t over the loudspeakers.
Two readings of non-finite errors: as registered (the interval becomes undefined) and finite-only (left out, counted).
Part A: the 8 cells with M in {96, 192}, rule KX-gsmf. Part B: all 16 MeshRIR cells, combining both exact-kernel
studies: rule KX-gsmf over 16 cells, the pooled j_cv_ext - gsmf_exact over the 32 loudspeakers (the number the paper
uses for the gsmf comparison, with the registered -0.27 dB beside it), and the registered M2 rule's gsmf part
recomputed with the exact kernel (descriptive).
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
import analyze_exact as AE  # noqa: E402

REG = HERE / 'studies/TB-M2/primary'
EXACT = HERE / 'studies/TB-M2-EXACT/primary'
PAIRS = (('j_cv_ext', 'gsmf'), ('j_cv_ext', 'gsmf_exact'), ('gsmf_exact', 'gsmf'), ('gsmf_exact', 'gsmf_cubature'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def checked(d):
    rc = json.loads((d / 'run-receipt.json').read_text())
    if sha(d / 'rows.jsonl') != rc['outputs']['rows.jsonl']:
        raise SystemExit(f'rows of {d} do not match their receipt')
    return [json.loads(l) for l in open(d / 'rows.jsonl')], dict(protocol_commit=rc['protocol_commit'], status=rc.get('status'))


def key(r):
    return (r['source'], r['band'], r['M'], r['draw'], r['seed'])


def merge(rows, reg):
    out, repro = [], 0
    for r in rows:
        if 'error' in r:
            continue
        g = reg[key(r)]
        e = {'j_cv_ext': AV.db(g['nmse']['j_cv_ext']), 'gsmf': AV.db(g['nmse']['gsmf']),
             'gsmf_exact': AV.db(r['nmse']['gsmf_exact']), 'gsmf_cubature': AV.db(r['nmse']['gsmf_cubature'])}
        repro += r['nmse']['gsmf_cubature'] == g['nmse']['gsmf']
        out.append(dict(source=r['source'], band=r['band'], M=r['M'], draw=r['draw'], cell_key=f"{int(r['band'])}Hz-M{r['M']}", e=e))
    return out, repro


def table(merged, cells):
    by = defaultdict(list)
    for r in merged:
        by[r['cell_key']].append(r)
    alpha = AV.ALPHA / AE.FAMILY_N
    tab = {}
    for c in cells:
        comp = {}
        for a, b in PAIRS:
            d = AV.cell_diff(by[c], 'mesh', a, b)
            m, lo, hi = AV.tint(list(d.values()), alpha)
            comp[f'{a}-vs-{b}'] = dict(mean=m, low=lo, high=hi, units=len(d), cls=AV.classify(m, lo, hi), non_inferior=bool(hi < AV.NI_MARGIN),
                                       defined=bool(math.isfinite(m) and math.isfinite(lo) and math.isfinite(hi)),
                                       finite_only=AE.finite_cell(by[c], a, b, alpha))
        tab[c] = dict(n=len(by[c]), comparisons=comp)
    pooled = {f'{a}-vs-{b}': dict(AV.pooled(merged, 'mesh', cells, a, b), finite_only=AE.finite_pooled(by, cells, a, b)) for a, b in PAIRS}
    return tab, pooled


def kx(tab, pooled, cells):
    """TB-M2-EXACT's rule for gsmf over these cells, in both readings."""
    k_reg, k_ex = 'j_cv_ext-vs-gsmf', 'j_cv_ext-vs-gsmf_exact'
    res = {}
    for tag in ('as_registered', 'finite_only'):
        get = (lambda c, k: tab[c]['comparisons'][k]) if tag == 'as_registered' else (lambda c, k: tab[c]['comparisons'][k]['finite_only'])
        pk = (lambda k: pooled[k]) if tag == 'as_registered' else (lambda k: pooled[k]['finite_only'])
        rc = {c: get(c, k_reg)['cls'] for c in cells}
        ec = {c: get(c, k_ex)['cls'] for c in cells}
        lost = [c for c in cells if rc[c] == 'win' and ec[c] != 'win']
        losses = [c for c in cells if ec[c] == 'loss']
        p_reg, p_ex = pk(k_reg), pk(k_ex)
        kept = not (AE.defined(p_reg) and p_reg['high'] < 0) or (AE.defined(p_ex) and p_ex['high'] < 0)
        ni_ex = sum(bool(get(c, k_ex)['non_inferior']) for c in cells)
        res[tag] = dict(registered={x: list(rc.values()).count(x) for x in ('win', 'tie', 'loss')},
                        exact={x: list(ec.values()).count(x) for x in ('win', 'tie', 'loss')},
                        wins_lost=lost, cells_lost_by_primary=losses, pooled_registered=p_reg, pooled_exact=p_ex,
                        non_inferior_exact=f'{ni_ex}/{len(cells)}', robust_to_kernel=bool(not lost and not losses and kept))
    return dict(rule='robust to the kernel: no registered cell win of j_cv_ext becomes a tie or loss with the exact kernel, no cell '
                     'loss to the exact gsmf, and a registered pooled win (defined, upper bound < 0) stays a pooled win', **res)


def stop_hit(rule):
    return any(rule[t]['cells_lost_by_primary'] or (AE.defined(rule[t]['pooled_exact']) and rule[t]['pooled_exact']['low'] > 0)
               for t in ('as_registered', 'finite_only'))


def order(cells):
    return sorted(cells, key=lambda c: (int(c.split('Hz')[0]), int(c.split('M')[1])))


def main():
    out = Path(sys.argv[1])
    rows_hi = [json.loads(l) for l in open(out / 'rows.jsonl')]
    reg_rows, reg_rc = checked(REG)
    reg = {key(r): r for r in reg_rows if 'error' not in r and r.get('kind') == 'mesh'}
    ex_rows, ex_rc = checked(EXACT)
    hi, repro_hi = merge(rows_hi, reg)
    lo, repro_lo = merge(ex_rows, reg)
    cells_a = order({r['cell_key'] for r in hi})
    tab_a, pooled_a = table(hi, cells_a)
    rule_a = kx(tab_a, pooled_a, cells_a)
    both = lo + hi
    cells_b = order({r['cell_key'] for r in both})
    tab_b, pooled_b = table(both, cells_b)
    rule_b = kx(tab_b, pooled_b, cells_b)
    m2 = {}
    for tag in ('as_registered', 'finite_only'):
        get = (lambda c: tab_b[c]['comparisons']['j_cv_ext-vs-gsmf_exact']) if tag == 'as_registered' else \
            (lambda c: tab_b[c]['comparisons']['j_cv_ext-vs-gsmf_exact']['finite_only'])
        p = pooled_b['j_cv_ext-vs-gsmf_exact'] if tag == 'as_registered' else pooled_b['j_cv_ext-vs-gsmf_exact']['finite_only']
        ni = sum(bool(get(c)['non_inferior']) for c in cells_b)
        m2[tag] = dict(non_inferior=f'{ni}/{len(cells_b)}', pooled=p,
                       passed=bool(ni == len(cells_b) and len(cells_b) == 16 and AE.defined(p) and p['high'] < 0))
    head = pooled_b['j_cv_ext-vs-gsmf_exact']
    res = dict(study='TB-M2-EXACT-HI', n_rows=len(rows_hi), n_errors=sum('error' in r for r in rows_hi), problems=len(hi),
               registered_rows=reg_rc, tb_m2_exact_rows=ex_rc, family_n=AE.FAMILY_N,
               cubature_reproduces_registered=dict(hi=f'{repro_hi}/{len(hi)}', tb_m2_exact_gsmf=f'{repro_lo}/{len(lo)}'),
               part_a=dict(cells=cells_a, rule_kx_gsmf=rule_a, pooled=pooled_a, cells_table=tab_a),
               part_b=dict(cells=cells_b, problems=len(both), complete=bool(len(cells_b) == 16 and len(both) == 1024),
                           headline=dict(reading='pooled j_cv_ext - gsmf_exact over the 16 MeshRIR cells, 32 loudspeakers; the paper uses '
                                                 'it for the gsmf comparison, with the registered value (cubature) beside it',
                                         exact=head, registered=pooled_b['j_cv_ext-vs-gsmf']),
                           rule_kx_gsmf=rule_b, m2_gsmf_part_with_exact_kernel=m2, pooled=pooled_b, cells_table=tab_b),
               stop=dict(rule='ask the user before any manuscript work if j_cv_ext loses a cell or shows a pooled loss to the exact gsmf, '
                              'in either reading, in part A or part B', triggered=bool(stop_hit(rule_a) or stop_hit(rule_b))))
    (out / 'summary.json').write_text(json.dumps(res, indent=1, default=float))
    print('cubature reproduces the registered rows:', res['cubature_reproduces_registered'])
    for name, rule in (('A (M 96, 192)', rule_a), ('B (16 cells)', rule_b)):
        for t in ('as_registered', 'finite_only'):
            r = rule[t]
            print(f"KX-gsmf {name} [{t}]: registered {r['registered']} exact {r['exact']} wins lost {r['wins_lost']} "
                  f"cell losses {r['cells_lost_by_primary']} non-inferior {r['non_inferior_exact']} robust {r['robust_to_kernel']}")
    for t in ('as_registered', 'finite_only'):
        print(f"M2 gsmf part with the exact kernel [{t}]: non-inferior {m2[t]['non_inferior']}, passed {m2[t]['passed']}")
    for k, iv in pooled_b.items():
        f = iv['finite_only']
        print(f"  16-cell pooled {k}: {iv['mean']:+.3f} [{iv['low']:+.3f}, {iv['high']:+.3f}]; finite-only {f['mean']:+.3f} "
              f"[{f['low']:+.3f}, {f['high']:+.3f}] left out {f['problems_left_out']}")
    print('combined complete:', res['part_b']['complete'], 'stop:', res['stop']['triggered'])


if __name__ == '__main__':
    main()
