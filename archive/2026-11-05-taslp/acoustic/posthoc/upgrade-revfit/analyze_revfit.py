"""Revised fitting on every registered problem: descriptive analysis (PLAN.md; post hoc).

    python3 -B posthoc/upgrade-revfit/analyze_revfit.py

Reads whichever of revfit-mesh.jsonl, revfit-miracle.jsonl and revfit-sim.jsonl exist next to this file, writes
summary-<study>.json for each and summary-overall.json, and prints a short table. Rerun unchanged when more rows arrive.

Problem error: 10 log10 of the six-bin mean NMSE (analyze_v2.db). Paired difference: revised minus registered (dB).
Statistics are those of the registered analyses, imported unchanged from analyze_v2.py: units (loudspeakers for the
room and anechoic grids, the two draws averaged; fields for the simulation), per-unit cell differences, the 95 %
Student-t pooled interval, and the class rule (Bonferroni over the family's cells, 0.1 dB margin) against the best
generic estimator of each condition (lowest registered cell mean among analyze_v2.GENERIC). Comparator errors are the
registered ones; only j_cv_ext is replaced.
Families as registered: room grid 16 conditions (analyze_v2.py, TB-M2); each anechoic scenario 12 conditions
(analyze_m3.py); simulation: the 63 source-proximal cells, the far control A1-beside1.2 classed at the same level
(analyze_v2.py, TB-S2 S1). The simulation pooled interval is given over the source-proximal cells (the registered
pooling) and over all 64 cells; the stability reading requires both.
"""
from __future__ import annotations
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parents[1]
sys.path.insert(0, str(A))
import analyze_v2 as AV  # noqa: E402

P, REV = AV.PRIMARY, 'revised'
BOUND = 0.1                     # PLAN.md: pooled interval within +-0.1 dB
GATE_TOL = 1e-6                 # PLAN.md gate
REG = {'mesh': 'TB-M2', 'miracle': 'TB-M3-MIRACLE', 'sim': 'TB-S2'}
EXPECTED = {'mesh': 1024, 'miracle': 1536, 'sim': 2048}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ident(key):
    return json.dumps({a: key[a] for a in key if a != 'M' or key[a] is not None}, sort_keys=True)


def reg_key(study, r):
    if study == 'mesh':
        return dict(kind='mesh', source=r['source'], band=r['band'], M=r['M'], draw=r['draw'], seed=r['seed'])
    if study == 'miracle':
        return dict(kind='miracle', scenario=r['scenario'], source=r['source'], band=r['band'], M=r['M'], draw=r['draw'], seed=r['seed'])
    return dict(kind='sim', cell=r['cell'], seed=r['seed'])


def registered(study):
    out = {}
    for l in open(A / 'studies' / REG[study] / 'primary' / 'rows.jsonl'):
        r = json.loads(l)
        if 'error' in r or (study == 'mesh' and r['kind'] != 'mesh'):
            continue
        r['e'] = {m: AV.db(v) for m, v in r['nmse'].items()}
        out[ident(reg_key(study, r))] = r
    return out


def relmax(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def read_rows(path, study, regs):
    """Rows, plus every irregularity found: errors, duplicates, keys outside the registered set, missing problems."""
    raw = [json.loads(l) for l in open(path) if l.strip()]
    errors = [r for r in raw if 'error' in r]
    ok = [r for r in raw if 'error' not in r]
    ids = Counter(ident(r['key']) for r in ok)
    dup = {k: n for k, n in ids.items() if n > 1}
    unknown = [k for k in ids if k not in regs]
    missing = [k for k in regs if k not in ids]
    errored_ids = {ident(r['key']) for r in errors}
    need = ('key', 'M', 'K', 'K_reg', 'nmse', 'nmse_reg', 'cv_err', 'point_caps', 'point_refinements', 'region_caps',
            'region_refinements', 'seconds')
    bad_fields = [ident(r['key']) for r in ok if any(f not in r for f in need) or r.get('fitting') != 'revised'
                  or len(r['nmse']) != len(r['nmse_reg']) or not np.all(np.isfinite(r['nmse'])) or min(r['nmse']) <= 0]
    seen, rows = set(), []
    for r in ok:                                    # first occurrence of a duplicated key is used; duplicates are listed
        k = ident(r['key'])
        if k in seen or k not in regs:
            continue
        seen.add(k); rows.append(r)
    qa = dict(lines=len(raw), error_rows=len(errors), errored_keys_without_good_row=sorted(errored_ids - set(ids)),
              duplicate_keys=dup, duplicates_identical={k: len({json.dumps(r['nmse']) for r in ok if ident(r['key']) == k}) == 1 for k in dup},
              keys_not_registered=unknown, registered_problems=len(regs), expected_problems=EXPECTED[study],
              missing_problems=len(missing), missing_examples=missing[:10], bad_field_rows=bad_fields)
    return rows, qa


def self_check(study, rows, regs):
    """Stored nmse_reg and K_reg against the registered rows (exact equality is expected: they were copied)."""
    rel = [relmax(r['nmse_reg'], regs[ident(r['key'])]['nmse'][P]) for r in rows]
    kmis = [ident(r['key']) for r in rows if r['K_reg'] != regs[ident(r['key'])]['info']['j_K']['cv']]
    mmis = [ident(r['key']) for r in rows if r['M'] != regs[ident(r['key'])]['M_mics']]
    exact = sum(r['nmse_reg'] == regs[ident(r['key'])]['nmse'][P] for r in rows)
    return dict(rows=len(rows), nmse_reg_exactly_equal=exact, nmse_reg_max_rel_diff=max(rel) if rel else None,
                K_reg_mismatch=kmis, M_mismatch=mmis, passed=bool(rows and exact == len(rows) and not kmis and not mmis))


def gate(study, regs):
    """PLAN.md gate: registered options reproduce the registered j_cv_ext (rel <= 1e-6) and orders."""
    f = HERE / f'gate-{study}.jsonl'
    if not f.exists():
        return dict(note='PLAN.md defines no gate for this set' if study == 'sim' else 'gate file missing')
    gs = [json.loads(l) for l in open(f)]
    good = [g for g in gs if 'error' not in g]
    rel = [relmax(g['nmse'], regs[ident(g['key'])]['nmse'][P]) for g in good]
    kok = [g['K'] == regs[ident(g['key'])]['info']['j_K']['cv'] for g in good]
    n = 16 if study == 'mesh' else 12
    return dict(file=f.name, problems=len(gs), errors=len(gs) - len(good), expected=n, max_rel_diff=max(rel) if rel else None,
                orders_equal=sum(kok), passed=bool(len(good) == n and max(rel) <= GATE_TOL and all(kok)),
                registered_fitting_caps=dict(point_caps=sum(g['point_caps'] for g in good), point_refinements=sum(g['point_refinements'] for g in good),
                                             region_caps=sum(g['region_caps'] for g in good), region_refinements=sum(g['region_refinements'] for g in good)),
                registered_fitting_seconds_median=float(np.median([g['seconds'] for g in good])) if good else None,
                registered_fitting_seconds=[g['seconds'] for g in good])


def cell_key(study, r):
    return r['cell'] if study == 'sim' else f"{int(r['band'])}Hz-M{r['M']}"


def cell_order(study, cells):
    if study == 'sim':
        return sorted(cells)
    return sorted(cells, key=lambda c: (int(c.split('Hz')[0]), int(c.split('M')[1])))


def classes(rs_by_cell, cells, kind, alpha):
    out = {}
    for c in cells:
        rs = rs_by_cell[c]
        bg = min(AV.GENERIC, key=lambda m: float(np.mean([r['e'][m] for r in rs])))
        o = dict(best_generic=bg)
        for tag, a in (('registered', P), ('revised', REV)):
            m, lo, hi = AV.tint(list(AV.cell_diff(rs, kind, a, bg).values()), alpha)
            o[tag] = dict(mean=m, low=lo, high=hi, cls=AV.classify(m, lo, hi))
        o['changed'] = o['registered']['cls'] != o['revised']['cls']
        out[c] = o
    return out


def dist(v):
    v = np.asarray(v, float)
    return dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()), max=float(v.max()),
                abs_gt_0_1=int(np.sum(np.abs(v) > 0.1)), better_gt_0_1=int(np.sum(v < -0.1)), worse_gt_0_1=int(np.sum(v > 0.1)))


def orders(rs):
    Kr, Kn = [r['K_reg'] for r in rs], [r['K'] for r in rs]
    tr = Counter((a, b) for a, b in zip(Kr, Kn))
    n = len(rs)
    return dict(registered_hist=np.bincount(Kr, minlength=6).tolist(), revised_hist=np.bincount(Kn, minlength=6).tolist(),
                changed=sum(a != b for a, b in zip(Kr, Kn)), up=sum(b > a for a, b in zip(Kr, Kn)), down=sum(b < a for a, b in zip(Kr, Kn)),
                changed_fraction=sum(a != b for a, b in zip(Kr, Kn)) / n,
                at_cap_registered=sum(k == 5 for k in Kr), at_cap_revised=sum(k == 5 for k in Kn),
                at_cap_fraction_registered=sum(k == 5 for k in Kr) / n, at_cap_fraction_revised=sum(k == 5 for k in Kn) / n,
                transitions={f'{a}->{b}': v for (a, b), v in sorted(tr.items())})


def caps(rs):
    pc, pn = sum(r['point_caps'] for r in rs), sum(r['point_refinements'] for r in rs)
    rc, rn = sum(r['region_caps'] for r in rs), sum(r['region_refinements'] for r in rs)
    return dict(point_caps=pc, point_refinements=pn, point_cap_fraction=pc / pn if pn else None,
                region_caps=rc, region_refinements=rn, region_cap_fraction=rc / rn if rn else None,
                problems_with_any_cap=sum((r['point_caps'] + r['region_caps']) > 0 for r in rs))


def reg_seconds(reg, K):
    t = reg['timing']
    return t['j_fit'] + t['j_cv'] + (t.get(f'j_ext_K{K}', 0.0) if K > 0 else 0.0)


def analyse_set(study, label, rs, regs, kind, family_cells_fn):
    """One analysed set: the room grid, one anechoic scenario, or the simulation."""
    by = defaultdict(list)
    for r in rs:
        by[r['cell_key']].append(r)
    cells = cell_order(study, by)
    complete = all(len(by[c]) == len(next(iter(by.values()))) for c in cells)
    fam = family_cells_fn(cells)
    cl = classes(by, cells, kind, AV.ALPHA / len(fam))
    d = [r['e'][REV] - r['e'][P] for r in rs]
    pooled = {'all_cells' if study != 'sim' else 'source_proximal': AV.pooled(rs, kind, fam, REV, P)}
    if study == 'sim':
        pooled['all_cells'] = AV.pooled(rs, kind, cells, REV, P)
    per_cell = {}
    for c in cells:
        u = list(AV.cell_diff(by[c], kind, REV, P).values())
        m, lo, hi = AV.tint(u, AV.ALPHA)
        dd = [r['e'][REV] - r['e'][P] for r in by[c]]
        per_cell[c] = dict(problems=len(by[c]), units=len(u), mean=m, low95=lo, high95=hi, min=float(min(dd)), max=float(max(dd)),
                           orders_changed=sum(r['K'] != r['K_reg'] for r in by[c]))
    groups = defaultdict(list)
    for r in rs:
        groups[r['cell_key'].split('-')[0]].append(r)      # simulation: cell family (A1..A9, FF, base); grids: band
    per_group = {}
    for g, grs in sorted(groups.items()):
        gc = cell_order(study, {r['cell_key'] for r in grs})
        per_group[g] = dict(cells=len(gc), problems=len(grs), pooled95=AV.pooled(grs, kind, gc, REV, P),
                            range=[float(min(r['e'][REV] - r['e'][P] for r in grs)), float(max(r['e'][REV] - r['e'][P] for r in grs))])
    fam_changes = [c for c in fam if cl[c]['changed']]
    other_changes = [c for c in cells if c not in fam and cl[c]['changed']]
    cnt = lambda tag, cs: dict(Counter(cl[c][tag]['cls'] for c in cs))
    moves = sorted(rs, key=lambda r: -abs(r['e'][REV] - r['e'][P]))[:10]
    ts = [r['seconds'] for r in rs]; tr = [reg_seconds(regs[ident(r['key'])], r['K_reg']) for r in rs]
    within = all(p['low'] >= -BOUND and p['high'] <= BOUND for p in pooled.values())
    return dict(
        label=label, problems=len(rs), cells=len(cells), units=len({u for c in cells for u in AV.units(by[c], kind)}),
        complete_balanced=complete, class_family_cells=len(fam), class_alpha=AV.ALPHA / len(fam),
        paired_diff_db=dict(pooled95=pooled, problems=dist(d), per_cell=per_cell, per_group=per_group),
        orders=orders(rs),
        classes=dict(registered=cnt('registered', fam), revised=cnt('revised', fam), changed_in_family=fam_changes,
                     changed_outside_family=other_changes, per_cell=cl),
        convergence_revised=caps(rs),
        seconds=dict(revised_median=float(np.median(ts)), revised_total=float(np.sum(ts)),
                     registered_median=float(np.median(tr)), registered_total=float(np.sum(tr)),
                     ratio_median=float(np.median(np.asarray(ts) / np.asarray(tr))),
                     registered_definition='registered row timing j_fit + j_cv + j_ext_K<registered K> (other machine; indicative only)'),
        largest_moves=[dict(key=r['key'], diff_db=r['e'][REV] - r['e'][P], registered_db=r['e'][P], revised_db=r['e'][REV],
                            K_reg=r['K_reg'], K=r['K']) for r in moves],
        stable=dict(pooled_within_0_1_db=within, no_class_change=not fam_changes and not other_changes,
                    stable=bool(within and not fam_changes and not other_changes)))


def run(study):
    path = HERE / f'revfit-{study}.jsonl'
    if not path.exists():
        return None
    regs = registered(study)
    rows, qa = read_rows(path, study, regs)
    out = dict(study=study, registered_study=REG[study], inputs={path.name: sha(path), f'studies/{REG[study]}/primary/rows.jsonl':
               sha(A / 'studies' / REG[study] / 'primary' / 'rows.jsonl')}, quality=qa, self_check=self_check(study, rows, regs),
               gate=gate(study, regs), complete=qa['missing_problems'] == 0 and len(rows) == EXPECTED[study])
    for r in rows:
        rg = regs[ident(r['key'])]
        r.update(e=dict(rg['e']), cell_key=cell_key(study, rg))
        r['e'][REV] = AV.db(r['nmse'])
        for f in ('band', 'source', 'draw', 'seed', 'scenario', 'cell'):
            if f in rg:
                r[f] = rg[f]
    gf = HERE / f'gate-{study}.jsonl'
    if gf.exists():                                 # same-problem, same-machine time only if both ran on one machine
        rev = {ident(r['key']): r for r in rows}
        gk = [ident(json.loads(l)['key']) for l in open(gf)]
        out['gate']['revised_seconds_same_problems'] = [rev[k]['seconds'] if k in rev else None for k in gk]
        out['gate']['revised_caps_same_problems'] = caps([rev[k] for k in gk if k in rev]) if any(k in rev for k in gk) else None
    kind = 'sim' if study == 'sim' else 'mesh'
    if study == 'miracle':
        out['sets'] = {sc: analyse_set(study, sc, [r for r in rows if r['scenario'] == sc], regs, kind, lambda cs: cs)
                       for sc in ('A1', 'R2') if any(r['scenario'] == sc for r in rows)}
    elif study == 'sim':
        out['sets'] = {'sim': analyse_set(study, 'sim', rows, regs, kind, lambda cs: [c for c in cs if c != AV.FAR])}
    else:
        out['sets'] = {'mesh': analyse_set(study, 'mesh', rows, regs, kind, lambda cs: cs)}
    if rows:
        out['orders_all'] = orders(rows)
        out['convergence_revised_all'] = caps(rows)
    out['registered_classes_match_registered_summary'] = check_registered_classes(study, out['sets'])
    out['verdict'] = verdict(out)
    return out


def check_registered_classes(study, sets):
    """Recomputed registered classes against the registered summary.json (a check of the machinery reuse)."""
    s = json.loads((A / 'studies' / REG[study] / 'primary' / 'summary.json').read_text())
    mism = []
    for name, st in sets.items():
        table = s['cells'] if study != 'miracle' else s[name]['cells_table']
        for c, o in st['classes']['per_cell'].items():
            if study == 'sim' and c == AV.FAR:
                continue
            ref = table[c]
            if ref['best_generic'] != o['best_generic'] or ref['comparisons'][f"{P}-vs-{o['best_generic']}"]['cls'] != o['registered']['cls']:
                mism.append(f'{name}:{c}')
    return dict(mismatches=mism, passed=not mism)


def verdict(out):
    if not out['complete']:
        return 'incomplete: not all registered problems have a revised row'
    if not (out['self_check']['passed'] and out['registered_classes_match_registered_summary']['passed']
            and out['gate'].get('passed', True)):
        return 'checks failed: see self_check, gate and registered_classes_match_registered_summary'
    bad = [n for n, s in out['sets'].items() if not s['stable']['stable']]
    return 'stable' if not bad else 'not stable: ' + ', '.join(bad)


def fmt(p):
    return f"{p['mean']:+.4f} [{p['low']:+.4f}, {p['high']:+.4f}]"


def main():
    res = {}
    for study in ('mesh', 'miracle', 'sim'):
        r = run(study)
        if r is None:
            print(f'{study}: no rows file, pending')
            continue
        (HERE / f'summary-{study}.json').write_text(json.dumps(r, indent=1, default=float))
        res[study] = r
        q = r['quality']
        print(f"\n{study}: {sum(s['problems'] for s in r['sets'].values())}/{q['expected_problems']} problems; "
              f"error rows {q['error_rows']}, duplicates {len(q['duplicate_keys'])}, unregistered keys {len(q['keys_not_registered'])}, "
              f"bad rows {len(q['bad_field_rows'])}; self-check {'PASS' if r['self_check']['passed'] else 'FAIL'}; "
              f"gate {r['gate'].get('passed', r['gate'].get('note'))}; registered classes reproduced "
              f"{'PASS' if r['registered_classes_match_registered_summary']['passed'] else 'FAIL'}")
        for n, s in r['sets'].items():
            o, c, k = s['orders'], s['classes'], s['convergence_revised']
            for pn, p in s['paired_diff_db']['pooled95'].items():
                print(f"  {n:5s} pooled ({pn}) revised - registered dB: {fmt(p)}  units {p['units']}")
            pd = s['paired_diff_db']['problems']
            print(f"  {n:5s} problems: range [{pd['min']:+.4f}, {pd['max']:+.4f}], |d|>0.1 dB {pd['abs_gt_0_1']} "
                  f"(better {pd['better_gt_0_1']}, worse {pd['worse_gt_0_1']})")
            print(f"  {n:5s} order hist reg {o['registered_hist']} rev {o['revised_hist']}; changed {o['changed']} "
                  f"({o['changed_fraction']:.4f}; up {o['up']}, down {o['down']}); at cap {o['at_cap_registered']} -> {o['at_cap_revised']}")
            print(f"  {n:5s} classes reg {c['registered']} rev {c['revised']}; changed {c['changed_in_family'] + c['changed_outside_family']}")
            print(f"  {n:5s} caps point {k['point_caps']}/{k['point_refinements']}, region {k['region_caps']}/{k['region_refinements']}; "
                  f"seconds median rev {s['seconds']['revised_median']} vs reg {s['seconds']['registered_median']:.2f}")
            print(f"  {n:5s} stable under PLAN.md: {s['stable']['stable']}")
            print(f"  {'cell':16s} {'mean':>9s} {'low95':>9s} {'high95':>9s} {'min':>8s} {'max':>8s} Kchg  class reg -> rev")
            for cc, v in s['paired_diff_db']['per_cell'].items():
                cl = c['per_cell'][cc]
                print(f"  {cc:16s} {v['mean']:+9.4f} {v['low95']:+9.4f} {v['high95']:+9.4f} {v['min']:+8.3f} {v['max']:+8.3f} {v['orders_changed']:4d}  "
                      f"{cl['registered']['cls']} -> {cl['revised']['cls']}{'  CHANGED' if cl['changed'] else ''}")
        print(f'  verdict: {r["verdict"]}')
    allsets = {f'{st}:{n}': s for st, r in res.items() for n, s in r['sets'].items()}
    overall = dict(studies_present=list(res), studies_pending=[s for s in ('mesh', 'miracle', 'sim') if s not in res],
                   per_set={k: dict(problems=s['problems'], pooled95=s['paired_diff_db']['pooled95'], classes_changed=
                                    s['classes']['changed_in_family'] + s['classes']['changed_outside_family'],
                                    orders_changed=s['orders']['changed'], stable=s['stable']['stable']) for k, s in allsets.items()},
                   problems=sum(s['problems'] for s in allsets.values()),
                   orders_changed=sum(s['orders']['changed'] for s in allsets.values()),
                   classes_changed=sum(len(s['classes']['changed_in_family'] + s['classes']['changed_outside_family']) for s in allsets.values()),
                   verdicts={k: r['verdict'] for k, r in res.items()},
                   verdict=('stable in every study' if len(res) == 3 and all(r['verdict'] == 'stable' for r in res.values())
                            else 'pending' if all(r['verdict'] == 'stable' for r in res.values()) else 'not stable or checks failed in: ' +
                            ', '.join(k for k, r in res.items() if r['verdict'] != 'stable')))
    (HERE / 'summary-overall.json').write_text(json.dumps(overall, indent=1, default=float))
    print('\noverall:', overall['verdict'])


if __name__ == '__main__':
    main()
