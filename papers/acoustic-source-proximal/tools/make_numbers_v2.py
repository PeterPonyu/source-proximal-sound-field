"""Numbers and generated tables of the version-2 manuscript, read from the committed registered outputs.

python3 -B tools/make_numbers_v2.py ../../archive/2026-11-05-taslp/acoustic generated

Every interval, class and rule outcome is read from a committed summary (the registered analyses and the labelled
post hoc analyses committed beside them); nothing statistical is recomputed here. The only aggregations are counts of
per-cell classes and selected orders read from the summaries, per-problem kernel shifts and stop-rule identity read
from the committed rows, and medians of recorded process times for the runtime table. Each summary's rows are checked
against the SHA-256 in their run receipt first.

Writes generated/v2-numbers.tex (\\vdef{key}{value}; the manuscript prints them with \\vn{key}) and the generated
tables generated/v2-*.tex.
"""
from __future__ import annotations
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
import numpy as np

S = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2]).resolve()
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(S / 'common'))
import sim  # noqa: E402

V = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(study, sub='primary', name='summary.json', receipt=True, summary_receipted=False):
    d = S / 'studies' / study / sub
    if receipt:
        rc = json.loads((d / 'run-receipt.json').read_text())
        if sha(d / 'rows.jsonl') != rc['outputs']['rows.jsonl']:
            raise SystemExit(f'rows of {d} do not match their receipt')
        # only some receipts record their summary (the MIRACLE study's does); check it wherever it is recorded,
        # and require it where the run is known to record it, so a receipt without it cannot skip the check
        if summary_receipted and name not in rc['outputs']:
            raise SystemExit(f'the receipt of {d} does not record {name}')
        if name in rc['outputs'] and sha(d / name) != rc['outputs'][name]:
            raise SystemExit(f'{name} of {d} does not match its receipt')
    return json.loads((d / name).read_text())


def rows(study, sub='primary'):
    return [json.loads(l) for l in open(S / 'studies' / study / sub / 'rows.jsonl')]


# ------------------------------------------------------------------ formatting

def num(x, d=2, sign=True):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return r'undefined'
    s = f'{x:+.{d}f}' if sign else f'{x:.{d}f}'
    if s.lstrip('+-').strip('0.') == '':        # avoid "-0.00"
        s = ('+' if sign else '') + s.lstrip('+-')
    return '$' + s.replace('-', '-') + '$'


def iv(d, digits=2, unit=True):
    m, lo, hi = d['mean'], d['low'], d['high']
    if any(v is None or not math.isfinite(v) for v in (m, lo, hi)):
        return 'undefined'
    return f"{num(m, digits)}{'~dB' if unit else ''} [{num(lo, digits)}, {num(hi, digits)}]"


def put(key, val):
    if key in V:
        raise SystemExit(f'duplicate key {key}')
    V[key] = val


def cnt(n):
    return str(int(n))


def wtl(d):
    return f"{d['wins']}/{d['ties']}/{d['losses']}"


WORDS = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
         10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen'}

# ------------------------------------------------------------------ cell labels

CELLS = sim.cells()
FAR = 'A1-beside1.2'


def sim_label(c):
    v = CELLS[c]
    if c == 'base':
        return 'Base'
    fam, rest = c.split('-', 1)
    if fam == 'A1':
        return 'Inside the region' if rest == 'in' else f"{rest[:-3].capitalize()}, {rest[-3:]} m" + (' (far control)' if c == FAR else '')
    if fam == 'A2':
        return f"{int(v['f'])} Hz, {v['M']} mics"
    if fam == 'A3':
        h = rest.split('-h')[1]
        hs = '0.4 m' if h == 'base' else f'{h}$\\lambda$'
        return f"Floor phase {int(v['floor_phase'])}$^\\circ$, height {hs}"
    if fam == 'A4':
        return 'Monopole source' if v['beta'] == 0.0 else f"Strong directivity ($\\beta={v['beta']}$)"
    if fam == 'A5':
        return f"Coherent driver {v['delta']} m above"
    if fam == 'A6':
        return f"Second source, amplitude {v['second']}"
    if fam == 'A7':
        return f"Mic.\\ position error {v['jitter'] * 1000:g} mm"
    if fam == 'A8':
        return f"Speed error {v['speed'] * 100:g}\\%"
    if fam == 'A9':
        return f"SNR {int(v['snr'])} dB"
    if fam == 'FF':
        return f'Factorial run {int(rest)}'
    return c


def meas_label(c):
    f, m = c.split('Hz-M')
    return f'{int(f)} Hz, {m} mics'


def meas_cells():
    return [f'{f}Hz-M{m}' for f in (125, 250, 500, 1000) for m in (24, 48, 96, 192)]


def list_words(items):
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ', '.join(items[:-1]) + ' and ' + items[-1]


def cells_phrase(cs):
    """'125 Hz with 24, 96 and 192 microphones' from MeshRIR cell names (grouped by band)."""
    by = {}
    for c in cs:
        f, m = c.split('Hz-M')
        by.setdefault(int(f), []).append(m)
    items = sorted(by.items())
    if len(items) == 1:
        return f"{items[0][0]}~Hz with {list_words(items[0][1])} microphones"
    # several bands: '500~Hz (24, 48 and 192 microphones) and 1000~Hz (192 microphones)'
    return list_words([f"{f}~Hz ({list_words(ms)} microphones)" for f, ms in items])


GEN_NAMES = {'sinc_ev': 'sinc kernel', 'rbf': 'squared-exponential kernel', 'adk_standin': 'adaptive kernel',
             'sbl_em_common': 'sparse Bayesian (EM)', 'sbl_mackay_common': 'sparse Bayesian (fixed point)',
             'sbl_em_guard': 'guarded sparse Bayesian (EM)', 'sbl_mackay_guard': 'guarded sparse Bayesian (fixed point)'}

# ------------------------------------------------------------------ simulation (registered simulation study)

s2 = load('TB-S2')
R = s2['rules']; C = s2['cells']
prox = [c for c in CELLS if c != FAR]
put('sim-problems', f"{s2['n_rows']:,}")
put('sim-cells', cnt(len(CELLS))); put('sim-prox', cnt(len(prox))); put('sim-fields', '32')
put('sim-errors', cnt(s2['n_errors']))
put('sim-S1', wtl(R['S1'])); put('sim-S1-wins', cnt(R['S1']['wins']))
bg = Counter(C[c]['best_generic'] for c in prox)
for k, v in bg.items():
    put(f'sim-bg-{k}', cnt(v))
for k in GEN_NAMES:
    if k in ('sbl_em_guard', 'sbl_mackay_guard'):
        continue
    V.setdefault(f'sim-bg-{k}', '0')
g = R['S2']['per_competitor']['gsmf']; m5 = R['S2']['per_competitor']['mo26_cap5']
put('sim-S2-gsmf', wtl(g)); put('sim-S2-gsmf-wins', cnt(g['wins'])); put('sim-S2-gsmf-ties', cnt(g['ties']))
put('sim-S2-gsmf-losses', cnt(g['losses'])); put('sim-S2-gsmf-ni', cnt(g['non_inferior']))
put('sim-S2-gsmf-notni', cnt(len(prox) - g['non_inferior']))
put('sim-S2-gsmf-pooled', iv(g['pooled']))
put('sim-S2-mo26', wtl(m5)); put('sim-S2-mo26-wins', cnt(m5['wins'])); put('sim-S2-mo26-pooled', iv(m5['pooled']))
fail = [c for c in prox if not C[c]['comparisons']['j_cv_ext-vs-gsmf']['non_inferior']]
loss = [c for c in prox if C[c]['comparisons']['j_cv_ext-vs-gsmf']['cls'] == 'loss']
put('sim-S2-loss-cell', ', '.join(sim_label(c).replace(' Hz, ', ' Hz with ').replace(' mics', ' microphones') for c in loss)); put('sim-S2-loss-iv', ', '.join(iv(C[c]['comparisons']['j_cv_ext-vs-gsmf']) for c in loss))
put('sim-far-gsmf', iv(C[FAR]['comparisons']['j_cv_ext-vs-gsmf']))
s3 = R['S3']['per_cell']
put('sim-S3-a06', iv(s3['A6-second0.6'])); put('sim-S3-a10', iv(s3['A6-second1.0']))
for k in ('S4', 'S5', 'S7'):
    put(f'sim-{k}', wtl(R[k])); put(f'sim-{k}-wins', cnt(R[k]['wins'])); put(f'sim-{k}-pooled', iv(R[k]['pooled']))
put('sim-S4-ties', cnt(R['S4']['ties']))
v1m = [C[c]['comparisons']['j_cv_ext-vs-v1']['mean'] for c in prox]
put('sim-S7-range', f"{num(max(v1m))} to {num(min(v1m))}")


def cell_range(cells, comp, key_of):
    """Range of the per-condition mean differences, from the registered per-condition summaries."""
    ms = [cells[c]['comparisons'][key_of(c)]['mean'] for c in comp]
    lo_, hi_ = min(ms), max(ms)
    return f"{num(hi_)} to {num(lo_)}" if hi_ < 0 else f"{num(lo_)} to {num(hi_)}"


put('sim-S1-range', cell_range(C, prox, lambda c: f"j_cv_ext-vs-{C[c]['best_generic']}"))
put('sim-S2-gsmf-range', cell_range(C, prox, lambda c: 'j_cv_ext-vs-gsmf'))
put('sim-S2-mo26-range', cell_range(C, prox, lambda c: 'j_cv_ext-vs-mo26_cap5'))
put('sim-S6', wtl(R['S6'])); put('sim-S6-wins', cnt(R['S6']['wins']))
D = R['descriptive']
put('sim-sw', wtl(D['schwarz_vs_cv'])); put('sim-hybrid', wtl(D['hybrid'])); put('sim-hybrid-pooled', iv(D['pooled_vs_hybrid']))


def classes(cells, comp, key):
    cl = Counter(cells[c]['comparisons'][key]['cls'] for c in comp if key in cells[c]['comparisons'])
    return f"{cl.get('win', 0)}/{cl.get('tie', 0)}/{cl.get('loss', 0)}", cl


for ref in ('ref_point', 'ref_region'):
    t, cl = classes(C, prox, f'j_cv_ext-vs-{ref}')
    put(f'sim-{ref}', t)
kh = np.sum([C[c]['K_cv_hist'] for c in prox], axis=0)
for K in range(6):
    put(f'sim-K{K}', f'{int(kh[K]):,}')
put('sim-K-total', f'{int(kh.sum()):,}')
put('sim-cap-pct', f'{100 * kh[5] / kh.sum():.0f}')
capf = [100 * C[c]['cap_bound_fraction'] for c in prox]
put('sim-cap-cell-range', f'{min(capf):.0f} to {max(capf):.0f}')
put('sim-cap-median', f"{100 * D['cap_bound_fraction_median']:.0f}")

# ------------------------------------------------------------------ measured (registered measured study and its follow-ups)

m2 = load('TB-M2'); RM = m2['rules']; MC = m2['cells']
MCELLS = meas_cells()
put('meas-problems', f"{sum(MC[c]['n'] for c in MCELLS):,}"); put('meas-cells', cnt(len(MCELLS))); put('meas-spk', '32')
put('meas-M1', wtl(RM['M1'])); put('meas-M1-wins', cnt(RM['M1']['wins']))
ties = [c for c in MCELLS if MC[c]['comparisons'][f"j_cv_ext-vs-{MC[c]['best_generic']}"]['cls'] == 'tie']
put('meas-M1-tie-cells', cells_phrase(ties))
put('meas-M1-range', cell_range(MC, MCELLS, lambda c: f"j_cv_ext-vs-{MC[c]['best_generic']}"))
put('meas-M2-gsmf-range', cell_range(MC, MCELLS, lambda c: 'j_cv_ext-vs-gsmf'))
_m125 = [abs(MC[c]['comparisons']['j_cv_ext-vs-' + MC[c]['best_generic']]['mean']) for c in MCELLS if c.startswith('125')]
put('meas-M1-125-max', f'{max(_m125):.2f}')
DM = RM['descriptive']
put('meas-M1-pooled-task', iv(DM['pooled_vs_best_generic_task']))
sbs = json.loads((S / 'studies/TB-M2-SBLFIX/m1-side-by-side.json').read_text())
reg_k, cor_k = 'M1 (registered generic set)', 'M1-corrected'
put('meas-M1-pooled-cell', iv(sbs[reg_k]['pooled_vs_per_cell_best']))
fix = load('TB-M2-SBLFIX')
put('meas-M1c', wtl(fix['M1_corrected'])); put('meas-M1c-pooled-task', iv(fix['pooled_vs_task_best_generic']))
put('meas-M1c-pooled-cell', iv(sbs[cor_k]['pooled_vs_per_cell_best']))
bgr = Counter(MC[c]['best_generic'] for c in MCELLS); bgc = Counter(fix['M1_corrected']['best_generic'].values())
put('meas-bg-reg', list_words([f"the {GEN_NAMES[k]} in {WORDS[v]}" for k, v in bgr.most_common()]))
put('meas-bg-cor', list_words([f"the {GEN_NAMES[k]} in {WORDS[v]}" for k, v in bgc.most_common()]))
put('meas-sbl-cor-cells', cnt(sum(v for k, v in bgc.items() if 'sbl' in k)))
sens = json.loads((S / 'studies/TB-M2/sensitivity-nonfinite.json').read_text())
nf = sens['nonfinite_tasks_of_1024']
put('meas-nf-em', cnt(nf['sbl_em_common'])); put('meas-nf-mackay', cnt(nf['sbl_mackay_common'])); put('meas-nf-mo26', cnt(nf['mo26_cap5']))
put('meas-nf-guard', f"{fix['nonfinite_guarded']['sbl_em_guard']} and {fix['nonfinite_guarded']['sbl_mackay_guard']}")
put('meas-nf-guard-phrase', 'no such failures' if all(v == 0 for v in fix['nonfinite_guarded'].values()) else f"{fix['nonfinite_guarded']['sbl_em_guard']} and {fix['nonfinite_guarded']['sbl_mackay_guard']} such failures")
put('meas-em-guard', iv(fix['primary_vs_sbl_em_guard']['pooled'])); put('meas-mackay-guard', iv(fix['primary_vs_sbl_mackay_guard']['pooled']))
pg = RM['M2']['per_competitor']['gsmf']; pm = RM['M2']['per_competitor']['mo26_cap5']
put('meas-M2-gsmf', wtl(pg)); put('meas-M2-gsmf-ni', cnt(pg['non_inferior'])); put('meas-M2-gsmf-pooled', iv(pg['pooled']))
# the undefined condition is counted apart from the ties (post hoc recount, posthoc/review3/saved.json, A5)
R3S = json.loads((S / 'posthoc' / 'review3' / 'saved.json').read_text())
_a5 = R3S['A5_measured_per_bin_variant']
assert (_a5['win'], _a5['tie'] + _a5['undefined'], _a5['loss']) == (pm['wins'], pm['ties'], pm['losses'])
put('meas-M2-mo26', f"{_a5['win']}/{_a5['tie']}/{_a5['loss']}"); put('meas-M2-mo26-undef', WORDS[_a5['undefined']])
put('meas-M2-mo26-ni', cnt(pm['non_inferior']))
sm = sens['mo26_cap5_without_nonfinite']
put('meas-M2s-mo26', wtl(sm)); put('meas-M2s-mo26-pooled', iv(sm['pooled']))
put('meas-M3', wtl(RM['M3'])); put('meas-M4', wtl(RM['M4']))
sinc_ties = [c for c in MCELLS if MC[c]['comparisons']['j_cv_ext-vs-sinc_ev']['cls'] == 'tie']
put('meas-M3-tie-cells', cells_phrase(sinc_ties))
put('meas-M3-maxmean', num(max(MC[c]['comparisons']['j_cv_ext-vs-sinc_ev']['mean'] for c in MCELLS)))
for k, nm in (('extension', 'ext'), ('shared_coordinates', 'shared'), ('schwarz_vs_cv', 'sw'), ('v1', 'v1'), ('hybrid', 'hybrid'),
              ('two_stage_point', 'tsp')):
    put(f'meas-{nm}', wtl(DM[k]))
put('meas-v1-pooled', iv(DM['pooled_vs_v1']))
tsl = [c for c in MCELLS if MC[c]['comparisons']['j_cv_ext-vs-two_stage_point']['cls'] == 'loss']
put('meas-tsp-loss-cells', cells_phrase(tsl))
put('meas-tsp-loss-iv', list_words([iv(MC[c]['comparisons']['j_cv_ext-vs-two_stage_point']) for c in tsl]))
tsw = [c for c in MCELLS if MC[c]['comparisons']['j_cv_ext-vs-two_stage_point']['cls'] == 'win']
put('meas-tsp-win-cells', cells_phrase(tsw))
t, _ = classes(MC, MCELLS, 'j_cv_ext-vs-two_stage_region'); put('meas-tsr', t)
for ref in ('ref_point', 'ref_region'):
    t, _ = classes(MC, MCELLS, f'j_cv_ext-vs-{ref}'); put(f'meas-{ref}', t)
HO = RM['held_out_same_recordings']
hb = Counter(HO['vs_best_generic'].values())
put('meas-held-bg', f"{hb.get('win', 0)}/{hb.get('tie', 0)}/{hb.get('loss', 0)}")
put('meas-held-gsmf', wtl(HO['vs_gsmf'])); put('meas-held-mo26', wtl(HO['vs_mo26_cap5']))
mk = np.sum([MC[c]['K_cv_hist'] for c in MCELLS], axis=0)
for K in range(6):
    put(f'meas-K{K}', f'{int(mk[K]):,}')
put('meas-cap-pct', f'{100 * mk[5] / mk.sum():.0f}')
mcap = [100 * MC[c]['cap_bound_fraction'] for c in MCELLS]
put('meas-cap-cell-range', f'{min(mcap):.0f} to {max(mcap):.0f}')
put('meas-K12-pct', f'{100 * (mk[1] + mk[2]) / mk.sum():.0f}')
put('meas-stop-losses', cnt(RM['stop_rule']['cell_losses']))

# ------------------------------------------------------------------ the closest competitors: exact kernel, faithful algorithms, official code

ex = load('TB-M2-EXACT'); exr = ex['rules']
for comp, nm in (('gsmf', 'gsmf'), ('mo26_cap5', 'mo26')):
    r = exr[comp]['as_registered']; f = exr[comp]['finite_only']
    put(f'ex-{nm}-reg', '/'.join(str(r['registered'][k]) for k in ('wins', 'ties', 'losses')))
    put(f'ex-{nm}-ex', '/'.join(str(r['exact'][k]) for k in ('wins', 'ties', 'losses')))
    put(f'ex-{nm}-pooled-reg', iv(r['pooled_registered'])); put(f'ex-{nm}-pooled-ex', iv(r['pooled_exact']))
    put(f'ex-{nm}-pooled-reg-fin', iv(f['pooled_registered']))
    put(f'ex-{nm}-robust', 'robust' if r['robust_to_kernel'] and f['robust_to_kernel'] else 'not robust')
pe = ex['pooled']
put('ex-effect-gsmf', iv(pe['gsmf_exact-vs-gsmf'])); put('ex-effect-mo26', iv(pe['mo26_cap5_exact-vs-mo26_cap5']['finite_only']))
put('ex-twins', f"{ex['cubature_reproduces_registered']['gsmf']}")
put('ex-twins-mo26', f"{ex['cubature_reproduces_registered']['mo26_cap5']}")
put('ex-kmaxR-gsmf-med', f"{ex['k_max_R']['cubature']['gsmf']['median']:.2f}"); put('ex-kmaxR-mo26-med', f"{ex['k_max_R']['cubature']['mo26_cap5']['median']:.2f}")
erow = rows('TB-M2-EXACT')


def db(v):
    v = np.asarray(v, float)
    return 10 * math.log10(float(np.mean(v))) if np.all(np.isfinite(v)) and np.mean(v) > 0 else float('nan')


for comp, nm in (('gsmf', 'gsmf'), ('mo26_cap5', 'mo26')):
    ch = np.array([db(r['nmse'][f'{comp}_exact']) - db(r['nmse'][f'{comp}_cubature']) for r in erow])
    ch = ch[np.isfinite(ch)]
    put(f'ex-{nm}-shift-range', f'{num(ch.min())} to {num(ch.max())}')
    put(f'ex-{nm}-shift-gt1', f'{int((abs(ch) > 1).sum())} of {len(ch)}')
hi = load('TB-M2-EXACT-HI')
for part, nm in (('part_a', 'A'), ('part_b', 'B')):
    r = hi[part]['rule_kx_gsmf']['as_registered']
    put(f'hi-{nm}-reg', '/'.join(str(r['registered'][k]) for k in ('win', 'tie', 'loss')))
    put(f'hi-{nm}-ex', '/'.join(str(r['exact'][k]) for k in ('win', 'tie', 'loss')))
    put(f'hi-{nm}-ni', r['non_inferior_exact'].replace('/', ' of '))
    put(f'hi-{nm}-pooled-reg', iv(r['pooled_registered'])); put(f'hi-{nm}-pooled-ex', iv(r['pooled_exact']))
put('hi-head-ex', iv(hi['part_b']['headline']['exact'])); put('hi-head-reg', iv(hi['part_b']['headline']['registered']))
put('hi-effect16', iv(hi['part_b']['pooled']['gsmf_exact-vs-gsmf'])); put('hi-effectA', iv(hi['part_a']['pooled']['gsmf_exact-vs-gsmf'], 3))
put('hi-twins', hi['cubature_reproduces_registered']['hi'])
put('hi-m2', 'passes' if hi['part_b']['m2_gsmf_part_with_exact_kernel']['as_registered']['passed'] else 'fails')
hrow = rows('TB-M2-EXACT-HI')
for M in (96, 192):
    ch = np.array([db(r['nmse']['gsmf_exact']) - db(r['nmse']['gsmf_cubature']) for r in hrow if r['M'] == M])
    put(f'hi-shift-range-{M}', f'{num(ch.min())} to {num(ch.max())}')
pa = load('TB-PROX', 'a1-analysis', receipt=False)['arms']['MeshRIR']
P = pa['pooled']
put('px-PX1', iv(P['j_cv_ext-vs-mo26_prox'])); put('px-PX2', iv(P['j_cv_ext-vs-gsmf_prox']))
put('px-mo26-reg', iv(P['mo26_prox-vs-mo26_cap5'])); put('px-gsmf-reg', iv(P['gsmf_prox-vs-gsmf']))
put('px-PX1-mean', num(-P['j_cv_ext-vs-mo26_prox']['mean'], 2, sign=False)); put('px-PX2-mean', num(-P['j_cv_ext-vs-gsmf_prox']['mean'], 2, sign=False))
dsn = json.loads((S / 'studies/TB-PROX/descriptive-sinc.json').read_text())
put('px-mo26-sinc', iv(dsn['pooled']['mo26_prox-vs-sinc_ev'])); put('px-gsmf-sinc', iv(dsn['pooled']['gsmf_prox-vs-sinc_ev']))
put('px-gsmf-sinc-1k', iv(dsn['bands']['1000.0']['gsmf_prox-vs-sinc_ev']))
put('px-mo26cap-sinc', iv(dsn['pooled']['mo26_cap5-vs-sinc_ev'])); put('px-gsmfreg-sinc', iv(dsn['pooled']['gsmf-vs-sinc_ev']))
put('px-j-sinc', iv(dsn['pooled']['j_cv_ext-vs-sinc_ev']))
ps = load('TB-PROX-S2')['arms']['MeshRIR']
put('ps-PXS', iv(ps['pooled']['j_cv_ext-vs-mo26_prox_s2'])); put('ps-dev', iv(ps['pooled']['mo26_prox_s2-vs-mo26_prox'], 4))
key = lambda r: (r['source'], r['band'], r['M'], r['draw'], r['seed'])
s2r = {key(r): r for r in rows('TB-PROX-S2') if 'error' not in r}
pxr = {key(r): r for r in rows('TB-PROX') if 'error' not in r and r['variant'] == 'mo26'}
same = tot = 0
for k in s2r:
    for x, y in zip(s2r[k]['nmse']['mo26_prox_s2'], pxr[k]['nmse']['mo26_prox']):
        tot += 1; same += x == y
put('ps-identical', f'{same} of {tot}')
rib = load('TB-M2-RIB'); dpb = json.loads((S / 'studies/TB-M2-RIB/descriptive-perbin.json').read_text())
put('rib-R1', iv(rib['R1']['pooled'])); put('rib-R1-cls', wtl(rib['R1']))
put('rib-standin', iv(rib['R2_standin_vs_official']['pooled'])); put('rib-off-sinc', iv(rib['official_vs_sinc']['pooled']))
put('rib-perbin', iv(dpb['ind_cv_ext-vs-ribeiro_official']['pooled'])); put('rib-perbin-cls', wtl(dpb['ind_cv_ext-vs-ribeiro_official']))
put('rib-shared', iv(dpb['j_cv_ext-vs-ind_cv_ext']['pooled']))
put('rib-train-med', f"{rib['fit_seconds']['median']:.0f}"); put('rib-train-max', f"{rib['fit_seconds']['max']:.0f}")

# ------------------------------------------------------------------ the six failing simulated conditions, and per-band ranges of the published algorithms
FF_HIGH = (('beta', 0.45, 'strong directivity'), ('delta', 0.2, 'a coherent driver'), ('second', 0.6, 'a second source'),
           ('jitter', 0.01, '10~mm position errors'), ('speed', 0.02, 'a 2\\% speed error'), ('snr', 20.0, '20~dB SNR'),
           ('floor_phase', 180.0, 'an inverting floor'), ('dist', 0.6, 'the loudspeaker 0.6~m below the region'))


def _ff_desc(c):
    v = CELLS[c]
    hi = [txt for k, val, txt in FF_HIGH if float(v[k] if not isinstance(v[k], str) else -1) == val]
    return 'factorial run ' + str(int(c.split('-')[1])) + (', with ' + list_words(hi) if hi else '')


def _inline(c):
    if c.startswith('FF-'):
        return _ff_desc(c)
    lab = sim_label(c)
    lab = lab.replace(' Hz, ', ' Hz with ').replace(' mics', ' microphones').replace(', height', ' at height')
    return lab[0].lower() + lab[1:] if not lab[0].isdigit() else lab


_items = [f"{_inline(c)} (${C[c]['comparisons']['j_cv_ext-vs-gsmf']['mean']:+.2f}$~dB)" for c in fail]
put('sim-S2-fail-list', '; '.join(_items[:-1]) + '; and ' + _items[-1])
_names = [_inline(c) for c in fail]
put('sim-S2-fail-names', '; '.join(_names[:-1]) + '; and ' + _names[-1])
bands_px = [pa['cells'][str((float(b_), 24))] for b_ in ('125.0', '250.0', '500.0', '1000.0')]
for pair, key in (('j_cv_ext-vs-mo26_prox', 'px-band-j-mo26'), ('j_cv_ext-vs-gsmf_prox', 'px-band-j-gsmf'),
                  ('mo26_prox-vs-mo26_cap5', 'px-band-mo26-reg'), ('gsmf_prox-vs-gsmf', 'px-band-gsmf-reg')):
    ms = sorted(d[pair]['mean'] for d in bands_px)
    lo_, hi_ = (ms[-1], ms[0]) if ms[0] < 0 else (ms[0], ms[-1])
    put(key, f"{num(lo_)} to {num(hi_)}")
    put(key + '-cls', ', '.join(sorted({d[pair]['cls'] for d in bands_px})))

# ------------------------------------------------------------------ magnitudes for the abstract (rounded from the same values)
put('abs-meas', f"{-fix['pooled_vs_task_best_generic']['mean']:.2f}")
put('abs-gsmf', f"{-hi['part_b']['headline']['exact']['mean']:.2f}")
put('abs-rib', f"{-rib['R1']['pooled']['mean']:.2f}")
put('abs-px1', f"{-P['j_cv_ext-vs-mo26_prox']['mean']:.1f}")
put('abs-px2', f"{-P['j_cv_ext-vs-gsmf_prox']['mean']:.1f}")
put('abs-sim-gsmf', f"{-g['pooled']['mean']:.2f}")

# ------------------------------------------------------------------ generated tables

def tex_iv(d, digits=2):
    if d is None or any(v is None or not math.isfinite(v) for v in (d['mean'], d['low'], d['high'])):
        return 'undefined'
    return f"{num(d['mean'], digits)} [{num(d['low'], digits)}, {num(d['high'], digits)}]"


# (0) the second measured study (studies/TB-M3-MIRACLE): registered summary, read before the outcomes table
M3 = load('TB-M3-MIRACLE', summary_receipted=True)
sys.path.insert(0, str(S / 'studies' / 'TB-M3-MIRACLE'))
import coverage_check  # noqa: E402
# fresh check of the consumed rows (the stored record must also match it); covers every method in the rows
_cov = coverage_check.check(S / 'studies' / 'TB-M3-MIRACLE' / 'primary')
assert _cov['complete'], 'the MIRACLE run does not cover the registered task set'
assert _cov['rows_sha256'] == json.loads((S / 'studies' / 'TB-M3-MIRACLE' / 'primary' / 'run-receipt.json').read_text())['outputs']['rows.jsonl']
assert set(coverage_check.USED) <= set(_cov['methods_checked']) and not _cov['nonfinite_by_method']
assert _cov == json.loads((S / 'studies' / 'TB-M3-MIRACLE' / 'primary' / 'coverage.json').read_text()), 'stale MIRACLE coverage record'
put('m3-coverage', 'present once each, with the registered seeds and finite errors for every method')
_xc = coverage_check.check(S / 'studies' / 'TB-M3-MIRACLE' / 'xcheck' / 'coverage-test')
assert _xc == json.loads((S / 'studies' / 'TB-M3-MIRACLE' / 'xcheck' / 'coverage-test' / 'coverage.json').read_text()), 'stale cross-check record'
assert not _xc['complete'] and _xc['missing'] + _xc['rows'] == 1536, 'the coverage check must reject the cross-machine sample'
put('m3-xc-rows', cnt(_xc['rows'])); put('m3-xc-missing', f"{_xc['missing']:,}")
assert M3['n_errors'] == 0 and M3['n_rows'] == 1536, (M3['n_errors'], M3['n_rows'])
# analyze_m3.py sizes its Bonferroni family from the conditions it sees; require the registered family
assert all(M3[_sc]['cells'] == 12 and M3[_sc]['units'] == 32 and M3[_sc]['problems'] == 768 for _sc in ('A1', 'R2')), \
    'the MIRACLE summary is not over the registered 12 conditions and 32 sources'
_m3r = M3['rules']
for _sc in ('A1', 'R2'):
    x = M3[_sc]
    assert x['problems'] == 768 and x['cells'] == 12 and x['units'] == 32
    T3 = x['cells_table']
    put(f'm3-{_sc}-bg', wtl(x['best_generic']))
    put(f'm3-{_sc}-bg-wins', cnt(x['best_generic']['wins'])); put(f'm3-{_sc}-bg-losses', WORDS[x['best_generic']['losses']])
    _cls = {c: T3[c]['comparisons'][f"j_cv_ext-vs-{T3[c]['best_generic']}"]['cls'] for c in T3}
    put(f'm3-{_sc}-loss-cells', cells_phrase([c for c, v in _cls.items() if v == 'loss']) or 'none')
    put(f'm3-{_sc}-bg-pooled', iv(x['pooled_vs_best_generic_cell'])); put(f'm3-{_sc}-oracle', iv(x['pooled_vs_best_generic_task']))
    for b in ('250', '500', '1000'):
        put(f'm3-{_sc}-band-{b}', iv(x['by_band'][b]))
    put(f'm3-{_sc}-every', cnt(x['every_member_wins']))
    _nsbl = sum(v.startswith('sbl_') for v in x['best_generic_by_cell'].values())
    put(f'm3-{_sc}-bg-sbl', cnt(_nsbl))
    for k, nm in (('sinc', 'sinc'), ('gsmf', 'gsmf'), ('mo26_cap5', 'mo26'), ('v1', 'v1'), ('two_stage_point', 'tsp'),
                  ('two_stage_region', 'tsr'), ('ref_point', 'refpt'), ('ref_region', 'refrg')):
        put(f'm3-{_sc}-{nm}', wtl(x[k]))
        if 'pooled' in x[k]:
            put(f'm3-{_sc}-{nm}-pooled', iv(x[k]['pooled']))
    _h = x['K_cv_hist']; _n = sum(_h)
    put(f'm3-{_sc}-cap-pct', f'{100 * _h[5] / _n:.0f}'); put(f'm3-{_sc}-K12-pct', f'{100 * (_h[1] + _h[2]) / _n:.0f}')
    put(f'm3-{_sc}-K0', cnt(_h[0]))
    _md = x['median_error_db']
    put(f'm3-{_sc}-med-prop', num(_md['proposed'], 1)); put(f'm3-{_sc}-med-bg', num(_md['best_generic_of_condition'], 1))
    put(f'm3-{_sc}-med-tsp', num(_md['arrival_time_point'], 1))
put('m3-n-failed', WORDS[sum(not v['passed'] for v in _m3r.values())])
assert not _m3r['G1']['passed'] and not _m3r['G4']['passed'], 'the abstract states that both rules against the best generic estimator failed'
_a1, _r2 = M3['A1']['best_generic'], M3['R2']['best_generic']
put('abs-m3-A1-phrase', f"won {WORDS[_a1['wins']]} of 12 conditions and lost {WORDS[_a1['losses']]}")
put('abs-m3-R2-phrase', f"won {WORDS[_r2['wins']]} and lost {WORDS[_r2['losses']]}" if _r2['wins'] != _r2['losses'] else f"won and lost {WORDS[_r2['wins']]} each")
put('abs-m3-A1', f"{-M3['A1']['pooled_vs_best_generic_cell']['mean']:.2f}")
put('abs-m3-R2-phrase2', f"won {WORDS[_r2['wins']]} and lost {WORDS[_r2['losses']]}")
put('m3-stop', 'yes' if M3['stop_rule']['triggered'] else 'no')

# (1) every registered outcome, failures included
rule_rows = [
    ('Simulation', 'Beats the best generic estimator in $\\geq 90\\%$ of source-proximal conditions, no loss',
     'pass' if R['S1']['passed'] else 'fail', f"{wtl(R['S1'])} of {len(prox)}"),
    ('', 'Non-inferior to both five-ball variants in every source-proximal condition, pooled superior to each', 'pass' if R['S2']['passed'] else 'fail',
     f"shared: non-inferior in {g['non_inferior']} of {len(prox)}, {wtl(g)}, pooled {tex_iv(g['pooled'])}; "
     f"per-bin: non-inferior in {m5['non_inferior']} of {len(prox)}, {wtl(m5)}, pooled {tex_iv(m5['pooled'])}"),
    ('', 'Cross-validated order beats one point in both two-source conditions', 'pass' if R['S3']['passed'] else 'fail',
     f"{tex_iv(s3['A6-second0.6'])}; {tex_iv(s3['A6-second1.0'])}"),
    ('', 'Region extension loses in no condition, pooled superior to points', 'pass' if R['S4']['passed'] else 'fail',
     f"{wtl(R['S4'])}; pooled {tex_iv(R['S4']['pooled'])}"),
    ('', 'Shared coordinates win in $\\geq$ half, lose in none, pooled superior', 'pass' if R['S5']['passed'] else 'fail',
     f"{wtl(R['S5'])}; pooled {tex_iv(R['S5']['pooled'])}"),
    ('', 'No loss to the sinc kernel in any condition, far control included', 'pass' if R['S6']['passed'] else 'fail', f"{wtl(R['S6'])} of {len(CELLS)}"),
    ('', 'Pooled superior to the single-frequency estimator', 'pass' if R['S7']['passed'] else 'fail', f"pooled {tex_iv(R['S7']['pooled'])}"),
    ('Room grid', 'Beats the best generic estimator in $\\geq 12$ of 16 conditions, loses in none', 'pass' if RM['M1']['passed'] else 'fail',
     f"{wtl(RM['M1'])}; with corrected sparse Bayesian comparators {wtl(fix['M1_corrected'])}"),
    ('', 'Non-inferior to both five-ball variants in every condition, pooled superior to each', 'pass' if RM['M2']['passed'] else 'fail',
     f"shared: {wtl(pg)}, pooled {tex_iv(pg['pooled'])}; per-bin: {V['meas-M2-mo26']} and {V['meas-M2-mo26-undef']} undefined condition; pooled undefined"),
    ('', 'No loss to the sinc kernel', 'pass' if RM['M3']['passed'] else 'fail', f"{wtl(RM['M3'])}"),
    ('', 'Evidence-chosen remainder kernel beats the best generic estimator in every 125--250~Hz condition', 'pass' if RM['M4']['passed'] else 'fail', f"{wtl(RM['M4'])} of {RM['M4']['cells']}"),
    ('Anechoic grid', 'Primary scenario: beats the best generic estimator in $\\geq 9$ of 12 conditions, loses in none',
     'pass' if _m3r['G1']['passed'] else 'fail', f"{wtl(M3['A1']['best_generic'])}; pooled {tex_iv(M3['A1']['pooled_vs_best_generic_cell'])} per condition, {tex_iv(M3['A1']['pooled_vs_best_generic_task'])} oracle"),
    ('', 'Primary scenario: no loss to the sinc kernel', 'pass' if _m3r['G2']['passed'] else 'fail', wtl(M3['A1']['sinc'])),
    ('', 'Primary scenario: pooled superior to the shared five-ball variant', 'pass' if _m3r['G3']['passed'] else 'fail',
     f"{wtl(M3['A1']['gsmf'])}; pooled {tex_iv(M3['A1']['gsmf']['pooled'])}"),
    ('', 'Second scenario, farther and with a reflector: beats the best generic estimator in $\\geq 9$ of 12 conditions, loses in none',
     'pass' if _m3r['G4']['passed'] else 'fail', f"{wtl(M3['R2']['best_generic'])}; pooled {tex_iv(M3['R2']['pooled_vs_best_generic_cell'])} per condition, {tex_iv(M3['R2']['pooled_vs_best_generic_task'])} oracle"),
    ('Follow-ups', 'Room-grid win rule, corrected sparse Bayesian comparators', 'pass' if fix['M1_corrected']['passed'] else 'fail',
     f"{wtl(fix['M1_corrected'])}; pooled {tex_iv(fix['pooled_vs_task_best_generic'])} oracle"),
    ('', 'Exploratory, set after a same-problem preview with our adaptive kernel: pooled superior to published adaptive-kernel code (one bin)', 'pass' if rib['R1']['passed'] else 'fail',
     f"pooled {tex_iv(rib['R1']['pooled'])} over 6 loudspeakers"),
    ('', 'Exact ball kernel, both five-ball variants, 24 and 48 microphones: no lost win, no loss',
     'pass' if exr['gsmf']['as_registered']['robust_to_kernel'] and exr['mo26_cap5']['as_registered']['robust_to_kernel'] else 'fail',
     f"shared {exr['gsmf']['as_registered']['exact']['wins']}/{exr['gsmf']['as_registered']['exact']['ties']}/{exr['gsmf']['as_registered']['exact']['losses']}; "
     f"per-bin {exr['mo26_cap5']['as_registered']['exact']['wins']}/{exr['mo26_cap5']['as_registered']['exact']['ties']}/{exr['mo26_cap5']['as_registered']['exact']['losses']}"),
    ('', 'Exact ball kernel, shared five-ball variant, 16 conditions: no lost win, no loss',
     'pass' if hi['part_b']['rule_kx_gsmf']['as_registered']['robust_to_kernel'] else 'fail',
     f"{V['hi-B-ex'].replace('/', '/')}; pooled {tex_iv(hi['part_b']['headline']['exact'])}"),
]
put('rules-failed', WORDS[sum(r[2] == 'fail' for r in rule_rows)])
_px_rules = [pa['rules']['primary_vs_mo26_prox']['passed'], pa['rules']['primary_vs_gsmf_prox']['passed'],
             ps['rules']['primary_vs_mo26_prox_s2']['passed']]
put('px-rules-phrase', 'all three passed' if all(_px_rules) else f"{WORDS[sum(_px_rules)]} of the three passed")
_grp, _cur = [], None
for r in rule_rows:
    _cur = r[0] or _cur
    _grp.append(_cur)
_conf = [r for r, gname in zip(rule_rows, _grp) if gname in ('Simulation', 'Room grid', 'Anechoic grid')]
_fol = [r for r, gname in zip(rule_rows, _grp) if gname == 'Follow-ups']
assert len(_conf) == 15 and len(_fol) == 4, 'one row per registered rule (S1-S7, M1-M4, G1-G4) and per follow-up rule on the main claims'
# the rule against the published code was set after a same-problem preview: shown, but not counted
_fol = [r for r in _fol if not r[1].startswith('Exploratory')]
assert len(_fol) == 3
put('rules-n', WORDS[len(_conf)]); put('rules-n-failed', WORDS[sum(r[2] == 'fail' for r in _conf)])
put('rules-n-failed-cap', WORDS[sum(r[2] == 'fail' for r in _conf)].capitalize())
put('rules-fu-n', WORDS[len(_fol)]); put('rules-fu-failed', WORDS[sum(r[2] == 'fail' for r in _fol)])
_nf = sum(r[2] == 'fail' for r in _fol)
put('rules-fu-phrase', f'all {WORDS[len(_fol)]} rules of the registered follow-ups that bear on the main claims passed' if _nf == 0 else
    f'{WORDS[_nf]} of the {WORDS[len(_fol)]} rules of the registered follow-ups that bear on the main claims failed')
lines = [r'\begin{tabular}{@{}>{\raggedright\arraybackslash}p{1.25cm}>{\raggedright\arraybackslash}p{7.75cm}>{\raggedright\arraybackslash}p{0.95cm}>{\raggedright\arraybackslash}p{6.55cm}@{}}',
         r'\toprule', r'Study & Registered rule & Outcome & Numbers (wins/ties/losses; pooled dB)\\', r'\midrule']
for st, rule, oc, nums in rule_rows:
    oc_t = r'\textbf{fail}' if oc == 'fail' else 'pass'
    lines.append(f'{st} & {rule} & {oc_t} & {nums}\\\\')
lines += [r'\bottomrule', r'\end{tabular}']
(OUT / 'v2-outcomes-table.tex').write_text('\n'.join(lines) + '\n')

# (2) runtime from the recorded process times (median seconds per problem of six bins)
m2rows = [r for r in rows('TB-M2') if r.get('kind') == 'mesh' and 'error' not in r]


def tsum(r, keys):
    t = r['timing']
    return sum(v for k, v in t.items() if any(k == p or (p.endswith('*') and k.startswith(p[:-1])) for p in keys))


def t_prop(r):
    """The proposed estimator: greedy fit, cross-validation and the extension at the cross-validated order only (the
    Schwarz-order extension, an ablation, is not counted)."""
    t = r['timing']; k = r['info']['j_K']['cv']
    return t['j_fit'] + t['j_cv'] + (t.get(f'j_ext_K{k}', 0.0) if k > 0 else 0.0)


RT = [('Proposed (with extension)', t_prop),
      ('Points only (no extension)', ['j_fit', 'j_cv']),
      ('Proposed, each bin fitted alone', ['ind_fit', 'ind_cv_*', 'ind_ext_*']),
      ('Single-frequency estimator', ['v1']),
      ('Per-bin five-ball variant', ['mo26_cap5']), ('Shared five-ball variant', ['gsmf']),
      ('Sparse Bayesian (EM)', ['sbl_em']), ('Sparse Bayesian (fixed point)', ['sbl_mackay']),
      ('Adaptive kernel (ours)', ['adk']), ('Squared-exponential kernel', ['rbf'])]
_rt = []
for name, keys in RT:
    vals = []
    for M in (24, 48, 96, 192):
        vals.append(np.median([keys(r) if callable(keys) else tsum(r, keys) for r in m2rows if r['M'] == M]))
    _rt.append(name + ' & ' + ' & '.join(f'{v:,.0f}' for v in vals))
lines = [r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule', r'Estimator & $M=24$ & $48$ & $96$ & $192$\\', r'\midrule']
lines += [row + r'\\' for row in _rt]          # one column wide: the supplement is double-column
lines += [r'\bottomrule', r'\end{tabular}']
(OUT / 'v2-runtime-table.tex').write_text('\n'.join(lines) + '\n')
prim = {M: np.median([t_prop(r) for r in m2rows if r['M'] == M]) for M in (24, 192)}
put('rt-cv-share', f"{100 * min(v['cross_validation_share'] for v in R3S['A4_runtime']['proposed'].values()):.0f}--"
                   f"{100 * max(v['cross_validation_share'] for v in R3S['A4_runtime']['proposed'].values()):.0f}")
assert all(abs(prim[M] - R3S['A4_runtime']['proposed'][str(M)]['proposed_cv_extension_only']) < 1e-9 for M in (24, 192))
put('rt-prim-24', f'{prim[24]:,.0f}'); put('rt-prim-192', f'{prim[192]:,.0f}')
lap = [r for r in rows('TB-PROX') if 'error' not in r]
put('rt-mo26prox', f"{np.median([r['process_seconds'] for r in lap if r['variant'] == 'mo26']):,.0f}")
a1 = [r for r in rows('TB-PROX', 'gsmf-a1') if 'error' not in r]
put('rt-gsmfprox', f"{np.median([r['process_seconds'] for r in a1]):,.0f}")
# iteration budgets of the proximal adaptations, counted from the committed rows
pxb = [b for r in lap if r['variant'] == 'mo26' for b in r['info']['bins']]
put('px-bins', f'{len(pxb)}')
put('px-s1-cap', str(sum(int(b['iterations'][0]) >= 2000 for b in pxb)))
put('px-s2-cap', str(sum(int(b['iterations'][1]) > 100 for b in pxb)))        # 101 evaluations = the 100-step cap
psb = [b for r in rows('TB-PROX-S2') if 'error' not in r for b in r['info']['bins']]
put('ps-s2-cap', str(sum(int(b['iterations'][1]) > 100 for b in psb)))
assert all(int(r['info']['iterations']) == 2000 for r in a1) and len(a1) == 128
gmed = float(np.median([int(r['info']['regions']) for r in a1]))
put('px-g-regions', f'{gmed:,.0f}')                                              # 699.5 rounds to 700, as in the record
put('rt-ex-gsmf-24', f"{np.median([r['seconds']['gsmf_exact'] for r in erow if r['M'] == 24]):,.0f}")
put('rt-ex-gsmf-192', f"{np.median([r['seconds']['gsmf_exact'] for r in hrow if r['M'] == 192]):,.0f}")
put('rt-ex-mo26-24', f"{np.median([r['seconds']['mo26_cap5_exact'] for r in erow if r['M'] == 24]):,.0f}")

# (3) the paper-informed proximal adaptations per band
lines = [r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule',
         r'Band & \multicolumn{2}{c}{Sparse adaptation} & \multicolumn{2}{c}{Group-sparse adaptation}\\',
         r' & proposed $-$ it & it $-$ cap-5 kernel & proposed $-$ it & it $-$ group-sparse kernel\\', r'\midrule']
for b in ('125.0', '250.0', '500.0', '1000.0'):
    cellk = str((float(b), 24))
    d = pa['cells'][cellk]
    lines.append(f"{int(float(b))}~Hz & {tex_iv(d['j_cv_ext-vs-mo26_prox'])} & {tex_iv(d['mo26_prox-vs-mo26_cap5'])} & "
                 f"{tex_iv(d['j_cv_ext-vs-gsmf_prox'])} & {tex_iv(d['gsmf_prox-vs-gsmf'])}\\\\")
lines += [r'\midrule', f"Pooled & {tex_iv(P['j_cv_ext-vs-mo26_prox'])} & {tex_iv(P['mo26_prox-vs-mo26_cap5'])} & "
          f"{tex_iv(P['j_cv_ext-vs-gsmf_prox'])} & {tex_iv(P['gsmf_prox-vs-gsmf'])}\\\\", r'\bottomrule', r'\end{tabular}']
(OUT / 'v2-faithful-table.tex').write_text('\n'.join(lines) + '\n')

# (4) supplement: every simulation condition, as two side-by-side halves of means with class marks
def _mc(d):
    v = d['mean']; mark = {'win': '', 'tie': r'^{\circ}', 'loss': r'^{\times}'}[d['cls']]
    return f"${v:+.2f}{mark}$".replace('+', '+')
_rows = []
for c in CELLS:
    d = C[c]['comparisons']
    _rows.append(f"{sim_label(c)} & {_mc(d['j_cv_ext-vs-' + C[c]['best_generic']])} & {_mc(d['j_cv_ext-vs-gsmf'])} & "
                 f"{_mc(d['j_cv_ext-vs-mo26_cap5'])} & {_mc(d['j_cv_ext-vs-sinc_ev'])} & {100 * C[c]['cap_bound_fraction']:.0f}")
_h = (len(_rows) + 1) // 2
_hdr1 = r'Condition & best & shared & per-bin & sinc & cap'
lines = [r'\begin{tabular}{@{}lrrrrr@{\hspace{10pt}}lrrrrr@{}}', r'\toprule', _hdr1 + ' & ' + _hdr1 + r'\\', r'\midrule']
for k in range(_h):
    right = _rows[_h + k] if _h + k < len(_rows) else ' & & & & & '
    lines.append(_rows[k] + ' & ' + right + r'\\')
lines += [r'\bottomrule', r'\end{tabular}']
(OUT / 'v2-sim-cells-table.tex').write_text('\n'.join(lines) + '\n')

# (5) supplement: every measured condition (room grid), as two side-by-side halves of means with class marks
def _mc2(d):
    if d is None or d.get('mean') is None or not math.isfinite(d['mean']) or d.get('cls') not in ('win', 'tie', 'loss'):
        return 'undef.'
    return _mc(d)
fixc = fix['M1_corrected']['per_cell']
exb = hi['part_b']['cells_table']
_mrows = []
for c in MCELLS:
    d = MC[c]['comparisons']
    _mrows.append(f"{meas_label(c)} & {_mc2(d[f'j_cv_ext-vs-' + MC[c]['best_generic']])} & {_mc2(fixc[c])} & {_mc2(d['j_cv_ext-vs-gsmf'])} & "
                  f"{_mc2(exb[c]['comparisons']['j_cv_ext-vs-gsmf_exact'])} & {_mc2(d['j_cv_ext-vs-mo26_cap5'])} & {_mc2(d['j_cv_ext-vs-two_stage_point'])}")
_h = len(_mrows) // 2
_hdr2 = r'Condition & best & corr. & shared & exact & per-bin & arrival'
lines = [r'\begin{tabular}{@{}lrrrrrr@{\hspace{10pt}}lrrrrrr@{}}', r'\toprule', _hdr2 + ' & ' + _hdr2 + r'\\', r'\midrule']
for k in range(_h):
    lines.append(_mrows[k] + ' & ' + _mrows[_h + k] + r'\\')
lines += [r'\bottomrule', r'\end{tabular}']
(OUT / 'v2-meas-cells-table.tex').write_text('\n'.join(lines) + '\n')

# (6) the six simulation conditions where non-inferiority to the group-sparse kernel fails
lines = [r'\begin{tabular}{@{}lrl@{}}', r'\toprule', r'Condition & Proposed $-$ group-sparse (dB) & Class\\', r'\midrule']
for c in fail:
    d = C[c]['comparisons']['j_cv_ext-vs-gsmf']
    lines.append(f"{sim_label(c)} & {tex_iv(d)} & {d['cls']}\\\\")
lines += [r'\bottomrule', r'\end{tabular}']
(OUT / 'v2-s2-fail-table.tex').write_text('\n'.join(lines) + '\n')

# (7) supplement: the known-answer checks of the final estimator's pieces, as run before the registered measured study
kv2 = json.loads((S / 'studies/TB-M2/primary/known-answer-v2.json').read_text())


def sci(x):
    if x == 0:
        return '0'
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    return f'${m:.1f}\\times10^{{{e}}}$' if abs(e) >= 3 else f'${x:.4g}$'


lines = [r'\begin{longtable}{@{}p{13.2cm}rr@{}}', r'\caption{Known-answer checks of the pieces of the final estimator, run before the registered studies}\label{tab:s-ka2}\\',
         r'\toprule', r'Check & Value & Bound\\', r'\midrule\endfirsthead', r'\toprule', r'Check & Value & Bound\\', r'\midrule\endhead']
for t in kv2['tests']:
    name = (t['test'].replace('%', '\\%').replace('localisation', 'localization')
            .replace('group-sparse multi-frequency region kernel', 'shared five-ball variant')
            .replace('remainder family', 'remainder-kernel choice').replace('is Gaussian or sum', 'is squared-exponential or sum'))
    lines.append(f"{name} & {sci(t['value'])} & $\\le$ {sci(t['tolerance'])}\\\\")
lines += [r'\bottomrule', r'\end{longtable}']
(OUT / 'v2-ka-table.tex').write_text('\n'.join(lines) + '\n')
put('ka2-n', cnt(kv2['n'])); put('ka2-passed', 'all passed' if kv2['passed'] else 'not all passed')

# (7a) supplement: the fractional factorial design, factors by runs, read from the cell definitions
ffc = sorted((c for c in CELLS if c.startswith('FF-')), key=lambda c: int(c[3:]))
assert [int(c[3:]) for c in ffc] == list(range(16))
lines = [r'\begin{tabular}{@{}l' + 'c' * len(ffc) + '@{}}', r'\toprule',
         'Factor & ' + ' & '.join(str(int(c[3:])) for c in ffc) + r'\\', r'\midrule']
for fac in 'ABCDEFGH':
    lines.append(fac + ' & ' + ' & '.join('$+$' if CELLS[c]['ff_signs'][fac] > 0 else '$-$' for c in ffc) + r'\\')
lines += [r'\bottomrule', r'\end{tabular}']
(OUT / 'v2-ff-design.tex').write_text('\n'.join(lines) + '\n')

# (7b) post hoc envelope check of the exact uniform-ball kernel (not registered), and its effect estimate
KE = S / 'studies' / 'AUDITS' / 'kernel-envelope-2026-11-06'
ke = json.loads((KE / 'kernel-envelope.json').read_text())['summary']


def sci1(x):
    e = int(math.floor(math.log10(abs(x)))); m = x / 10 ** e
    return f'${m:.1f}\\times10^{{{e}}}$' if round(m, 1) != 1.0 else f'$10^{{{e}}}$'


put('ke-c1-mic', sci1(ke['C1_max_mic_80_40'])); put('ke-c1-tgt', sci1(ke['C1_max_tgt_80_40']))
put('ke-c2', f"{100 * ke['C2_max_reg_vs_ref']:.1f}\\%"); put('ke-c3', sci1(ke['C3_max_rel_err']))
put('ke-round', sci1(ke['C1_max_reg40_vs_mp40']))
corner = set()                                   # problems with a ball where the check failed (PLAN.md outcome rule 2)
for st, meths in (('TB-M2-EXACT', ('gsmf', 'mo26_cap5')), ('TB-M2-EXACT-HI', ('gsmf',))):
    for r in rows(st):
        for m in meths:
            flat = [x for b in r['info']['exact'][f'{m}_radii'] for x in (b if isinstance(b, list) else [b])]
            if (r['band'] == 1000.0 and max(flat) >= 0.4) or (r['band'] in (500.0, 1000.0) and max(flat) >= 0.45):
                corner.add((st, r['source'], r['band'], r['M'], r['draw']))
put('ke-n-problems', str(len(corner)))
if (KE / 'effect.json').exists():
    ef = json.loads((KE / 'effect.json').read_text())['summary']
    assert ef['errors'] == 0
    put('ke-effect', sci1(ef['max_abs_change_db'])); put('ke-effect-balls', str(ef['balls_refined']))
st = json.loads((KE / 'stress.json').read_text())
assert st['summary']['errors'] == 0
put('ke-stress', f"{st['summary']['max_abs_change_db']:.4f}"); put('ke-stress-bin', f"{st['summary']['max_abs_bin_change_db']:.4f}")
tn = [r['targets_in_or_near'] for r in st['rows']]
put('ke-stress-targets', f'{min(tn)} to {max(tn)}')

# (8) post hoc identifiability check (not registered): Monte Carlo check of the bound, theory curves, comparison
CRB = S / 'posthoc' / 'crb'
mc = json.loads((CRB / 'check-bound.json').read_text())
put('crb-mc-one-bound', f"{mc['one bin']['bound_rms_m']:.3f}~m"); put('crb-mc-one-emp', f"{mc['one bin']['empirical_rms_m']:.3f}~m")
put('crb-mc-six-bound', f"{mc['six bins']['bound_rms_m']:.3f}~m"); put('crb-mc-six-emp', f"{mc['six bins']['empirical_rms_m']:.3f}~m")
put('crb-mc-excess', f"{100 * max(mc[b]['empirical_rms_m'] / mc[b]['bound_rms_m'] - 1 for b in ('one bin', 'six bins')):.1f}")
put('crb-mc-trials', str(mc['one bin']['trials']))
th = json.loads((CRB / 'theory.json').read_text())['curves']
kas, mics = sorted({c['ka'] for c in th}), sorted({c['M'] for c in th})
assert kas == [0.5, 2.0, 8.0] and mics == [24, 48, 96, 192]
cross, slopes, kaf = [], [], []
for M in mics:
    for ka in kas:
        c = sorted((r for r in th if r['M'] == M and r['ka'] == ka), key=lambda r: r['d_over_a'])
        q = np.array([r['d_over_a'] for r in c]); b = np.array([r['one_bin'] for r in c])
        i = int(np.argmax(b >= 1.0)); assert b[0] < 1.0 <= b[-1] and i > 0
        cross.append(float(np.exp(np.interp(0.0, np.log(b[i - 1:i + 1]), np.log(q[i - 1:i + 1])))))
        sel = q >= 1.5
        slopes.append(float(np.polyfit(np.log(q[sel]), np.log(b[sel]), 1)[0]))
    for qv in sorted({r['d_over_a'] for r in th}):
        v = [r['one_bin'] for r in th if r['M'] == M and r['d_over_a'] == qv]
        kaf.append(max(v) / min(v))
put('crb-th-cross-lo', f'{min(cross):.1f}'); put('crb-th-cross-hi', f'{max(cross):.1f}')
put('crb-th-slope-lo', f'{min(slopes):.1f}'); put('crb-th-slope-hi', f'{max(slopes):.1f}')
put('crb-th-ka-factor', f'{max(kaf):.1f}')
div = [1.0 / r['ratio'] for r in th]
put('crb-th-div-lo', f'{min(div):.2f}'); put('crb-th-div-hi', f'{max(div):.2f}')
if (CRB / 'comparison.json').exists():
    cmpd = json.loads((CRB / 'comparison.json').read_text())
    for st, tag in (('TB-S2', 's2'), ('TB-M2', 'm2')):
        r = cmpd['studies'][st]
        n_in = sum(c['in_band'] for c in r['per_condition'])
        put(f'crb-{tag}-inband', f"{n_in} of {r['conditions']}")
        put(f'crb-{tag}-tracks', 'tracked' if r['tracks'] else 'did not track')
        for k, key in (('R_obs_median', 'robs'), ('R_pred_median', 'rpred'), ('R_pred_a_median', 'rpred-a'),
                       ('R_pred_b_median', 'rpred-b'), ('R_obs_rms_median', 'robs-rms'), ('R_pred_rms_median', 'rpred-rms'),
                       ('obs_over_pred_median', 'oop'), ('obs_over_pred_min', 'oop-min'), ('obs_over_pred_max', 'oop-max'),
                       ('spearman_R_obs_R_pred', 'spearman')):
            put(f'crb-{tag}-{key}', num(r[k], 2, sign=False) if key == 'spearman' else f'{r[k]:.2f}')
        put(f'crb-{tag}-AJ', f"{r['A_J_median']:.1f}"); put(f'crb-{tag}-Af', f"{r['A_f_median']:.1f}")
        put(f'crb-{tag}-eta', f"{10 * math.log10(r['plugin_eta_median']):.1f}~dB")
        put(f'crb-{tag}-rho', num(10 * math.log10(r['plugin_rho_median']), 1, sign=False) + '~dB')
        if tag == 'm2':
            low = [c for c in r['per_condition'] if c['condition'].startswith('125 Hz')]
            assert len(low) == 4
            put('crb-m2-125-Af-lo', f"{min(c['A_f'] for c in low):.2f}"); put('crb-m2-125-Af-hi', f"{max(c['A_f'] for c in low):.2f}")
            put('crb-m2-125-Bf-lo', f"{min(c['median_B_f'] for c in low):.1f}"); put('crb-m2-125-Bf-hi', f"{max(c['median_B_f'] for c in low):.1f}")
        put(f'crb-{tag}-atbound', f"{100 * r['plugin_rho_at_bound_share']:.0f}")
    rb = cmpd['rebuild_max_rel_diff']
    put('crb-rebuild', 'were identical to the registered ones' if rb == 0 else
        'matched the registered ones to a relative difference of ' + num(rb, 1, sign=False))
    # the reading written after the comparison was read (posthoc/crb/reading.py)
    rd = json.loads((CRB / 'reading.json').read_text())
    sg, ot = rd['s2_single_driver_clear_of_floor'], rd['s2_other']
    assert sg['conditions'] + ot['conditions'] == cmpd['studies']['TB-S2']['conditions']
    put('crb-floor', f"{rd['floor_clearance_m']:.1f}")
    put('crb-single-n', WORDS[sg['conditions']])
    put('crb-single-agree', f"all {WORDS[sg['conditions']]}" if sg['in_band'] == sg['conditions']
        else f"{WORDS[sg['in_band']]} of the {WORDS[sg['conditions']]}")
    tr = {st: cmpd['studies'][st]['tracks'] for st in ('TB-S2', 'TB-M2')}
    put('crb-tracks-phrase', 'so in neither study did the bound track the observed ratio' if not any(tr.values())
        else 'so in both studies the bound tracked the observed ratio' if all(tr.values())
        else 'so the bound tracked the observed ratio ' + ('in simulation only' if tr['TB-S2'] else 'on the grid only'))
    put('crb-single-in', str(sg['in_band'])); put('crb-single-robs', f"{sg['R_obs_median']:.2f}")
    put('crb-single-rpred', f"{sg['R_pred_median']:.2f}"); put('crb-single-AJ', f"{sg['A_J_median']:.1f}")
    put('crb-other-n', str(ot['conditions'])); put('crb-other-in', WORDS[ot['in_band']])
    put('crb-other-robs', f"{ot['R_obs_median']:.2f}"); put('crb-other-AJ', f"{ot['A_J_median']:.1f}")
    m24 = rd['m2_by_microphones']['24']
    put('crb-m2-24-in', WORDS[m24['in_band']]); put('crb-m2-24-n', WORDS[m24['conditions']])
    n_more = sum(v['in_band'] for k, v in rd['m2_by_microphones'].items() if k != '24')
    put('crb-m2-more-in', 'none' if n_more == 0 else WORDS[n_more])
    put('crb-below-n', WORDS[len(rd['below_band'])])
    put('crb-below-AJ-lo', f"{rd['below_band_A_J_range'][0]:.2f}"); put('crb-below-AJ-hi', f"{rd['below_band_A_J_range'][1]:.2f}")
    put('crb-below-Af-lo', f"{rd['below_band_A_f_range'][0]:.1f}"); put('crb-below-Af-hi', f"{rd['below_band_A_f_range'][1]:.1f}")
    put('crb-eff-lo', f"{rd['efficient_median_over_rms'][0]:.2f}"); put('crb-eff-hi', f"{rd['efficient_median_over_rms'][1]:.2f}")

# (9) post hoc comparison of the cross-validated order with fixed orders, and the stricter classification (not registered)
FO = json.loads((S / 'posthoc' / 'fixed-order' / 'fixed-order.json').read_text())['studies']
for st, tag in (('TB-S2', 's2'), ('TB-M2', 'm2')):
    r = FO[st]
    assert r['cv_equals_selected_fixed_order_mismatches'] == 0
    for K in (1, 2, 5):
        d = r['cv_vs_fixed'][f'K{K}']
        put(f'fo-{tag}-K{K}', wtl(d)); put(f'fo-{tag}-K{K}-pooled', iv(d['pooled']))
        put(f'fo-{tag}-K{K}-wins', cnt(d['wins'])); put(f'fo-{tag}-K{K}-ties', cnt(d['ties']))
    d = r['fixed_vs_fixed']['K5-vs-K1']
    put(f'fo-{tag}-K5K1', wtl(d)); put(f'fo-{tag}-K5K1-pooled', iv(d['pooled']))
    put(f'fo-{tag}-best', WORDS[r['best_fixed_order']['K']])
    for comp, nm in (('best_generic', 'bg'), ('gsmf', 'gsmf'), ('mo26_cap5', 'mo26'), ('sinc_ev', 'sinc')):
        stc, reg = r['strict'][comp]['strict'], r['strict'][comp]['registered']
        put(f'st-{tag}-{nm}', wtl(stc) + (f", {WORDS[stc['undefined']]} undefined" if stc['undefined'] else ''))
        _got = f"{reg['wins']}/{reg['ties'] + (0 if (tag, nm) == ('m2', 'mo26') else reg['undefined'])}/{reg['losses']}"
        assert _got == {('s2', 'bg'): V['sim-S1'], ('s2', 'gsmf'): V['sim-S2-gsmf'],
                                                        ('s2', 'mo26'): V['sim-S2-mo26'], ('s2', 'sinc'): V['sim-S6'],
                                                        ('m2', 'bg'): V['meas-M1'], ('m2', 'gsmf'): V['meas-M2-gsmf'],
                                                        ('m2', 'mo26'): V['meas-M2-mo26'], ('m2', 'sinc'): V['meas-M3']}[(tag, nm)]

_cvbest = [FO[st]['best_fixed_order']['cv_vs_best']['mean'] for st in ('TB-S2', 'TB-M2')]
assert all(v >= 0 for v in _cvbest)
put('abs-fo', f'{max(_cvbest):.2f}')                                     # cross-validated minus best fixed order, worst study

# (9b) post hoc addendum to the identifiability check: the reflection on the planar grid, per-problem attainment
RF = json.loads((S / 'posthoc' / 'crb' / 'reflection.json').read_text())
ra = RF['reflection_measured']
assert ra['max_abs_diff_from_registered_distance_m'] < 1e-9 and ra['plane_z_spread_m'] == 0.0
put('crb-mirror-n', f"{ra['closer_to_mirror']:,}"); put('crb-mirror-of', f"{ra['problems']:,}")
put('crb-plane-n', f"{ra['within_1mm_of_plane']:,}")
put('crb-m2-dcat', f"{ra['median_d_catalogue_m']:.3f}"); put('crb-m2-dinv', f"{ra['median_d_reflection_invariant_m']:.3f}")
at = RF['attainment']
put('crb-s2-att', f"{at['TB-S2']['median_over_conditions']['median']:.2f}")
put('crb-m2-att', f"{at['TB-M2']['median_over_conditions']['median']:.2f}")
put('crb-m2-att-inv', f"{at['TB-M2']['reflection_invariant']['median_over_conditions']['median']:.2f}")
assert abs(RF['efficient_median_range'][0] - float(V['crb-eff-lo'])) < 0.005 and abs(RF['efficient_median_range'][1] - float(V['crb-eff-hi'])) < 0.005

# (9c) post hoc descriptive margins (posthoc/margins)
MG = json.loads((S / 'posthoc' / 'margins' / 'margins.json').read_text())
for b_, tag in (('two_stage_point', 'pt'), ('two_stage_region', 'rg')):
    for part, ptag in (('all', 'all'), ('below_1kHz', 'below')):
        x = MG['arrival_time_baselines'][b_][part]
        assert x['undefined'] == 0
        put(f'at-{tag}-{ptag}', f"{x['wins']}/{x['ties']}/{x['losses']}"); put(f'at-{tag}-{ptag}-pooled', iv(x['pooled']))
assert MG['arrival_time_baselines']['two_stage_point']['all']['wins'] == int(V['meas-tsp'].split('/')[0])
# conditions below 1 kHz in which the arrival-time point model was more accurate (losses of the proposed estimator)
put('at-pt-below-losses', WORDS[MG['arrival_time_baselines']['two_stage_point']['below_1kHz']['losses']])
for f_ in ('125', '250', '500', '1000'):
    put(f'mb-{f_}', iv(MG['measured_margin_by_band'][f_]))
    put(f'mb-{f_}-mean', num(MG['measured_margin_by_band'][f_]['mean']))
sm = MG['simulated_margin']
put('sm-all', iv(sm['all_63'])); put('sm-noin', iv(sm['without_inside_62'])); put('sm-in', iv(sm['inside_alone']))
md = MG['median_error_db']
put('abs-m-prop', num(md['measured']['proposed'], 1)); put('abs-m-bg', num(md['measured']['best_generic_of_condition'], 1))
put('abs-m-tsp', num(md['measured']['arrival_time_point'], 1))
put('abs-s-prop', num(md['simulated_source_proximal']['proposed'], 1))
put('abs-s-bg', num(md['simulated_source_proximal']['best_generic_of_condition'], 1))
put('fo-s2-K1-cost', f"{-FO['TB-S2']['cv_vs_fixed']['K1']['pooled']['mean']:.2f}")
put('fo-m2-K5-cost', f"{-FO['TB-M2']['cv_vs_fixed']['K5']['pooled']['mean']:.2f}")

# (10) source proximity and scan spacing, from the geometry only (posthoc/geometry)
GEO = json.loads((S / 'posthoc' / 'geometry' / 'geometry.json').read_text())
assert all(c['identical'] for c in GEO['check_against_unpatched_builder'])


def span2(d, k=2):
    return f"{d['min']:.{k}f}--{d['max']:.{k}f}"


gs = GEO['simulation']
put('geo-base-da', span2(gs['base']['d_over_a'])); put('geo-far-da', span2(gs[FAR]['d_over_a']))
put('geo-below01-da', span2(gs['A1-below0.1']['d_over_a']))
put('geo-in-da', span2(gs['A1-in']['d_over_a']))
near = [c for c in prox if gs[c]['d_over_a']['max'] < 1.0]           # source inside the smallest ball about the centre in every field
put('geo-sim-inside-ball', cnt(len(near)))
put('geo-sim-da-max', f"{max(gs[c]['d_over_a']['max'] for c in prox):.2f}")
a2 = {int(gs[c]['f']): gs[c]['clearance_wavelengths'] for c in CELLS if c.startswith('A2-')}
put('geo-base-wl-125', f"{a2[125]['min']:.2f}"); put('geo-base-wl-1000', f"{a2[1000]['max']:.2f}")
put('geo-base-clear', f"{gs['base']['clearance_m']['min']:.2f}--{gs['base']['clearance_m']['max']:.2f}")
gm = GEO['measured']
put('abs-da', f"{gm['125']['d_over_a']['min']:.1f}--{gm['125']['d_over_a']['max']:.1f}")
put('geo-meas-da', span2(gm['125']['d_over_a'])); put('geo-meas-a', f"{gm['125']['a_m']:.2f}")
put('geo-sim-a', f"{gs['base']['a_m']:.2f}")
for f in ('125', '250', '500', '1000'):
    put(f'geo-meas-wl-{f}', span2(gm[f]['clearance_wavelengths']))
sg = GEO['scan_grid']
binds = [k for k, v in sg.items() if v['cap_binds']]
assert sorted(binds) == ['measured-1000', 'simulation-1000', 'simulation-500'], binds
_sg3 = json.loads((S / 'studies' / 'TB-M3-MIRACLE' / 'primary' / 'scan-geometry.json').read_text())
assert sorted(k for k, v in _sg3.items() if isinstance(v, dict) and v['cap_binds']) == ['A1-1000', 'A1-500', 'R2-1000', 'R2-500']
_w1k = [sg['simulation-1000']['step_wavelengths'], sg['measured-1000']['step_wavelengths'], _sg3['A1-1000']['step_wavelengths'], _sg3['R2-1000']['step_wavelengths']]
_m1k = [sg['simulation-1000']['step_m'], sg['measured-1000']['step_m'], _sg3['A1-1000']['step_m'], _sg3['R2-1000']['step_m']]
put('geo-scan-1k-m', f"{min(_m1k):.2f}--{max(_m1k):.2f}"); put('geo-scan-1k-wl', f"{min(_w1k):.2f}--{max(_w1k):.2f}")
_w500 = [sg['simulation-500']['step_wavelengths'], _sg3['A1-500']['step_wavelengths'], _sg3['R2-500']['step_wavelengths']]
put('geo-scan-500-wl', f"{min(_w500):.2f}--{max(_w500):.2f}")

# (11) post hoc follow-ups to the third review (posthoc/review3): driver geometry, margins by geometry, every member
A1 = R3S['A1_driver_geometry']; A2 = R3S['A2_margins_by_driver_geometry']; A3 = R3S['A3_every_generic_member']
_int = A1['interior_conditions']
assert _int == ['A1-in', 'A5-delta0.4', 'FF-02', 'FF-03', 'FF-06', 'FF-07'], _int
assert all(A1['per_condition'][c]['interior_all_fields'] for c in _int)
put('r3-n-int', WORDS[len(_int)]); put('r3-n-int-cap', WORDS[len(_int)].capitalize()); put('r3-n-ext', cnt(A2['no_interior_driver']['cells']))
_coh = [A1['per_condition'][c]['drivers']['coherent']['signed_distance_m'] for c in _int if c != 'A1-in']
assert all(abs(d['min'] + 0.1) < 1e-9 and abs(d['max'] + 0.1) < 1e-9 for d in _coh)
put('r3-ext', iv(A2['no_interior_driver']['best_generic'])); put('r3-int', iv(A2['interior_driver']['best_generic']))
put('r3-int5', iv(A2['interior_driver_without_A1_in']['best_generic']))
assert A2['no_interior_driver']['best_generic_classes'] == dict(win=A2['no_interior_driver']['cells'], tie=0, loss=0)
_eg = A2['no_interior_driver']['gsmf']
put('r3-ext-gsmf', wtl(_eg)); put('r3-ext-gsmf-ni', cnt(_eg['non_inferior'])); put('r3-ext-gsmf-pooled', iv(_eg['pooled']))
put('r3-ext-mo26-pooled', iv(A2['no_interior_driver']['mo26_cap5']['pooled']))
put('r3-ext-mo26-ni', cnt(A2['no_interior_driver']['mo26_cap5']['non_inferior']))
put('abs-sim-ext', f"{-A2['no_interior_driver']['best_generic']['mean']:.2f}")
put('r3-em-sim', cnt(A3['simulation']['win_against_every_member']))
put('r3-em-meas', cnt(A3['measured_guarded']['win_against_every_member']))
assert A3['simulation']['loss_to_some_member'] == 0 and A3['measured_guarded']['loss_to_some_member'] == 0

# (12) post hoc refits (posthoc/review3/refit-summary.json): the stand-in's corrected gradient, the fitted state
RS = json.loads((S / 'posthoc' / 'review3' / 'refit-summary.json').read_text())
_g = RS['gate']
assert RS['errors'] == 0 and RS['rows'] == 3168, (RS['errors'], RS['rows'])
put('r3-gate-phrase', 'every refit reproduced the registered errors bit for bit before any new quantity was computed'
    if _g['adk_failed'] == 0 and _g['proposed_failed'] == 0 and _g['adk_max'] == 0 and _g['proposed_max'] == 0 else
    f"{_g['adk_failed'] + _g['proposed_failed']} refits did not reproduce the registered errors and are excluded")
_ex = RS['B1_exposure']; _rl = RS['B1_rules']
_nb = sum(v['bins'] for v in _ex.values()); _nc = sum(v['bins_clipped_at_some_evaluation'] for v in _ex.values())
assert all(v['unclipped_bins_identical'] for v in _ex.values())
_s1c, _m1c, _m1g = _rl['S1_corrected'], _rl['M1_corrected_adk'], _rl['M1_corrected_adk_guarded_sbl']
put('r3-adk-clip', f"{_nc:,}"); put('r3-adk-bins', f"{_nb:,}")
put('r3-adk-S1', wtl(_s1c)); put('r3-adk-M1', wtl(_m1c)); put('r3-adk-M1g', wtl(_m1g))
put('r3-adk-M1g-pooled', iv(_m1g['pooled_vs_per_problem_oracle'])); put('r3-adk-S1-pooled', iv(_s1c['pooled_vs_per_problem_oracle']))
put('r3-adk-rib', iv(_rl['RIB_R2_corrected']['pooled']))
_unch = (wtl(_s1c) == V['sim-S1'] and wtl(_m1c) == V['meas-M1'] and wtl(_m1g) == V['meas-M1c'])
if _nc == 0:
    put('r3-adk-phrase', 'a post hoc refit of every registered problem found that no bin reached the clip, so the corrected gradient gives identical fits')
else:
    _nfin = sum(v['bins_clipped_at_final_state'] for v in _ex.values())
    put('r3-adk-phrase', f"post hoc refits of every simulated and room-grid problem with the corrected gradient " + ("left the outcomes of the rules against the best generic estimator unchanged" if _unch else
        f"changed the rules against the best generic estimator to {wtl(_s1c)} in simulation and {wtl(_m1c)} on the room grid")
        + f" (the clip was reached during the fit in {_nc:,} of {_nb:,} bins" + (" and at no final state)" if _nfin == 0 else f" and at the final state in {_nfin:,})"))
put('r3-adk-unchanged', 'yes' if _unch else 'no')
_b2 = RS['B2_fitted_state']
_pc = [_b2[st]['point_clearance_cv_order'] for st in ('TB-M2', 'TB-S2')]
_pa = [_b2[st]['point_clearance_every_order'] for st in ('TB-M2', 'TB-S2')]
_bc = [_b2[st]['ball_clearance'] for st in ('TB-M2', 'TB-S2')]
_n_pt = sum(x['below_guard'] for x in _pc); _n_pa = sum(x['below_guard'] for x in _pa); _n_b = sum(x['below_guard'] for x in _bc)
_n_in = sum(x['problems_with_mic_inside_a_ball'] for x in _bc)
_min_pt = min(x['span']['min'] for x in _pa); _min_b = min(x['span']['min'] for x in _bc)
put('r3-clear-pt-min', f'{_min_pt:.3f}'); put('r3-clear-ball-min', f'{_min_b:.3f}')
if _n_pa == 0 and _n_b == 0:
    put('r3-clear-phrase', f'no fitted point or ball within 0.05~m of a microphone, the smallest clearance being {min(_min_pt, _min_b):.3f}~m')
else:
    _def_mm = 1e3 * (0.05 - min(_min_pt, _min_b))
    put('r3-clear-deficit-mm', f'{_def_mm:.2f}')
    put('r3-clear-phrase', f"that {_n_pa} of the {sum(x['span']['n'] for x in _pa):,} point-model fits of orders one to five had a point nearer than 0.05~m to a microphone, "
        f"as did {_n_b} of the {sum(x['balls'] for x in _bc):,} fitted balls, none by more than {_def_mm:.2f}~mm"
        + (f", and a microphone inside a ball in {_n_in} problems" if _n_in else ''))
_km, _ks = _b2['TB-M2']['kernel'].get('all'), _b2['TB-S2']['kernel'].get('all')
def pct(x, d=2):
    return f'{100 * x:.{d}f}\\%'
put('r3-ker-m-med', pct(_km['frob']['median'])); put('r3-ker-m-p95', pct(_km['frob']['p95'])); put('r3-ker-m-max', pct(_km['frob']['max'], 1))
put('r3-ker-s-med', pct(_ks['frob']['median'])); put('r3-ker-s-p95', pct(_ks['frob']['p95'])); put('r3-ker-s-max', pct(_ks['frob']['max'], 1))
put('r3-ker-m-floor', '{:,}'.format(_km['at_radius_floor'])); put('r3-ker-m-n', f"{_km['ball_bins']:,}")
put('r3-ker-s-floor', '{:,}'.format(_ks['at_radius_floor'])); put('r3-ker-s-n', f"{_ks['ball_bins']:,}")
put('r3-ker-m-kR', f"{_km['kR']['median']:.2f}"); put('r3-ker-s-kR', f"{_ks['kR']['median']:.2f}")
put('r3-ker-m-kRmax', f"{_km['kR']['max']:.2f}"); put('r3-ker-s-kRmax', f"{_ks['kR']['max']:.2f}")
_fm, _fs = _b2['TB-M2']['frozen_exact_swap'], _b2['TB-S2']['frozen_exact_swap']
put('r3-fz-m', iv(_fm['pooled'], 3)); put('r3-fz-s', iv(_fs['pooled'], 3))
put('r3-fz-m-max', f"{max(abs(_fm['change_db']['max']), abs(_fm['change_db']['min'])):.2f}")
put('r3-fz-s-max', f"{max(abs(_fs['change_db']['max']), abs(_fs['change_db']['min'])):.2f}")
put('r3-fz-m-over', cnt(_fm['abs_over_0p1'])); put('r3-fz-s-over', cnt(_fs['abs_over_0p1']))
put('r3-fz-m-n', f"{_fm['problems']:,}"); put('r3-fz-s-n', f"{_fs['problems']:,}")
put('r3-fz-skip', cnt(_fm['skipped_series_invalid'] + _fs['skipped_series_invalid']))
put('r3-tnode-1cm', cnt(_b2['TB-M2']['target_to_node_min']['below_1cm'] + _b2['TB-S2']['target_to_node_min']['below_1cm']))

# (13) post hoc reproduction of the published sparse configuration (posthoc/review3/reproduction-summary.json)
RP = json.loads((S / 'posthoc' / 'review3' / 'reproduction-summary.json').read_text())
assert RP['errors'] == 0 and RP['bins'] == 300 and RP['bands'] == 20
_n = RP['pooled_nmse_db']
for d, tag in (('0.15', '015'), ('0.2', '02'), ('0.25', '025')):
    put(f'rp-ad-{tag}', num(_n[f'adaptation_opt_{d}'], 1)); put(f'rp-ad1-{tag}', num(_n[f'adaptation_stage1_{d}'], 1))
    put(f'rp-pub-{tag}', num(RP['published_fig6']['opt'][d], 1)); put(f'rp-pub1-{tag}', num(RP['published_fig6']['stage1'][d], 1))
put('rp-prop-bin', num(_n['proposed_bin'], 1)); put('rp-prop-shared', num(_n['proposed_shared'], 1)); put('rp-sinc', num(_n['sinc'], 1))
put('rp-diff', iv(RP['paired_per_trial']['proposed_bin_minus_adaptation_opt_0p2']))
put('rp-reproduced', 'yes' if RP['criterion']['reproduced'] else 'no')
put('rp-S', '/'.join(str(v[0]) for v in RP['dictionary_sizes'].values()))

# (14) post hoc reproduction of the published group-sparse configuration and the diagnostic of the sparse one
RG = json.loads((S / 'posthoc' / 'review3' / 'reproduction-gsmf-summary.json').read_text())
assert RG['errors'] == 0 and RG['criterion']['reproduced'] in (True, False)
_g1 = RG['delta_0.1']
put('rg-ad', num(_g1['chosen']['all'], 1)); put('rg-pub', num(_g1['published']['all'], 1))
put('rg-ad-low', num(_g1['chosen']['le_1k'], 1)); put('rg-pub-low', num(_g1['published']['le_1k'], 1))
put('rg-reproduced', 'yes' if RG['criterion']['reproduced'] else 'no')
put('rg-prop-shared', num(RG['proposed_shared']['all'], 1)); put('rg-prop-bin', num(RG['proposed_bin']['all'], 1))
put('rg-diff', iv(RG['paired']['proposed_shared_minus_adaptation'], 1))
put('rg-diff-bin', iv(RG['paired']['proposed_bin_minus_adaptation'], 1))
put('rg-zeta', _g1['chosen_zeta'])
_g2 = RG['delta_0.05']
put('rg-ad-005', num(_g2['chosen']['all'], 1)); put('rg-pub-005', num(_g2['published']['all'], 1))
put('rg-S', '/'.join(str(d[1]) for d in sorted(RG['dictionary'], key=lambda d: -d[0])))
put('rg-prop-shared-low', num(RG['proposed_shared']['le_1k'], 1)); put('rg-prop-bin-low', num(RG['proposed_bin']['le_1k'], 1))
put('rp-bin-prop-K0', cnt(RP['proposed_bin_K'].get('0', 0)))
RD = json.loads((S / 'posthoc' / 'review3' / 'reproduction-diagnostic-summary.json').read_text())
assert RD['errors'] == 0
put('rd-shared-fail', cnt(RD['shared']['failing'])); put('rd-bin-fail', cnt(RD['bin']['failing']))
put('rd-bin-order0', cnt(RD['bin']['failing_order0']))
put('rd-bin-withpt', cnt(RD['bin']['failing_with_point'])); put('rd-bin-mind', f"{RD['bin']['failing_min_dist_m']:.2f}")
put('rd-shared-maxmm', f"{1e3 * RD['shared']['max_dist_m_all_bins']:.1f}"); put('rd-shared-w0', cnt(RD['shared']['failing_first_weight_within_a_decade_of_bound']))
assert RD['shared']['failing_with_point'] == RD['shared']['failing']
_fb = {float(k): v for k, v in RD['shared']['failing_by_freq'].items()}
put('rd-shared-fmin', f"{int(min(f for f, v in _fb.items() if v > 0))}")
_cn = {float(k): v for k, v in RD['shared']['sinc_cond_median_by_freq'].items()}
put('rd-cond-hi', f"{min(v for f, v in _cn.items() if f >= 600):.1f}--{max(v for f, v in _cn.items() if f >= 600):.1f}")
put('rp-pair-shared', iv(RP['paired_per_trial']['proposed_shared_minus_adaptation_opt_0p2'], 1))

# (15) the second measured study (studies/TB-M3-MIRACLE): geometry from the extract receipts
for _sc in ('A1', 'R2'):
    _rc = json.loads((S / 'studies' / 'TB-M3-MIRACLE' / 'extract' / f'miracle-{_sc}-receipt.json').read_text())
    assert _rc['qualification']['passed']
    put(f'm3-da-{_sc}', f"{_rc['qualification']['d_over_a'][0]:.2f}--{_rc['qualification']['d_over_a'][1]:.2f}")
    put(f'm3-dist-{_sc}', f"{_rc['qualification']['plane_to_array_m']:.2f}")
put('m3-a', f"{_rc['qualification']['region_a_m']:.2f}")
_da_lo = min(json.loads((S / 'studies' / 'TB-M3-MIRACLE' / 'extract' / f'miracle-{x}-receipt.json').read_text())['qualification']['d_over_a'][0] for x in ('A1', 'R2'))
put('da-meas-all', f"{_da_lo:.2f} to {gm['125']['d_over_a']['max']:.2f}")

# (15b) the inherited known-answer checks of the earlier estimator's studies (generated/ka-table.tex, version-1 generator)
_ka1 = [l for l in (OUT / 'ka-table.tex').read_text().splitlines() if l.endswith(r'\\') and ' & ' in l and not l.startswith('Check &')]
put('ka1-n', f'{len(_ka1)}')

# (16) post hoc: the extension with a Gaussian source density, and the order cap raised to eight (plans D and F)
def _reading(iv_d, better, worse, same):
    if iv_d['high'] < 0:
        return better
    if iv_d['low'] > 0:
        return worse
    return same


import os
_pending = [n for n in ('gauss-summary.json', 'cap8-summary.json') if not (S / 'posthoc' / 'review3' / n).exists()]
if _pending and os.environ.get('V2_ALLOW_PENDING') != '1':
    raise SystemExit(f'pending post hoc summaries: {_pending}')
if 'gauss-summary.json' not in _pending:
    GS = json.loads((S / 'posthoc' / 'review3' / 'gauss-summary.json').read_text())
    assert GS['errors'] == 0 and GS['gate_failed'] == 0 and GS['rows'] == 3072, (GS['errors'], GS['gate_failed'], GS['rows'])
    for _st, _tag in (('TB-M2', 'm'), ('TB-S2', 's')):
        x = GS[_st]
        put(f'gs-{_tag}', iv(x['gauss_minus_registered']['pooled'], 3)); put(f'gs-{_tag}-cls', '{win}/{tie}/{loss}'.format(**x['gauss_minus_registered']['classes']))
        put(f'gs-{_tag}-bg', '{win}/{tie}/{loss}'.format(**x['gauss_vs_best_generic']['classes']))
        put(f'gs-{_tag}-oracle', iv(x['gauss_vs_best_generic']['pooled_vs_per_problem_oracle']))
        put(f'gs-{_tag}-ext', iv(x['gauss_minus_points'])); put(f'gs-{_tag}-ext-reg', iv(x['registered_minus_points']))
        put(f'gs-{_tag}-read', _reading(x['gauss_minus_registered']['pooled'], 'more accurate', 'less accurate', 'indistinguishable'))
    _gm, _gs = GS['TB-M2']['gauss_minus_registered']['pooled'], GS['TB-S2']['gauss_minus_registered']['pooled']
    _rd = {_reading(_gm, 'b', 'w', 's'), _reading(_gs, 'b', 'w', 's')}
    _gv = dict(b='the Gaussian density was the more accurate choice', s='the choice made no difference',
               w='the registered uniform density was the more accurate choice')
    _inside = all(abs(v) < 0.1 for _d in (_gm, _gs) for v in (_d['low'], _d['high']))
    put('gs-phrase', ('the registered uniform density, with its 33-node rule, was the less accurate choice' if _rd == {'b'} else
        'the registered uniform density, with its 33-node rule, was the more accurate choice here' if _rd == {'w'} else
        'the choice of density and node rule made no difference at this resolution' if _rd == {'s'} else
        f"{_gv[_reading(_gm, 'b', 'w', 's')]} on the room grid and {_gv[_reading(_gs, 'b', 'w', 's')]} in simulation")
        + (', though both intervals lay inside the 0.1~dB margin' if _inside and _rd != {'s'} else ''))
    # main text: the practical reading first, then the pre-set reading, which has no margin
    _pre = [f"{_gv[_reading(_d, 'b', 'w', 's')]} {_w}" for _d, _w in ((_gm, 'on the room grid'), (_gs, 'in simulation')) if _reading(_d, 'b', 'w', 's') != 's']
    put('gs-phrase-main', 'the choice made no difference' if not _pre else
        ('the choice made no practical difference (both intervals inside the 0.1~dB margin), although by the pre-set reading '
         + ' and '.join(_pre)) if _inside else V['gs-phrase'])
    _ka = json.loads((S / 'posthoc' / 'review3' / 'gauss' / 'known-answer.json').read_text())
    assert _ka['passed'] and len(_ka['ii_accuracy']) == 18 and _ka['nodes']['n'] == 48
    _lo = [r for r in _ka['ii_accuracy'] if r['kR'] < 3]; _hi = [r for r in _ka['ii_accuracy'] if r['kR'] >= 3]
    assert len({round(r['kR'], 2) for r in _hi}) == 1
    _pc = lambda v: f"{100 * v:.1f}" if v < 0.1 else f"{100 * v:.0f}"
    _rng = lambda k: f"{min(r[k] for r in _ka['ii_accuracy']):g}--{max(r[k] for r in _ka['ii_accuracy']):g}"
    put('gs-ka-r', _rng('R')); put('gs-ka-cl', _rng('clearance'))
    put('gs-ka-n', cnt(len(_ka['ii_accuracy']))); put('gs-ka-kr-lo', f"{max(r['kR'] for r in _lo):.2f}")
    put('gs-ka-kr-hi', f"{_hi[0]['kR']:.1f}")
    put('gs-ka-g-lo', _pc(max(r['gauss48_rel_err'] for r in _lo))); put('gs-ka-u-lo', _pc(max(r['uniform33_rel_err'] for r in _lo)))
    put('gs-ka-g-hi', f"{_pc(min(r['gauss48_rel_err'] for r in _hi))}--{_pc(max(r['gauss48_rel_err'] for r in _hi))}")
    put('gs-ka-u-hi', f"{_pc(min(r['uniform33_rel_err'] for r in _hi))}--{_pc(max(r['uniform33_rel_err'] for r in _hi))}")
    _gm_, _ge_ = f"{_ka['i_gradient_rel_err']:.0e}".split('e')
    put('gs-ka-grad', f"${int(_gm_)}\\times10^{{{int(_ge_)}}}$")
    _m, _e = f"{_gs['low']:.1e}".split('e')
    put('gs-s-lo', f"${_m}\\times10^{{{int(_e)}}}$")
if 'cap8-summary.json' not in _pending:
    C8 = json.loads((S / 'posthoc' / 'review3' / 'cap8-summary.json').read_text())
    assert C8['errors'] == 0 and C8['gate_failed'] == 0 and C8['rows'] == 3072, (C8['errors'], C8['gate_failed'], C8['rows'])
    for _st, _tag in (('TB-M2', 'm'), ('TB-S2', 's')):
        x = C8[_st]
        put(f'c8-{_tag}', iv(x['cap8_minus_registered']['pooled'], 3)); put(f'c8-{_tag}-cls', '{win}/{tie}/{loss}'.format(**x['cap8_minus_registered']['classes']))
        put(f'c8-{_tag}-bg', '{win}/{tie}/{loss}'.format(**x['cap8_vs_best_generic']['classes']))
        put(f'c8-{_tag}-oracle', iv(x['cap8_vs_best_generic']['pooled_vs_per_problem_oracle']))
        put(f'c8-{_tag}-at8', f"{100 * x['share_at_8']:.0f}"); put(f'c8-{_tag}-changed', f"{100 * x['share_changed']:.0f}")
    _cm, _cs = C8['TB-M2']['cap8_minus_registered']['pooled'], C8['TB-S2']['cap8_minus_registered']['pooled']
    assert _reading(_cs, 'b', 'w', 's') == 'b' and _reading(_cm, 'b', 'w', 's') == 's', 'conclusion wording assumes a simulated gain and no room-grid change'
    put('c8-s-gain', f"{-_cs['mean']:.2f}")
    _rd = {_reading(_cm, 'b', 'w', 's'), _reading(_cs, 'b', 'w', 's')}
    _cv = dict(b='cost accuracy', s='made no measurable difference', w='was the more accurate choice')
    put('c8-phrase', 'the cap of five cost accuracy' if _rd == {'b'} else 'raising the cap cost accuracy' if _rd == {'w'} else
        'raising the cap did not change the accuracy at this resolution' if _rd == {'s'} else
        f"the cap of five {_cv[_reading(_cs, 'b', 'w', 's')]} in simulation and {_cv[_reading(_cm, 'b', 'w', 's')]} on the room grid")

# ------------------------------------------------------------------ spelled-out win/tie/loss counts for prose (key + '-w')
import re as _re


def _words(v):
    w, t, l = (int(x) for x in v.split('/'))
    ws = 'no wins' if w == 0 else f"{w} {'win' if w == 1 else 'wins'}"
    ts = 'no ties' if t == 0 else f"{t} {'tie' if t == 1 else 'ties'}"
    ls = 'no losses' if l == 0 else f"{l} {'loss' if l == 1 else 'losses'}"
    return f'{ws}, {ts} and {ls}'


for k in list(V):
    if _re.fullmatch(r'\d+/\d+/\d+', V[k]):
        put(k + '-w', _words(V[k]))

# ------------------------------------------------------------------ write the numbers
with open(OUT / 'v2-numbers.tex', 'w') as fh:
    fh.write('% Generated by tools/make_numbers_v2.py from the committed registered outputs; do not edit.\n')
    for k in sorted(V):
        fh.write(f'\\vdef{{{k}}}{{{V[k]}}}\n')
print(f'{len(V)} numbers;', ', '.join(sorted(p.name for p in OUT.glob("v2-*.tex"))))
