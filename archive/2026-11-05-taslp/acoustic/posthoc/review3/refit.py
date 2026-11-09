"""Refits on the registered problems of TB-M2 and TB-S2 (PLAN.md, part B). Post hoc; no registered output changes.

Per registered row, the problem is rebuilt with the registered builder and seed, then
  B1  the adaptive stand-in is refitted with the traced registered objective (gate: the stored per-bin NMSE) and with
      the masked, corrected gradient, recording clipping over every evaluation;
  B2  the proposed estimator's greedy fit and extension at the stored cross-validated order are rerun (gate: the stored
      j_K0..j_K5 and j_cv_ext NMSE), and its fitted state, clearances, cubature accuracy at the fitted balls and the
      frozen-state swap to the exact ball kernel are recorded.
Rows stream to OUT/rows.jsonl; an errored task keeps its traceback. The known-answer checks run first and must pass.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
        python3 -B posthoc/review3/refit.py OUT --workers 20 [--limit N] [--only mesh|held|sim]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
sys.path.insert(0, str(S / 'common')); sys.path.insert(0, str(S / 'V2-dev')); sys.path.insert(0, str(HERE))
import estimators as E  # noqa: E402
import problems_v2 as PB  # noqa: E402
import v2  # noqa: E402
import adk_fix as AF  # noqa: E402
import cubature_diag as CD  # noqa: E402

GATE = 1e-12
DATA = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def registered_rows():
    out = []
    for study in ('TB-M2', 'TB-S2'):
        for l in open(S / 'studies' / study / 'primary' / 'rows.jsonl'):
            r = json.loads(l)
            if 'error' in r:
                continue
            if study == 'TB-M2':
                key = dict(kind=r['kind'], source=r['source'], band=r['band'], M=r['M'], draw=r['draw'], seed=r['seed'])
            else:
                key = dict(kind='sim', cell=r['cell'], seed=r['seed'], M=None)
            out.append(dict(study=study, key=key, stored=dict(nmse={m: r['nmse'][m] for m in
                                                                   ('adk_standin', 'j_cv_ext', 'j_K0', 'j_K1', 'j_K2', 'j_K3', 'j_K4', 'j_K5')},
                                                              K_cv=r['info']['j_K']['cv'])))
    return out


def init():
    z = np.load(S / 'B3-measured/meshrir-prep/meshrir-prep.npz', allow_pickle=False)
    DATA['mesh'] = {k: z[k] for k in z.files}
    z = np.load(S / 'B3-measured/heldout-prep/heldout.npz', allow_pickle=False)
    DATA['held'] = {k: z[k] for k in z.files}
    DATA['held_meta'] = json.loads((S / 'B3-measured/heldout-prep/heldout.json').read_text())


def build(key):
    if key['kind'] == 'sim':
        band, T, _, _ = PB.sim_band(key['cell'], key['seed'])
    elif key['kind'] == 'mesh':
        band, T, _, _ = PB.meshrir_band(DATA['mesh'], key['source'], key['band'], key['M'], key['seed'])
    else:
        band, T, _, _ = PB.heldout_band(DATA['held'], DATA['held_meta'], key['source'], key['band'], key['M'], key['seed'])
    return band, T


def relmax(new, old):
    new, old = np.asarray(new, float), np.asarray(old, float)
    return float(np.max(np.abs(new - old) / np.maximum(np.abs(old), 1e-300)))


def kernel_accuracy(band, rm):
    """Per ball and bin: trace-normalized 33-node microphone covariance against the trace-normalized exact series."""
    out = []
    for j, (c, R) in enumerate(zip(rm['centres'], rm['radii'])):
        dmin = float(np.min(np.linalg.norm(band.x - c, axis=1)))
        nodes = c[None, :] + R * E.BALL
        tgt_node = float(cdist(band.q, nodes).min())
        rec = dict(ball=j, radius=float(R), mic_clearance=dmin - float(R), target_to_node_min=tgt_node, bins=[])
        if dmin <= R:
            rec['series_invalid'] = True
            out.append(rec); continue
        for p in band.probs:
            F33 = E.region_factor(p.x, c, R, p.k)
            K33 = F33 @ F33.conj().T
            Ks = CD.series_K(p.x, c, R, p.k)
            K33n = K33 / np.mean(np.real(np.diag(K33))); Ksn = Ks / np.mean(np.real(np.diag(Ks)))
            dg = np.real(np.diag(K33n)) / np.real(np.diag(Ksn)) - 1
            rec['bins'].append(dict(kR=float(p.k * R), frob=float(np.linalg.norm(K33n - Ksn) / np.linalg.norm(Ksn)),
                                    diag_max=float(np.max(np.abs(dg)))))
        out.append(rec)
    return out


def one(task):
    t0 = time.process_time()
    key, stored = task['key'], task['stored']
    try:
        band, T = build(key)
        row = dict(study=task['study'], key=key)
        # B1: the adaptive stand-in, registered (traced) and corrected
        reg, cor, trace, fin, fin_cor = [], [], [], [], []
        for f, p in enumerate(band.probs):
            a = AF.adk_fit_variant(p, AF.adk_objective_traced)
            b = AF.adk_fit_variant(p, AF.adk_objective_masked)
            reg.append(E.nmse(a['pred'], T[:, f])); cor.append(E.nmse(b['pred'], T[:, f]))
            trace.append(a['trace']); fin.append(a['final_clipped']); fin_cor.append(b['final_clipped'])
        row['adk'] = dict(registered=reg, corrected=cor, trace=trace, final_clipped=fin, final_clipped_corrected=fin_cor,
                          gate_rel=relmax(reg, stored['nmse']['adk_standin']))
        t1 = time.process_time()
        # B2: the proposed estimator's fitted state
        models = v2.fit_band(band, 5)
        perK = {K: [E.nmse(v2.predict_points(band, m, f), T[:, f]) for f in range(band.F)] for K, m in models.items()}
        gate = {f'j_K{K}': relmax(v, stored['nmse'][f'j_K{K}']) for K, v in perK.items()}
        K = int(stored['K_cv'])
        clear_pts = {str(k): (float(cdist(band.x, m['S']).min()) if len(m['S']) else None) for k, m in models.items()}
        st = dict(K_cv=K, points={str(k): m['S'].tolist() for k, m in models.items() if k > 0}, point_clearance=clear_pts)
        t2 = time.process_time()
        if K == 0:
            prim = perK[0]
            st['balls'] = None
        else:
            rm = v2.extend(band, models[K], K)
            prim = [E.nmse(v2.predict_regions(band, rm, f), T[:, f]) for f in range(band.F)]
            st['balls'] = dict(centres=rm['centres'].tolist(), radii=rm['radii'].tolist(), lw=rm['lw'].tolist(),
                               ridge=rm['R'].tolist(), evidence=rm['evidence'],
                               clearance=[float(np.min(np.linalg.norm(band.x - c, axis=1)) - r)
                                          for c, r in zip(rm['centres'], rm['radii'])])
            t3 = time.process_time()
            st['kernel'] = kernel_accuracy(band, rm)
            if all(not k.get('series_invalid') for k in st['kernel']):
                st['frozen_exact'] = [E.nmse(CD.region_predict_series(band.probs[f], rm['centres'], rm['radii'], rm['lw'][f], rm['R'][f]),
                                             T[:, f]) for f in range(band.F)]
            else:
                st['frozen_exact'] = None
            st['seconds_kernel_checks'] = round(time.process_time() - t3, 2)
        gate['j_cv_ext'] = relmax(prim, stored['nmse']['j_cv_ext'])
        st['j_cv_ext'] = prim
        row['proposed'] = st
        row['gate_rel'] = gate
        row['seconds'] = dict(adk=round(t1 - t0, 2), fit_band=round(t2 - t1, 2), total=round(time.process_time() - t0, 2))
        return row
    except Exception:
        return dict(study=task['study'], key=key, error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--limit', type=int, default=None); ap.add_argument('--only', default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    p = subprocess.run([sys.executable, '-B', str(HERE / 'checks.py'), str(out / 'known-answer.json')], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit('known-answer checks failed:\n' + p.stdout + p.stderr)
    T = registered_rows()
    if a.only:
        T = [t for t in T if t['key']['kind'] == a.only]
    import sim
    size = lambda t: (0 if t['key']['kind'] != 'sim' else 1, -(t['key']['M'] if t['key']['M'] else sim.cells()[t['key']['cell']]['M']))
    T = sorted(T, key=size)                      # measured first, largest problems first
    if a.limit:
        T = T[:a.limit]
    commit = subprocess.run(['git', '-C', str(S), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(S), 'status', '--porcelain', '--', str(S)], capture_output=True, text=True).stdout.strip()
    (out / 'started.json').write_text(json.dumps(dict(tasks=len(T), workers=a.workers, commit=commit, study_tree_dirty=bool(dirty),
                                                      started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                                      inputs={n: sha(S / n) for n in ('studies/TB-M2/primary/rows.jsonl', 'studies/TB-S2/primary/rows.jsonl',
                                                                                     'common/estimators.py', 'common/v2.py', 'common/problems_v2.py',
                                                                                     'common/sim.py', 'V2-dev/cubature_diag.py',
                                                                                     'posthoc/review3/adk_fix.py', 'posthoc/review3/refit.py')}), indent=1))
    t0 = time.time(); n_err = 0
    with Pool(a.workers, initializer=init) as pool, open(out / 'rows.jsonl', 'w') as fh:
        for n, row in enumerate(pool.imap_unordered(one, T), 1):
            n_err += 'error' in row
            fh.write(json.dumps(row, default=float) + '\n'); fh.flush()
            if n % 50 == 0 or n == len(T):
                print(f'{n}/{len(T)} tasks, {time.time() - t0:.0f} s, errors {n_err}', flush=True)
    (out / 'receipt.json').write_text(json.dumps(dict(tasks=len(T), errors=n_err, elapsed_seconds=round(time.time() - t0, 1),
                                                      outputs={q.name: sha(q) for q in sorted(out.iterdir()) if q.is_file() and q.name != 'receipt.json'}),
                                                 indent=1))
    print('done', n_err, 'errors')


if __name__ == '__main__':
    main()
