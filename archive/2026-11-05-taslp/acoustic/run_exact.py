"""Registered runner for TB-M2-EXACT, the exact-kernel rerun of the two registered region-kernel comparators
(PROTOCOLS/TB-M2-EXACT.md).

`mo26_cap5` (v2.mo26_cap per bin) and `gsmf` (v2.gsmf_fit, predictions by v2.predict_regions), exactly as in TB-M2
(common/methods_v2.py), run twice on each registered TB-M2 MeshRIR problem with M <= 48 (32 loudspeakers x 4 bands x
M in {24, 48} x 2 draws = 512 problems): once as registered (the 33-node cubature of the uniform ball, which must
reproduce the registered TB-M2 rows bit for bit), once with the exact uniform-ball series (Anderson, Teal & Poletti
2016, eq. 40) swapped in for estimators.mo26_objective and estimators.region_predict by V2-dev/cubature_diag.exact_kernel
(the code of the descriptive diagnostic, imported unchanged). Same seeds, microphones, targets, bins, sound speeds and
scaling as TB-M2 (common/problems_v2.py); j_cv_ext comes from the registered TB-M2 rows.

Laptop (git mode):
  OPENBLAS_NUM_THREADS=1 nice -n 10 python3 -B run_exact.py --protocol-commit <sha> --root <repository root> \
      --output studies/TB-M2-EXACT/primary --workers 20
The output folder must not exist. The known-answer checks of the exact kernel (cubature_diag.checks: the series against
a dense uniform cubature, and the series objective's gradient against finite differences) run first and must pass.
Rows are streamed to rows.jsonl; an errored job is recorded with its traceback and never replaced. No new job starts
after the wall-time limit (4 h, counted after the known answers); jobs not started are recorded as not run.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'common')); sys.path.insert(0, str(HERE)); sys.path.insert(1, str(HERE / 'V2-dev'))
import release  # noqa: E402
import estimators as E  # noqa: E402
import problems_v2 as PB  # noqa: E402
import v2  # noqa: E402
import run_v2  # noqa: E402
import cubature_diag as CD  # noqa: E402

MESH_M = (24, 48)
WALL_LIMIT_S = 4 * 3600
MAX_WORKERS = 24
KA_LIMITS = dict(series_vs_dense=1e-4, objective_gradient_max_rel_err=1e-6)
PREP = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tasks():
    return [t for t in run_v2.tasks('TB-M2') if t[0] == 'mesh' and t[3] in MESH_M]


def init():
    warnings.simplefilter('ignore')
    z = np.load(HERE / 'B3-measured/meshrir-prep/meshrir-prep.npz', allow_pickle=False)
    PREP.update({k: z[k] for k in z.files})


def fit(band, T):
    """Both comparators under the registered cubature and under the exact series, in that order (TB-M2's call order)."""
    kmax = 2 * np.pi * float(band.freqs.max()) / band.c
    nmse, info, seconds = {}, {}, {}
    for tag, ctx in (('cubature', contextlib.nullcontext), ('exact', CD.exact_kernel)):
        with ctx():
            a = time.process_time()
            mo = [v2.mo26_cap(p) for p in band.probs]
            nmse[f'mo26_cap5_{tag}'] = [E.nmse(r['pred'], T[:, f]) for f, r in enumerate(mo)]
            seconds[f'mo26_cap5_{tag}'] = round(time.process_time() - a, 1)
            b = time.process_time()
            gs = v2.gsmf_fit(band)
            nmse[f'gsmf_{tag}'] = [E.nmse(v2.predict_regions(band, gs, f), T[:, f]) for f in range(band.F)]
            seconds[f'gsmf_{tag}'] = round(time.process_time() - b, 1)
            rm = [[float(x) for x in r['radii']] for r in mo]
            rg = [float(x) for x in gs['radii']]
            info[tag] = dict(mo26_cap5_radii=rm, gsmf_radii=rg, gsmf_regions=int(len(gs['radii']) - gs['pruned']),
                             k_max_R=dict(mo26_cap5=kmax * max([x for r in rm for x in r], default=0.0), gsmf=kmax * max(rg, default=0.0)))
    return nmse, info, seconds


def one(task):
    t0 = time.time()
    try:
        _, spk, fc, M, d, seed = task
        band, T, _, _ = PB.meshrir_band(PREP, spk, fc, M, seed)
        nmse, info, seconds = fit(band, T)
        return dict(kind='mesh', source=spk, band=fc, M=M, draw=d, seed=seed, c_used=band.c, F=band.F, M_mics=band.M,
                    nmse=nmse, info=info, seconds=seconds, wall_seconds=round(time.time() - t0, 1))
    except Exception:
        return dict(task=list(task), error=traceback.format_exc(), wall_seconds=round(time.time() - t0, 1))


def known_answers():
    warnings.simplefilter('ignore')
    import run_dev as RD
    RD.init()
    res = CD.checks()
    dense = {k: v['series_vs_dense'] for k, v in res.items() if isinstance(v, dict)}
    ok = all(v < KA_LIMITS['series_vs_dense'] for v in dense.values()) and \
        res['objective_gradient_max_rel_err'] < KA_LIMITS['objective_gradient_max_rel_err']
    return dict(checks=res, limits=KA_LIMITS, passed=bool(ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol-commit', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--workers', type=int, required=True); ap.add_argument('--root', required=True)
    a = ap.parse_args()
    if a.workers > MAX_WORKERS:
        raise SystemExit(f'at most {MAX_WORKERS} workers')
    rel = release.check(a.protocol_commit, HERE, root=a.root)
    out = (HERE / a.output) if not Path(a.output).is_absolute() else Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    ka = known_answers()
    (out / 'known-answer-exact.json').write_text(json.dumps(ka, indent=1, default=float))
    if not ka['passed']:
        raise SystemExit('known-answer checks of the exact kernel failed')
    T = sorted(tasks(), key=lambda t: (-t[3], t[2]))      # 48 microphones first, then by band
    (out / 'started.json').write_text(json.dumps(dict(study='TB-M2-EXACT', protocol_commit=a.protocol_commit, tasks=len(T),
                                                      workers=a.workers, wall_limit_s=WALL_LIMIT_S, nmax=CD.NMAX,
                                                      started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), release=rel), indent=1, default=str))
    t0 = time.time(); n_err = 0; done = 0                 # the wall-time limit counts from here, after the known answers
    with ProcessPoolExecutor(a.workers, initializer=init) as ex, open(out / 'rows.jsonl', 'w') as fh:
        it = iter(T); live = {}
        for _ in range(a.workers):
            t = next(it, None)
            if t is not None:
                live[ex.submit(one, t)] = t
        while live:
            fut = next(as_completed(live)); live.pop(fut)
            row = fut.result(); n_err += 'error' in row; done += 1
            fh.write(json.dumps(row, default=float) + '\n'); fh.flush()
            if time.time() - t0 < WALL_LIMIT_S:
                t = next(it, None)
                if t is not None:
                    live[ex.submit(one, t)] = t
            if done % 32 == 0 or not live:
                print(f'{done}/{len(T)} jobs, {time.time() - t0:.0f} s, errors {n_err}', flush=True)
        not_run = [list(t) for t in it]
    (out / 'failure.json').write_text(json.dumps(dict(n_errors=n_err, not_run_after_wall_limit=not_run), indent=1))
    outputs = {q.name: sha(q) for q in sorted(out.iterdir()) if q.is_file() and q.name != 'run-receipt.json'}
    rec = dict(study='TB-M2-EXACT', status='complete' if not n_err and not not_run else 'partial', protocol_commit=a.protocol_commit,
               elapsed_seconds=round(time.time() - t0, 1), n_jobs=len(T), wall_limit_s=WALL_LIMIT_S, n_done=done, n_errors=n_err,
               n_not_run=len(not_run), known_answers=dict(exact=ka['passed']), cross_machine_check=dict(passed=True, reference_machine=True),
               release_mode=rel['mode'], frozen_inputs=rel['frozen_inputs'], frozen_inputs_json_sha256=rel['frozen_inputs_json_sha256'],
               capsule_listing_sha256=rel['capsule_listing_sha256'], machine=rel['machine'], outputs=outputs)
    (out / 'run-receipt.json').write_text(json.dumps(rec, indent=1, default=str))
    print('done', rec['status'], rec['elapsed_seconds'])


if __name__ == '__main__':
    main()
