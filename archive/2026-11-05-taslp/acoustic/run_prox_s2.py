"""Registered runner for TB-PROX-S2, mo26-prox with the JASA 2026 paper's own stage-2 stop (PROTOCOLS/TB-PROX-S2.md).

Identical to run_prox.py (TB-PROX) except that only the mo26 variant runs, as common/prox2_s2.mo26_prox_s2: the
registered prox2.mo26_prox with its stage-2 stop on the unpenalised NLML (the paper's eqs. 26 and 68). Same 128
problems (the TB-M2 MeshRIR problems with M = 24 and draw 0, 32 loudspeakers x 4 bands), one job per problem; every
other method is taken from the registered TB-M2 rows and the registered-stop reading from the TB-PROX rows.

Box (capsule mode, the registered machine):
  source /root/taslp/env.sh && nice -n 10 python -B run_prox_s2.py --protocol-commit <sha> --capsule-root <capsule> \
      --frozen-sha256 <sha> --cross-check /root/taslp/xcheck.json --work /root/taslp --output studies/TB-PROX-S2/primary --workers N
Laptop (git mode): replace the capsule arguments with --root <repository root>.
The output folder must not exist. The known-answer suites (common/prox2_checks.py, as for TB-PROX, on a box its fast
correctness checks only; then common/prox2_s2_checks.py, the checks of the change) run first and must pass. Rows are
streamed to rows.jsonl; an errored job is recorded with its traceback and never replaced. No new job starts after the
wall-time limit (box 1.25 h, laptop WALL_GIT_H h, counted after the known answers); jobs not started are recorded as not run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'common')); sys.path.insert(0, str(HERE))
import release  # noqa: E402
import estimators as E  # noqa: E402
import problems_v2 as PB  # noqa: E402
import prox2 as PX  # noqa: E402
import prox2_s2 as S2  # noqa: E402
import warnings  # noqa: E402
import run_v2  # noqa: E402

MESH_M = (24,)
VARIANTS = ('mo26',)
WALL_GIT_H = 6
WALL_LIMIT_S = dict(capsule=int(1.25 * 3600), git=WALL_GIT_H * 3600)    # no new job after this wall time
MAX_WORKERS = 120
PREP = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tasks():
    mesh = [t for t in run_v2.tasks('TB-M2') if t[0] == 'mesh' and t[3] in MESH_M and t[4] == 0]
    return [t + (v,) for v in VARIANTS for t in mesh]


def init():
    warnings.simplefilter('ignore')
    z = np.load(HERE / 'B3-measured/meshrir-prep/meshrir-prep.npz', allow_pickle=False)
    PREP.update({k: z[k] for k in z.files})


def one(job):
    t0 = time.time()
    try:
        task, variant = job[:-1], job[-1]
        _, spk, fc, M, d, seed = task
        band, T, _, _ = PB.meshrir_band(PREP, spk, fc, M, seed)
        row = dict(kind='mesh', source=spk, band=fc, M=M, draw=d, seed=seed, c_used=band.c, variant=variant, F=band.F, M_mics=band.M)
        a = time.process_time()
        prep = PX.Prepared(band, PX.MO26)
        dic = dict(S=int(len(prep.cen)), delta=prep.delta, rho0=prep.rho, shrunk=int(np.sum(prep.rad < prep.rho - 1e-12)),
                   delta_wavelengths=[prep.delta * k / (2 * np.pi) for k in prep.ks])
        mo = [S2.mo26_prox_s2(prep, f) for f in range(band.F)]
        row.update(nmse={'mo26_prox_s2': [E.nmse(r['pred'], T[:, f]) for f, r in enumerate(mo)]}, info=dict(dictionary=dic, bins=[r['info'] for r in mo]))
        row.update(process_seconds=round(time.process_time() - a, 1), wall_seconds=round(time.time() - t0, 1))
        return row
    except Exception:
        return dict(task=list(job), error=traceback.format_exc(), wall_seconds=round(time.time() - t0, 1))


def size(job):
    """By band, lowest first (as TB-PROX within its mo26 jobs)."""
    return -job[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol-commit', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--workers', type=int, required=True)
    ap.add_argument('--root'); ap.add_argument('--capsule-root'); ap.add_argument('--frozen-sha256'); ap.add_argument('--work')
    ap.add_argument('--cross-check', help='box only: the passed cross_check.py output of that box')
    a = ap.parse_args()
    if a.workers > MAX_WORKERS:
        raise SystemExit(f'at most {MAX_WORKERS} workers')
    rel = release.check(a.protocol_commit, HERE, root=a.root, capsule_root=a.capsule_root, frozen_sha256=a.frozen_sha256, work=a.work)
    if a.capsule_root:
        xc = json.loads(Path(a.cross_check).read_text())
        if not xc.get('passed'):
            raise SystemExit('cross-machine check did not pass')
        xc = dict(passed=True, tasks=xc.get('tasks'), median_db_all=xc.get('median_db_all'), path=a.cross_check)
    else:      # the laptop is the reference machine of every exact cross-machine check (V2-dev/dev1 rows)
        xc = dict(passed=True, reference_machine=True)
    out = (HERE / a.output) if not Path(a.output).is_absolute() else Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    mode = 'capsule' if a.capsule_root else 'git'
    ka_args = [str(out / 'known-answer-prox.json')] + (['--fast'] if mode == 'capsule' else [])
    p = subprocess.run([sys.executable, '-B', '-W', 'ignore', str(HERE / 'common/prox2_checks.py')] + ka_args,
                       capture_output=True, text=True)
    ka = p.returncode == 0 and json.loads((out / 'known-answer-prox.json').read_text()).get('passed', False)
    if not ka:
        (out / 'failure.json').write_text(json.dumps(dict(stage='known_answers', stdout=p.stdout, stderr=p.stderr)))
        raise SystemExit('known-answer suite failed')
    p2 = subprocess.run([sys.executable, '-B', '-W', 'ignore', str(HERE / 'common/prox2_s2_checks.py'), str(out / 'known-answer-prox-s2.json')],
                        capture_output=True, text=True)
    ka2 = p2.returncode == 0 and json.loads((out / 'known-answer-prox-s2.json').read_text()).get('passed', False)
    if not ka2:
        (out / 'failure.json').write_text(json.dumps(dict(stage='known_answers_s2', stdout=p2.stdout, stderr=p2.stderr)))
        raise SystemExit('known-answer suite of the change failed')
    T = sorted(tasks(), key=lambda t: -size(t))
    settings = {k: getattr(PX, k) for k in ('GSMF', 'MO26', 'ICASSP_POWER', 'ICASSP_K0', 'JASA_POWER', 'JASA_K0', 'SIGMA_RATIO',
                                            'DICT_SPACING', 'DICT_MARGIN', 'S_MAX', 'SINC', 'RAD_BOUNDS', 'NMAX', 'N_S', 'FINE_QUAD')}
    settings['stage2_stop'] = 'unpenalised NLML rises (JASA 2026 eqs. 26 and 68; common/prox2_s2.py)'
    (out / 'started.json').write_text(json.dumps(dict(study='TB-PROX-S2', protocol_commit=a.protocol_commit, tasks=len(T),
                                                      workers=a.workers, wall_limit_s=WALL_LIMIT_S[mode], settings=settings,
                                                      started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), release=rel),
                                                 indent=1, default=str))
    t0 = time.time(); n_err = 0; done = 0          # the wall-time limit counts from here, after the known answers
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
            if time.time() - t0 < WALL_LIMIT_S[mode]:
                t = next(it, None)
                if t is not None:
                    live[ex.submit(one, t)] = t
            if done % 20 == 0 or not live:
                print(f'{done}/{len(T)} jobs, {time.time() - t0:.0f} s, errors {n_err}', flush=True)
        not_run = [list(t) for t in it]
    (out / 'failure.json').write_text(json.dumps(dict(n_errors=n_err, not_run_after_wall_limit=not_run), indent=1))
    outputs = {q.name: sha(q) for q in sorted(out.iterdir()) if q.is_file() and q.name != 'run-receipt.json'}
    rec = dict(study='TB-PROX-S2', status='complete' if not n_err and not not_run else 'partial', protocol_commit=a.protocol_commit,
               elapsed_seconds=round(time.time() - t0, 1), n_jobs=len(T), wall_limit_s=WALL_LIMIT_S[mode], n_done=done, n_errors=n_err, n_not_run=len(not_run),
               known_answers=dict(prox=ka, prox_s2=ka2), settings=settings, cross_machine_check=xc, release_mode=rel['mode'], frozen_inputs=rel['frozen_inputs'],
               frozen_inputs_json_sha256=rel['frozen_inputs_json_sha256'], capsule_listing_sha256=rel['capsule_listing_sha256'],
               machine=rel['machine'], outputs=outputs)
    (out / 'run-receipt.json').write_text(json.dumps(rec, indent=1, default=str))
    print('done', rec['status'], rec['elapsed_seconds'])


if __name__ == '__main__':
    main()
