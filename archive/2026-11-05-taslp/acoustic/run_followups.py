"""Registered runner for the two TB-M2 follow-up studies (registered 2026-11-05, after TB-M2).

TB-M2-SBLFIX: the two sparse Bayesian comparators rerun with the corrected guard (common/sbl_guard.py) on the 1,024
  MeshRIR problems of TB-M2 (same seeds, same microphones, same bins); every other method is taken from the
  registered TB-M2 rows.
TB-M2-RIB: the official Ribeiro et al. 2024 adaptive kernel (comparators/ribeiro2024, the authors' Julia library at
  commit 7a1e5c9b, training schedule of its SPM example) on 48 MeshRIR problems: bands 125/250/500/1000 Hz, M in {24, 48},
  draw 0, six loudspeakers fixed by a seeded draw (numpy default_rng(403500), 6 of 32 without replacement), bin index 2
  of each band.
Capsule mode as run_v2.py. A wall-time limit stops new problems once exceeded; unfinished problems are recorded as not
run (never replaced). Box only (the official library needs the Julia environment of /root/taslp/ribeiro2024).

source env.sh && nice -n 10 python -B run_followups.py --study TB-M2-SBLFIX|TB-M2-RIB --protocol-commit <sha> \
    --capsule-root <capsule> --frozen-sha256 <sha> --cross-check <json> --work /root/taslp --output studies/<study>/primary
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'common')); sys.path.insert(0, str(HERE / 'comparators/ribeiro2024')); sys.path.insert(0, str(HERE))
import release  # noqa: E402
import estimators as E  # noqa: E402
import problems_v2 as PB  # noqa: E402
import run_v2  # noqa: E402

LIMIT_S = {'TB-M2-SBLFIX': 2700, 'TB-M2-RIB': 2700}         # no new problem after 45 min of wall time
CALL_TIMEOUT_S = 1500                                        # Ribeiro: one bridge call at most 25 min
WORKERS = {'TB-M2-SBLFIX': 120, 'TB-M2-RIB': 15}             # Ribeiro: 15 Julia processes x 8 threads
RIB_SPEAKERS = sorted(int(s) for s in np.random.default_rng(403500).choice(32, 6, replace=False))
PREP = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tasks(study):
    mesh = [t for t in run_v2.tasks('TB-M2') if t[0] == 'mesh']
    if study == 'TB-M2-SBLFIX':
        return mesh
    return [t for t in mesh if t[1] in RIB_SPEAKERS and t[3] in (24, 48) and t[4] == 0]


def init():
    z = np.load(HERE / 'B3-measured/meshrir-prep/meshrir-prep.npz', allow_pickle=False)
    PREP.update({k: z[k] for k in z.files})


def one(study, task):
    t0 = time.time()
    try:
        _, spk, fc, M, d, seed = task
        band, T, ref, ts = PB.meshrir_band(PREP, spk, fc, M, seed)
        row = dict(kind='mesh', source=spk, band=fc, M=M, draw=d, seed=seed)
        if study == 'TB-M2-SBLFIX':
            import sbl_guard as G
            st = []
            row['nmse'] = {m: [E.nmse(G.sbl_guarded(band.probs[f], rule, st)['pred'], T[:, f]) for f in range(band.F)]
                           for m, rule in (('sbl_em_guard', 'em'), ('sbl_mackay_guard', 'mackay'))}
            row['dictionary'] = st[:band.F]
        else:
            import ribeiro_bridge as RB
            f = 2; p = band.probs[f]
            pred, info = RB.predict(p.x, p.y, p.q, p.f, p.c, seed=seed, return_info=True, timeout=CALL_TIMEOUT_S)
            row.update(bin=f, frequency=p.f, nmse={'ribeiro_official': E.nmse(pred, T[:, f])}, fit_seconds=info['fit_seconds'],
                       train_residual=info['train_residual'])
        row['wall_seconds'] = round(time.time() - t0, 1)
        return row
    except Exception:
        return dict(task=list(task), error=traceback.format_exc(), wall_seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', required=True, choices=('TB-M2-SBLFIX', 'TB-M2-RIB'))
    ap.add_argument('--protocol-commit', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--capsule-root', required=True); ap.add_argument('--frozen-sha256', required=True)
    ap.add_argument('--cross-check', required=True); ap.add_argument('--work', default='/root/taslp')
    a = ap.parse_args()
    rel = release.check(a.protocol_commit, HERE, capsule_root=a.capsule_root, frozen_sha256=a.frozen_sha256, work=a.work)
    xc = json.loads(Path(a.cross_check).read_text())
    if not xc.get('passed'):
        raise SystemExit('cross-machine check did not pass')
    if a.study == 'TB-M2-RIB':      # the box's Julia environment must be the frozen one
        frozen = sha(HERE / 'comparators/ribeiro2024/Manifest.toml'); live = sha(Path(a.work) / 'ribeiro2024/env/Manifest.toml')
        if frozen != live:
            raise SystemExit(f'Julia environment differs from the frozen Manifest.toml ({live} != {frozen})')
    out = HERE / a.output; out.mkdir(parents=True, exist_ok=False)
    T = tasks(a.study); T = sorted(T, key=lambda t: -t[3])
    (out / 'started.json').write_text(json.dumps(dict(study=a.study, protocol_commit=a.protocol_commit, tasks=len(T),
                                                      workers=WORKERS[a.study], wall_limit_s=LIMIT_S[a.study],
                                                      ribeiro_speakers=RIB_SPEAKERS if a.study == 'TB-M2-RIB' else None,
                                                      started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), release=rel,
                                                      cross_machine_check_reused=dict(path=a.cross_check, passed=xc['passed'])),
                                                 indent=1, default=str))
    t0 = time.time(); n_err = 0; done = 0; not_run = []
    with ProcessPoolExecutor(WORKERS[a.study], initializer=init) as ex, open(out / 'rows.jsonl', 'w') as fh:
        it = iter(T); live = {}
        for _ in range(WORKERS[a.study]):
            t = next(it, None)
            if t is not None:
                live[ex.submit(one, a.study, t)] = t
        while live:
            fut = next(as_completed(live)); live.pop(fut)
            row = fut.result(); n_err += 'error' in row; done += 1
            fh.write(json.dumps(row, default=float) + '\n'); fh.flush()
            if time.time() - t0 < LIMIT_S[a.study]:
                t = next(it, None)
                if t is not None:
                    live[ex.submit(one, a.study, t)] = t
            if done % 50 == 0:
                print(f'{done}/{len(T)} tasks, {time.time() - t0:.0f} s, errors {n_err}', flush=True)
        not_run = [list(t) for t in it]
    (out / 'failure.json').write_text(json.dumps(dict(n_errors=n_err, not_run_after_wall_limit=not_run), indent=1))
    outputs = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'run-receipt.json'}
    rec = dict(study=a.study, status='complete' if not n_err and not not_run else 'partial', protocol_commit=a.protocol_commit,
               elapsed_seconds=round(time.time() - t0, 1), n_tasks=len(T), n_done=done, n_errors=n_err, n_not_run=len(not_run),
               release_mode=rel['mode'], frozen_inputs=rel['frozen_inputs'], frozen_inputs_json_sha256=rel['frozen_inputs_json_sha256'],
               capsule_listing_sha256=rel['capsule_listing_sha256'], machine=rel['machine'],
               cross_machine_check=dict(passed=xc['passed'], tasks=xc.get('tasks'), reused_from=a.cross_check), outputs=outputs)
    (out / 'run-receipt.json').write_text(json.dumps(rec, indent=1, default=str))
    print('done', rec['status'], rec['elapsed_seconds'])


if __name__ == '__main__':
    main()
