"""Registered runner for the version-2 studies TB-S2 (simulation) and TB-M2 (measured).

Laptop (git mode):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python3 -B run_v2.py --study TB-S2 \
      --protocol-commit <sha> --root <repository root> --output studies/TB-S2/primary --workers N
Box (capsule mode, no .git): add --capsule-root <capsule> --frozen-sha256 <sha of FROZEN-INPUTS.json at the commit>
  --cross-check <cross_check.py output, must have passed> instead of --root.
The output folder must not exist. Both known-answer suites run first and must pass. Every task's row is streamed to
rows.jsonl; an errored task is recorded with its traceback and never replaced. run-receipt.json records the release
check (commit, frozen inputs, machine), the cross-machine check and the SHA-256 of every output.
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
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'common'))
import release  # noqa: E402
import sim  # noqa: E402
import problems_v2 as PB  # noqa: E402
import methods_v2 as MV  # noqa: E402

DECL = json.loads((HERE / 'SEEDS.json').read_text())['declared']
BANDS = (125.0, 250.0, 500.0, 1000.0)
DATA = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tasks(study):
    if study == 'TB-S2':
        lo, hi = DECL['TB-S2_simulation_fields']
        return [('sim', c, s) for c in sim.cells() for s in range(lo, hi + 1)]
    lo, hi = DECL['TB-M2_measured_draws']
    seeds = iter(range(lo, hi + 1))
    T = []
    for spk in range(32):
        for M in (24, 48, 96, 192):
            for d in range(2):
                s = next(seeds)
                T += [('mesh', spk, fc, M, d, s) for fc in BANDS]
    for spk in ('S4', 'S7', 'S8'):
        for M in (24, 48, 96, 192):
            for d in range(2):
                s = next(seeds)
                T += [('held', spk, fc, M, d, s) for fc in BANDS]
    assert next(seeds, None) is None, 'seed block size does not match the task list'
    return T


def init(study):
    if study == 'TB-M2':
        z = np.load(HERE / 'B3-measured/meshrir-prep/meshrir-prep.npz', allow_pickle=False)
        DATA['mesh'] = {k: z[k] for k in z.files}
        z = np.load(HERE / 'B3-measured/heldout-prep/heldout.npz', allow_pickle=False)
        DATA['held'] = {k: z[k] for k in z.files}
        DATA['held_meta'] = json.loads((HERE / 'B3-measured/heldout-prep/heldout.json').read_text())


def run_task(task):
    t0 = time.process_time()
    try:
        kind = task[0]
        if kind == 'sim':
            _, cell, seed = task
            band, T, sources, primary = PB.sim_band(cell, seed)
            out, info, timing = MV.evaluate(band, T, sources, primary, seed)
            row = dict(kind=kind, cell=cell, seed=seed)
        else:
            _, spk, fc, M, d, seed = task
            if kind == 'mesh':
                build = lambda: PB.meshrir_band(DATA['mesh'], spk, fc, M, seed)
            else:
                build = lambda: PB.heldout_band(DATA['held'], DATA['held_meta'], spk, fc, M, seed)
            band, T, ref, ts = build()
            out, info, timing = MV.evaluate(band, T, ref, ref[0], seed, measured=True, two_stage=ts,
                                            band_factory=lambda: build()[0])
            row = dict(kind=kind, source=spk, band=fc, M=M, draw=d, seed=seed, c_used=band.c, two_stage_ok=bool(ts))
        row.update(F=band.F, M_mics=band.M, nmse=out, info=info, timing=timing, process_seconds=round(time.process_time() - t0, 1))
        return row
    except Exception:
        return dict(task=list(task), error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--study', required=True, choices=('TB-S2', 'TB-M2'))
    ap.add_argument('--protocol-commit', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--workers', type=int, required=True)
    ap.add_argument('--root'); ap.add_argument('--capsule-root'); ap.add_argument('--frozen-sha256'); ap.add_argument('--cross-check')
    ap.add_argument('--work', default=None, help='box work directory holding machine.txt')
    a = ap.parse_args()
    if a.workers > 120:
        raise SystemExit('at most 120 workers')
    rel = release.check(a.protocol_commit, HERE, root=a.root, capsule_root=a.capsule_root, frozen_sha256=a.frozen_sha256, work=a.work)
    xc = None
    if a.capsule_root:
        xc = json.loads(Path(a.cross_check).read_text())
        if not xc.get('passed'):
            raise SystemExit('cross-machine check did not pass')
    out = (HERE / a.output) if not Path(a.output).is_absolute() else Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    ka = {}
    for name, script in (('v1', HERE / 'common/known_answers.py'), ('v2', HERE / 'common/v2_checks.py')):
        p = subprocess.run([sys.executable, '-B', str(script), str(out / f'known-answer-{name}.json')], capture_output=True, text=True)
        ka[name] = json.loads((out / f'known-answer-{name}.json').read_text()).get('passed', False) and p.returncode == 0
    if not all(ka.values()):
        (out / 'failure.json').write_text(json.dumps(dict(stage='known_answers', result=ka)))
        raise SystemExit(f'known-answer suite failed: {ka}')
    T = tasks(a.study)
    T = sorted(T, key=lambda t: -(sim.cells()[t[1]]['M'] if t[0] == 'sim' else t[3]))   # largest problems first
    t0 = time.time()
    (out / 'started.json').write_text(json.dumps(dict(study=a.study, protocol_commit=a.protocol_commit, workers=a.workers, tasks=len(T),
                                                      started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), release=rel,
                                                      cross_machine_check=xc), indent=1, default=str))
    n_err = 0
    with Pool(a.workers, initializer=init, initargs=(a.study,)) as pool, open(out / 'rows.jsonl', 'w') as fh:
        for n, row in enumerate(pool.imap_unordered(run_task, T), 1):
            n_err += 'error' in row
            fh.write(json.dumps(row, default=float) + '\n'); fh.flush()
            if n % 50 == 0 or n == len(T):
                print(f'{n}/{len(T)} tasks, {time.time() - t0:.0f} s, errors {n_err}', flush=True)
    errs = [json.loads(l) for l in open(out / 'rows.jsonl') if '"error"' in l and json.loads(l).get('error')]
    (out / 'failure.json').write_text(json.dumps(dict(stage='tasks', n_errors=len(errs), errors=errs), indent=1))
    outputs = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'run-receipt.json'}
    receipt = dict(study=a.study, status='complete' if not errs else 'complete_with_errors', protocol_commit=a.protocol_commit,
                   elapsed_seconds=round(time.time() - t0, 1), n_tasks=len(T), n_errors=len(errs), known_answers=ka,
                   release_mode=rel['mode'], frozen_inputs=rel['frozen_inputs'], frozen_inputs_json_sha256=rel['frozen_inputs_json_sha256'],
                   capsule_listing_sha256=rel['capsule_listing_sha256'], machine=rel['machine'],
                   cross_machine_check=None if xc is None else dict(passed=xc['passed'], tasks=xc.get('tasks'),
                                                                     median_db_all=xc.get('median_db_all')),
                   outputs=outputs)
    (out / 'run-receipt.json').write_text(json.dumps(receipt, indent=1, default=str))
    print('done', receipt['status'], receipt['elapsed_seconds'])


if __name__ == '__main__':
    main()
