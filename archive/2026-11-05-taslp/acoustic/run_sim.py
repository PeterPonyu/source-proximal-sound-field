"""Released runner for TB-S1 (registered simulation suite, C2).

A production invocation requires the preregistration commit: every file in FROZEN-INPUTS.json must be
byte-identical to that ancestor commit and match its recorded SHA-256; dependency versions, the one-thread cap and
nice 10 are checked. The known-answer suite runs first and must pass. Outputs are never overwritten. No outcome
summary is printed during the run; analysis is a separate registered script (analyze_sim.py).

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python3 -B run_sim.py \
    --protocol-commit <sha> --output studies/TB-S1/primary --workers 22
"""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE / 'common'))
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import estimators as E  # noqa: E402
import methods  # noqa: E402
import sim  # noqa: E402

SEEDS = range(401000, 401096)
REMAINDER_CELLS = ('base',) + tuple(c for c in sim.cells() if c.startswith('A3-'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_release(commit):
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('A full lowercase preregistration commit hash is required')
    subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', commit, 'HEAD'], check=True)
    manifest = json.loads((HERE / 'FROZEN-INPUTS.json').read_text())
    for rel in ['archive/2026-11-05-taslp/acoustic/FROZEN-INPUTS.json'] + list(manifest['files']):
        frozen = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{rel}'])
        if (ROOT / rel).read_bytes() != frozen:
            raise RuntimeError(f'Unfrozen file: {rel}')
    for rel, expected in manifest['files'].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f'Input SHA mismatch: {rel}')
    actual = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    if actual != manifest['dependency_versions']:
        raise RuntimeError(f'Dependency versions changed: {actual}')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(name) != '1':
            raise RuntimeError(f'Thread cap missing: {name}')
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError('Run requires nice -n 10')
    return manifest


def one(task):
    name, seed = task
    cell = sim.cells()[name]
    t0 = time.process_time()
    try:
        fd = sim.build(cell, seed)
        prob = E.Problem(fd['x'], fd['y'], fd['q'], fd['f'], c=fd['c_assumed'], box=fd['box'], room=fd['room'])
        out, info, timing = methods.evaluate(prob, fd['truth'], fd['sources'], seed, room=fd['room'])
        row = dict(cell=name, seed=seed, f=fd['f'], M=fd['M'], n_targets=len(fd['q']), **{f'nmse_{k}': v for k, v in out.items()})
        if name in REMAINDER_CELLS:
            rp = E.Problem(fd['x'], fd['remainder_mics'], fd['q'], fd['f'], c=fd['c_assumed'], box=fd['box'])
            row['nmse_remainder_sinc'] = E.nmse(E.sinc_fit(rp)['pred'], fd['remainder_targets'])
        row.update({k: v for k, v in info.items()})
        row['process_seconds'] = round(time.process_time() - t0, 3)
        row['timing'] = {k: round(v, 3) for k, v in timing.items()}
        row['primary_source'] = fd['primary'].tolist()
        row['region_origin'] = fd['origin'].tolist()
        return row
    except Exception:
        return dict(cell=name, seed=seed, error=traceback.format_exc())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol-commit', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--workers', type=int, default=22)
    a = ap.parse_args()
    manifest = check_release(a.protocol_commit)
    out = (HERE / a.output) if not Path(a.output).is_absolute() else Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    started = dict(protocol_commit=a.protocol_commit, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   cells=len(sim.cells()), seeds=[SEEDS[0], SEEDS[-1]], workers=a.workers, host=platform.node())
    (out / 'started.json').write_text(json.dumps(started, indent=1))
    ka = subprocess.run([sys.executable, '-B', str(HERE / 'common/known_answers.py'), str(out / 'known-answer-checks.json')],
                        capture_output=True, text=True)
    if ka.returncode != 0:
        (out / 'failure.json').write_text(json.dumps(dict(stage='known answers', stdout=ka.stdout[-4000:]), indent=1))
        sys.exit(1)
    tasks = [(c, s) for s in SEEDS for c in sim.cells()]
    t0 = time.time()
    rows, errors = [], []
    with Pool(a.workers) as pool, open(out / 'partial-rows.jsonl', 'w') as part:
        for i, row in enumerate(pool.imap_unordered(one, tasks, chunksize=1)):
            part.write(json.dumps(row, default=float) + '\n'); part.flush()
            (errors if 'error' in row else rows).append(row)
            if (i + 1) % 256 == 0:
                print(f'{i + 1}/{len(tasks)} fields, {time.time() - t0:.0f} s', flush=True)
    if errors:
        (out / 'failure.json').write_text(json.dumps(dict(stage='fields', errors=errors[:20], n_errors=len(errors)), indent=1))
    rows.sort(key=lambda r: (list(sim.cells()).index(r['cell']), r['seed']))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / 'field-metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    receipt = dict(status='complete' if not errors else 'complete_with_errors', protocol_commit=a.protocol_commit,
                   elapsed_seconds=round(time.time() - t0, 1), n_rows=len(rows), n_errors=len(errors),
                   outputs={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'run-receipt.json'},
                   frozen_inputs=manifest['files'])
    (out / 'run-receipt.json').write_text(json.dumps(receipt, indent=1))
    print('done', receipt['status'], receipt['elapsed_seconds'])


if __name__ == '__main__':
    main()
