"""Cross-machine check, run on the rented box before any registered task (draft, 2026-11-05).

source env.sh && nice -n 10 python -B cross_check.py CAPSULE_STUDY_DIR OUT.json [--workers N]

1. The version-1 and version-2 known-answer suites must pass.
2. A fixed list of development tasks (`reference-rows.jsonl`, written on the laptop by make_capsule.py) is re-run with
   the same code, and every (task, method, bin) error is compared with the laptop value.
   Tolerance, calibrated before any registered run: exact agreement. On 2026-11-05 the 97 development tasks run on
   both the laptop (Intel Core Ultra 9 275HX) and the box (AMD EPYC 7B13) agreed bit for bit in all 23,568 errors,
   with identical order and family selections (V2-dev/dev1-box/cross-machine.json), under the pinned stack and
   OPENBLAS_CORETYPE=Haswell. Any difference above 1e-9 dB therefore means the environment differs; the check fails
   and lists the entries.
3. A pass is required before the registered runner starts; the result goes into the run receipt.
"""
from __future__ import annotations
import argparse
import json
import platform
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

TOL_DB, FRACTION, MEDIAN_DB = 1e-9, 1.0, 1e-9


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('study'); ap.add_argument('out'); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    S = Path(a.study).resolve()
    sys.path.insert(0, str(S / 'common')); sys.path.insert(0, str(S / 'V2-dev'))
    res = dict(started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), python=platform.python_version(), machine=platform.machine())
    ka = {}
    for name, script in (('v1', S / 'common/known_answers.py'), ('v2', S / 'common/v2_checks.py')):
        out = Path(a.out).with_suffix(f'.{name}-ka.json')
        p = subprocess.run([sys.executable, '-B', str(script), str(out)], capture_output=True, text=True)
        ok = p.returncode == 0 and json.loads(out.read_text()).get('passed', False)
        ka[name] = dict(passed=bool(ok), returncode=p.returncode)
    res['known_answers'] = ka
    import run_dev
    ref = [json.loads(l) for l in open(S / 'remote/reference-rows.jsonl')]
    tasks = [tuple(r['_task']) for r in ref]
    with Pool(a.workers, initializer=run_dev.init) as pool:
        rows = pool.map(run_dev.run_task, tasks)
    diffs = {}
    for r0, r1 in zip(ref, rows):
        if 'error' in r1:
            diffs.setdefault('_errors', []).append(r1['error'][-400:]); continue
        for m, v0 in r0['nmse'].items():
            v1 = r1['nmse'].get(m)
            if v1 is None:
                continue
            diffs.setdefault(m, []).extend(np.abs(10 * np.log10(np.asarray(v1) / np.asarray(v0))).tolist())
    per = {}
    for m, d in diffs.items():
        if m == '_errors':
            continue
        d = np.asarray(d)
        per[m] = dict(n=len(d), fraction_within=float(np.mean(d <= TOL_DB)), median_db=float(np.median(d)), max_db=float(d.max()),
                      passed=bool(np.mean(d <= TOL_DB) >= FRACTION))
    alld = np.concatenate([np.asarray(v) for k, v in diffs.items() if k != '_errors'])
    res['tasks'] = len(tasks); res['per_method'] = per; res['errors'] = diffs.get('_errors', [])
    res['median_db_all'] = float(np.median(alld))
    res['passed'] = bool(all(v['passed'] for v in ka.values()) and not res['errors'] and all(v['passed'] for v in per.values())
                         and res['median_db_all'] <= MEDIAN_DB)
    res['tolerance'] = dict(tol_db=TOL_DB, fraction=FRACTION, median_db=MEDIAN_DB)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print('CROSS-MACHINE CHECK', 'PASSED' if res['passed'] else 'FAILED', json.dumps({k: v for k, v in res.items() if k != 'per_method'}))


if __name__ == '__main__':
    main()
