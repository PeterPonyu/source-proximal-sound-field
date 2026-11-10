"""TB-M4-MIRACLE-A2 runner (PROTOCOL.md). The TB-M3 runner's problems, builder and method set, imported unchanged
(with its two corrections), plus the multi-frequency shared-support SBL comparator (mfsbl_em, mfsbl_mackay).
A2 uses A1's seeds (matched design): the same seed gives the same microphone subset and the same internal draws.

All commands: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python3 -B run_m4.py ...
  run    EXTRACT_DIR OUT --workers N [--resume]
         The full A2 run (768 problems), started at once. EXTRACT_DIR holds miracle-A2.npz and its receipt from
         python3 -I -B ../TB-M3-MIRACLE/miracle_prep.py A2 A2.h5 bb9d5f12d39f53eadf9c448348d51036 EXTRACT_DIR
         --resume appends to OUT/rows.jsonl and skips every task already present, error rows included.
  dry    A1_EXTRACT_DIR OUT --workers N [--dry-tasks per-M:2 | smallest:K]
         Reruns stored TB-M3 A1 problems through this runner and compares every stored method error with
         studies/TB-M3-MIRACLE/primary/rows.jsonl (identical or 1e-6 relative). per-M:2 = the first 2 A1 tasks of
         each M in task order (the A1 comparability check run on the A2 box). Also checks the A2 task list.
  xcheck A2_EXTRACT_DIR FULL_RUN OUT --workers N
         After the full run has finished (FULL_RUN/finished.json exists): reruns its 20 smallest A2 problems here and
         compares them with the full run's rows. A mismatch is reported; it never stops or changes the run.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stdout
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
sys.path.insert(0, str(S / 'studies' / 'TB-M3-MIRACLE')); sys.path.insert(0, str(S / 'posthoc' / 'upgrade-mfsbl'))
import run_m3 as RM  # noqa: E402  (applies the TB-M3 corrections: guarded SBL, clip-masked adaptive stand-in)
import mfsbl as MF  # noqa: E402

A2_MD5 = 'bb9d5f12d39f53eadf9c448348d51036'
A2_PLANE_M = (1.437, 1.497)          # qualification: grid plane 1.467 m +- 3 cm from the array
assert RM.SEED0 == {'A1': 414000, 'R2': 414256}
RM.SEED0['A2'] = RM.SEED0['A1']      # matched design: A2 problem (source, M, draw) has A1's seed
M3_ROWS = S / 'studies/TB-M3-MIRACLE/primary/rows.jsonl'
INPUTS = ('common/estimators.py', 'common/v2.py', 'common/methods_v2.py', 'common/problems_v2.py', 'common/sbl_guard.py',
          'common/fields.py', 'common/mprir.py', 'posthoc/review3/adk_fix.py', 'analyze_v2.py',
          'studies/TB-M3-MIRACLE/run_m3.py', 'studies/TB-M3-MIRACLE/miracle_prep.py', 'studies/TB-M3-MIRACLE/analyze_m3.py',
          'posthoc/upgrade-mfsbl/mfsbl.py', 'posthoc/upgrade-mfsbl/PLAN.md', 'posthoc/upgrade-mfsbl/checks.json',
          'studies/TB-M4-MIRACLE-A2/run_m4.py', 'studies/TB-M4-MIRACLE-A2/analyze_m4.py', 'studies/TB-M4-MIRACLE-A2/PROTOCOL.md')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tkey(t):
    return (t[0], int(t[1]), float(t[2]), int(t[3]), int(t[4]))


def rkey(row):
    return tkey(row['task']) if 'task' in row else (row['scenario'], int(row['source']), float(row['band']), int(row['M']), int(row['draw']))


def run_task(task):
    row = RM.run_task(task)                                       # the TB-M3 row, computed by the TB-M3 code
    if 'error' in row:
        return row
    t0 = time.process_time()
    keys = list(MF.KEYS.values())
    try:
        sc, spk, fc, M, d, seed = task
        band, T, _, _ = RM.build(sc, spk, fc, M, seed)
        MF.add_methods(band, T, row['nmse'], row['info'], row['timing'])
    except Exception:
        for k in keys:
            for part in ('nmse', 'info', 'timing'):
                row[part].pop(k, None)
        row['mfsbl_error'] = traceback.format_exc()
    row['process_seconds_m3_methods'] = row['process_seconds']
    row['process_seconds'] = round(row['process_seconds'] + time.process_time() - t0, 1)
    return row


def check_tasks():
    """A2: 768 problems; each (source, band, M, draw) has A1's seed; no seed shared with R2; no key shared with A1."""
    t2, t1, tr = RM.tasks(['A2']), RM.tasks(['A1']), RM.tasks(['R2'])
    s1 = {tkey(t)[1:]: t[5] for t in t1}
    matched = all(s1[tkey(t)[1:]] == t[5] for t in t2)
    s2 = {t[5] for t in t2}
    keys2 = {tkey(t) for t in t2}
    ok = (len(t2) == 768 and len(keys2) == 768 and matched and min(s2) == 414000 and max(s2) == 414255 and len(s2) == 256
          and not (s2 & {t[5] for t in tr}) and not (keys2 & {tkey(t) for t in t1}))
    return dict(tasks=len(t2), seeds=[min(s2), max(s2)], distinct_seeds=len(s2), seeds_equal_a1=bool(matched),
                overlap_with_r2_seeds=len(s2 & {t[5] for t in tr}), passed=bool(ok))


def environment():
    env = dict(node=platform.node(), cpu=platform.processor(), python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    try:
        import threadpoolctl
        env['threadpools'] = threadpoolctl.threadpool_info()
    except Exception:
        buf = io.StringIO()
        with redirect_stdout(buf):
            np.show_config()
        env['numpy_config'] = buf.getvalue()
    return env


def qualify(extract):
    rec = json.loads((Path(extract) / 'miracle-A2-receipt.json').read_text())
    q = rec['qualification']
    got = sha(Path(extract) / 'miracle-A2.npz')
    ok = rec['md5'] == A2_MD5 and q['passed'] and A2_PLANE_M[0] <= q['plane_to_array_m'] <= A2_PLANE_M[1] and got == rec['extract_sha256']
    if not ok:
        raise SystemExit('A2 qualification failed; the study stops and is reported as not run: ' +
                         json.dumps(dict(q, extract_sha256=got, receipt_sha256=rec['extract_sha256'])))
    return dict(q, extract_sha256=got)


def stream(T, extract, scenarios, out, workers, append=False):
    """Runs T, appending rows to out/rows.jsonl as they finish; prints counts only, never an error value."""
    t0 = time.time(); n_err = n_mf = 0
    with Pool(workers, initializer=RM.init, initargs=(extract, scenarios)) as pool, open(out / 'rows.jsonl', 'a' if append else 'w') as fh:
        for n, row in enumerate(pool.imap_unordered(run_task, T), 1):
            n_err += 'error' in row; n_mf += 'mfsbl_error' in row
            fh.write(json.dumps(row, default=float) + '\n'); fh.flush()
            if n % 50 == 0 or n == len(T):
                print(f'{n}/{len(T)} tasks, {time.time() - t0:.0f} s, errors {n_err}, comparator errors {n_mf}', flush=True)
    return time.time() - t0


def read_rows(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def compare(row, ref):
    """Largest relative difference over the reference's methods and bins, and whether info is equal."""
    worst, n = 0.0, 0
    for m, v in ref['nmse'].items():
        if m not in row['nmse']:
            return dict(max_rel=float('inf'), values=n, missing=m)
        a, b = np.asarray(row['nmse'][m], float), np.asarray(v, float)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        rel = np.where(same, 0.0, np.abs(a - b) / np.maximum(np.abs(b), 1e-300))
        worst, n = max(worst, float(np.max(rel))), n + len(b)
    info = json.loads(json.dumps(row['info'], default=float))
    return dict(max_rel=worst, values=n, info_equal=all(info.get(k) == ref['info'][k] for k in ref['info']))


def select_a1(spec):
    T = RM.tasks(['A1'])
    kind, k = spec.split(':'); k = int(k)
    if kind == 'per-M':
        return [t for M in RM.MS for t in [t for t in T if t[3] == M][:k]]
    if kind == 'smallest':
        return sorted(T, key=lambda t: t[3])[:k]
    raise SystemExit(f'unknown --dry-tasks {spec}')


def cmd_dry(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    chk = check_tasks()
    T = select_a1(a.dry_tasks)
    (out / 'started.json').write_text(json.dumps(dict(mode='dry', dry_tasks=a.dry_tasks, tasks=[list(t) for t in T], task_check=chk,
                                                      environment=environment(), extract_a1=sha(Path(a.extract) / 'miracle-A1.npz'),
                                                      m3_rows=sha(M3_ROWS), inputs={n: sha(S / n) for n in INPUTS}), indent=1, default=str))
    elapsed = stream(T, a.extract, ['A1'], out, a.workers)
    ref = {rkey(r): r for r in read_rows(M3_ROWS) if r.get('scenario') == 'A1'}
    cmp = []
    for row in read_rows(out / 'rows.jsonl'):
        if 'error' in row:
            cmp.append(dict(task=row['task'], error=True)); continue
        r0 = ref[rkey(row)]
        c = compare(row, r0)
        cmp.append(dict(source=row['source'], band=row['band'], M=row['M'], draw=row['draw'], seed=row['seed'], **c,
                        mfsbl_error='mfsbl_error' in row,
                        mfsbl_db={k: float(10 * np.log10(np.mean(row['nmse'][k]))) for k in MF.KEYS.values() if k in row['nmse']},
                        seconds=dict(m3_methods_box=r0['process_seconds'], m3_methods_here=row['process_seconds_m3_methods'],
                                     total_here=row['process_seconds'],
                                     mfsbl_here={k: row['timing'][k] for k in MF.KEYS.values() if k in row['timing']})))
    ok = all(not c.get('error') and c['max_rel'] <= 1e-6 and c['info_equal'] for c in cmp)
    res = dict(task_list=chk, dry_tasks=a.dry_tasks, problems=len(cmp), reproduced_1e6=bool(ok),
               worst_rel=max([c.get('max_rel', float('inf')) for c in cmp] or [float('inf')]), elapsed_s=round(elapsed),
               cases=cmp, passed=bool(ok and chk['passed']))
    (out / 'dry-compare.json').write_text(json.dumps(res, indent=1, default=float))
    print('A1 reproduction', 'PASS' if res['passed'] else 'MISMATCH (reported, does not stop the A2 run)', 'worst', res['worst_rel'])


def cmd_run(a):
    out = Path(a.out)
    if a.resume:
        if not (out / 'started.json').exists():
            raise SystemExit('--resume needs an existing run folder')
    else:
        out.mkdir(parents=True, exist_ok=False)
    chk = check_tasks()
    if not chk['passed']:
        raise SystemExit('task list check failed: ' + json.dumps(chk))
    qual = qualify(a.extract)
    done = {rkey(r) for r in read_rows(out / 'rows.jsonl')}
    T = [t for t in sorted(RM.tasks(['A2']), key=lambda t: -t[3]) if tkey(t) not in done]   # largest problems first
    commit = subprocess.run(['git', '-C', str(S), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip() or None
    meta = dict(tasks=len(T), already_done=len(done), workers=a.workers, commit=commit, started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                task_check=chk, qualification=qual, environment=environment(), extracts={'A2': qual['extract_sha256']},
                inputs={n: sha(S / n) for n in INPUTS})
    name = f"resumed-{time.strftime('%Y%m%dT%H%M%S')}.json" if a.resume else 'started.json'
    (out / name).write_text(json.dumps(meta, indent=1, default=str))
    elapsed = stream(T, a.extract, ['A2'], out, a.workers, append=a.resume)
    rows = read_rows(out / 'rows.jsonl')
    fin = dict(tasks=len(rows), distinct_tasks=len({rkey(r) for r in rows}), errors=sum('error' in r for r in rows),
               mfsbl_errors=sum('mfsbl_error' in r for r in rows), elapsed_s_this_session=round(elapsed),
               finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'), rows_sha256=sha(out / 'rows.jsonl'))
    (out / 'finished.json').write_text(json.dumps(fin, indent=1))
    print('done', json.dumps(fin))


def cmd_xcheck(a):
    full = Path(a.full)
    if not (full / 'finished.json').exists():
        raise SystemExit('the full run has not finished; the cross-machine check reads no A2 value before it has')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    T = sorted(RM.tasks(['A2']), key=lambda t: t[3])[:20]
    (out / 'started.json').write_text(json.dumps(dict(mode='xcheck', tasks=[list(t) for t in T], environment=environment(),
                                                      extract_a2=sha(Path(a.extract) / 'miracle-A2.npz'),
                                                      full_run_rows=sha(full / 'rows.jsonl')), indent=1, default=str))
    stream(T, a.extract, ['A2'], out, a.workers)
    ref = {rkey(r): r for r in read_rows(full / 'rows.jsonl')}
    cmp = []
    for row in read_rows(out / 'rows.jsonl'):
        r0 = ref.get(rkey(row))
        if r0 is None or 'error' in row or 'error' in r0:
            cmp.append(dict(key=list(rkey(row)), comparable=False, here_error='error' in row, box_error=r0 is None or 'error' in r0))
            continue
        cmp.append(dict(key=list(rkey(row)), comparable=True, **compare(row, r0)))
    ok = all(c['comparable'] and c['max_rel'] <= 1e-6 and c['info_equal'] for c in cmp)
    res = dict(problems=len(cmp), identical=sum(c.get('max_rel') == 0 for c in cmp), passed=bool(ok),
               worst_rel=max([c.get('max_rel', float('inf')) for c in cmp] or [float('inf')]), cases=cmp,
               note='a mismatch is reported; it does not stop or change the registered run')
    (out / 'xcheck-compare.json').write_text(json.dumps(res, indent=1, default=float))
    print('cross-machine check', 'PASS' if ok else 'MISMATCH (reported)', 'worst', res['worst_rel'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('run'); p.add_argument('extract'); p.add_argument('out'); p.add_argument('--resume', action='store_true')
    p = sub.add_parser('dry'); p.add_argument('extract'); p.add_argument('out'); p.add_argument('--dry-tasks', default='per-M:2')
    p = sub.add_parser('xcheck'); p.add_argument('extract'); p.add_argument('full'); p.add_argument('out')
    for p in sub.choices.values():
        p.add_argument('--workers', type=int, required=True)
    a = ap.parse_args()
    dict(run=cmd_run, dry=cmd_dry, xcheck=cmd_xcheck)[a.cmd](a)


if __name__ == '__main__':
    main()
