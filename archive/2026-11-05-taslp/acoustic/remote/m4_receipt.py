"""Bookkeeping for m4_run_all.sh on the box (and m4_verify_return.py on the laptop): row counts, receipts, STATUS.txt.

python3 -B m4_receipt.py receipt WORK ACO JOB RC STARTED -- CMD...   write out/receipts/JOB.json after a job
python3 -B m4_receipt.py complete WORK JOB                          exit 0 if the job's rows are complete, no open errors
python3 -B m4_receipt.py gate WORK                                  exit 0 if the cross-machine check and the A1 dry run passed
python3 -B m4_receipt.py status WORK [--final]                      rewrite out/STATUS.txt
python3 -B m4_receipt.py hashes WORK                                write out/RETURN-SHA256.txt (files <= 50 MB)
python3 -B m4_receipt.py rebuild WORK ACO                           rebuild the A1/R2 extracts from the downloads; use them
                                                                    only if their SHA-256 equals the committed extracts
python3 -B m4_receipt.py fake WORK JOB                              M4_DRY only: write a complete set of synthetic rows

Row identity per job: TB-M4 rows by (scenario, source, band, M, draw); comparator rows by their key; revised-fitting
rows by their key. An error row whose key later has a good row (revfit resumes and retries errors) counts as
superseded; one without is open.
"""
from __future__ import annotations
import hashlib
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

A2_MD5 = 'bb9d5f12d39f53eadf9c448348d51036'
CAP = 50 * 2 ** 20
MFSBL_SETS = {'A1': 768, 'R2': 768, 'room': 1024, 'sim': 2048}


def m4key(r):
    return (r['task'][0], int(r['task'][1]), float(r['task'][2]), int(r['task'][3]), int(r['task'][4])) if 'task' in r else \
        (r['scenario'], int(r['source']), float(r['band']), int(r['M']), int(r['draw']))


def mfkey(r):
    return tuple(r['key'])


def rfkey(r):
    return json.dumps(r['key'], sort_keys=True)


# job -> output (relative to out/), kind, expected good rows, key function, inputs read (relative to the study folder)
JOBS = {
    'crosscheck': dict(path='xcheck/cross-check.json', kind='json', inputs=('remote/cross_check.py', 'remote/reference-rows.jsonl',
                                                                             'V2-dev/run_dev.py', 'V2-dev/prep/measured-dev.npz')),
    # run_m4.py dry: the A1 comparability check (default per-M:2, 8 stored TB-M3 A1 problems); reported, not a gate
    # the TB-M3 A1/R2 extracts rebuilt on the box (when the capsule does not carry them); must equal the committed bytes
    'a1r2-extract': dict(path='extract-a1r2/match.json', kind='json', inputs=('studies/TB-M3-MIRACLE/miracle_prep.py',)),
    'm4-dry': dict(path='m4-dry', kind='dir', expect=None, key=m4key,
                   inputs=('studies/TB-M4-MIRACLE-A2/run_m4.py', 'studies/TB-M3-MIRACLE/primary/rows.jsonl',
                           'studies/TB-M3-MIRACLE/extract/miracle-A1.npz')),
    'a2-prep': dict(path='extract-a2', kind='prep', inputs=('studies/TB-M3-MIRACLE/miracle_prep.py',)),
    # run_m4.py run: the registered 768 problems; error rows are kept and never retried (--resume skips them)
    'm4': dict(path='m4', kind='dir', expect=768, key=m4key,
               inputs=('studies/TB-M4-MIRACLE-A2/run_m4.py', 'studies/TB-M4-MIRACLE-A2/PROTOCOL.md', 'posthoc/upgrade-mfsbl/mfsbl.py',
                       'studies/TB-M3-MIRACLE/run_m3.py')),
    'm4-analysis': dict(path='m4/summary.json', kind='json', inputs=('studies/TB-M4-MIRACLE-A2/analyze_m4.py', 'analyze_v2.py',
                                                                     'studies/TB-M3-MIRACLE/primary/rows.jsonl')),
    'revfit-miracle': dict(path='revfit-miracle.jsonl', kind='file', expect=1536, key=rfkey,
                           inputs=('posthoc/upgrade-revfit/revfit.py', 'upgrade-v3/v3.py', 'studies/TB-M3-MIRACLE/primary/rows.jsonl',
                                   'studies/TB-M3-MIRACLE/extract/miracle-A1.npz', 'studies/TB-M3-MIRACLE/extract/miracle-R2.npz')),
    'mfsbl': dict(path='mfsbl', kind='dir', expect=sum(MFSBL_SETS.values()), key=mfkey, per_set=MFSBL_SETS,
                  inputs=('posthoc/upgrade-mfsbl/run_mfsbl.py', 'posthoc/upgrade-mfsbl/mfsbl.py', 'studies/TB-M3-MIRACLE/primary/rows.jsonl',
                          'studies/TB-M2/primary/rows.jsonl', 'studies/TB-M2-SBLFIX/primary/rows.jsonl', 'studies/TB-S2/primary/rows.jsonl',
                          'studies/TB-M3-MIRACLE/extract/miracle-A1.npz', 'studies/TB-M3-MIRACLE/extract/miracle-R2.npz',
                          'B3-measured/meshrir-prep/meshrir-prep.npz')),
    'revfit-sim': dict(path='revfit-sim.jsonl', kind='file', expect=2048, key=rfkey,
                       inputs=('posthoc/upgrade-revfit/revfit.py', 'upgrade-v3/v3.py', 'posthoc/review3/refit.py',
                               'studies/TB-M2/primary/rows.jsonl', 'studies/TB-S2/primary/rows.jsonl',
                               'B3-measured/meshrir-prep/meshrir-prep.npz', 'B3-measured/heldout-prep/heldout.npz')),
}
ORDER = list(JOBS)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rows_file(out, job):
    j = JOBS[job]
    if j['kind'] == 'file':
        return out / j['path']
    if j['kind'] == 'dir':
        return out / j['path'] / 'rows.jsonl'
    return None


def count(out, job):
    """Good keys, open and superseded error keys, duplicate good rows, per-set counts; None if no row file."""
    j, f = JOBS[job], rows_file(out, job)
    if f is None or not f.exists():
        return None
    good, bad, lines, bad_lines, mf_err = set(), set(), 0, 0, 0
    with open(f) as fh:
        for l in fh:
            if not l.strip():
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:          # a row being written while STATUS reads the file
                continue
            lines += 1
            mf_err += 'mfsbl_error' in r
            k = j['key'](r)
            if 'error' in r:
                bad.add(k); bad_lines += 1
            else:
                good.add(k)
    c = dict(rows=lines, good=len(good), distinct=len(good | bad), duplicates=lines - bad_lines - len(good), errors_open=len(bad - good),
             errors_superseded=len(bad & good), comparator_errors=mf_err, expected=j.get('expect'))
    if job == 'm4-dry' and (out / j['path'] / 'started.json').exists():
        c['expected'] = len(json.loads((out / j['path'] / 'started.json').read_text())['tasks'])
    if 'per_set' in j:
        c['per_set'] = {s: sum(1 for k in good if k[0] == s) for s in j['per_set']}
    if job == 'm4':
        c['scenarios'] = sorted({k[0] for k in good})
    return c


def complete(out, job):
    j = JOBS[job]
    if j['kind'] == 'json':
        p = out / j['path']
        if not p.exists():
            return False
        d = json.loads(p.read_text())
        return {'crosscheck': d.get('passed') is True, 'a1r2-extract': d.get('all_equal') is True}.get(job, True)
    if j['kind'] == 'prep':
        r = out / j['path'] / 'miracle-A2-receipt.json'
        return r.exists() and (out / j['path'] / 'miracle-A2.npz').exists()
    c = count(out, job)
    if c is None:
        return False
    if job == 'm4-dry':                          # complete when compared; a mismatch is a reported result, not a failure
        return (out / j['path'] / 'dry-compare.json').exists()
    if job == 'm4':                              # every task has a row (error rows are kept, never retried)
        return (out / j['path'] / 'finished.json').exists() and c['distinct'] == c['expected']
    ok = c['good'] == c['expected'] and c['errors_open'] == 0
    if 'per_set' in j:
        ok = ok and c['per_set'] == j['per_set']
    return ok


def commit_of(work, aco):
    cap = Path(work) / 'capsule' / 'CAPSULE-SHA256.txt'
    if cap.exists():
        return cap.read_text().split('\n', 1)[0].split()[1], sha(cap)
    import subprocess
    c = subprocess.run(['git', '-C', str(aco), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    return f'{c} (working tree, not a capsule)', None


def machine(out):
    rec = out / 'machine.txt'
    cpu = next((l.split(':', 1)[1].strip() for l in open(rec) if l.startswith('Model name')), None) if rec.exists() else None
    mem = None
    try:
        mem = round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2 ** 30, 1)
    except (ValueError, OSError):
        pass
    return dict(record='machine.txt', record_sha256=sha(rec) if rec.exists() else None, cpu_model=cpu or platform.processor(),
                nproc=os.cpu_count(), mem_total_gib=mem, hostname=socket.gethostname(), single_machine=True)


def outputs(out, job):
    p = out / JOBS[job]['path']
    files = [p] if p.is_file() else sorted(q for q in p.rglob('*') if q.is_file()) if p.exists() else []
    return {str(q.relative_to(out)): sha(q) for q in files}


def receipt(work, aco, job, rc, started, cmd):
    out, aco = Path(work) / 'out', Path(aco)
    commit, listing = commit_of(work, aco)
    rec = dict(job=job, command=cmd, exit_code=int(rc), started=started, finished=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               registration_commit=commit, capsule_listing_sha256=listing, machine=machine(out),
               python=platform.python_version(), threads={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                                                                                         'MKL_NUM_THREADS', 'OPENBLAS_CORETYPE')},
               post_hoc=job.startswith(('revfit', 'mfsbl')), registered=job in ('a2-prep', 'm4-dry', 'm4', 'm4-analysis'),
               counts=count(out, job), complete=bool(complete(out, job)), outputs=outputs(out, job),
               inputs={n: sha(aco / n) for n in JOBS[job]['inputs'] if (aco / n).exists()})
    try:
        import numpy, scipy
        rec.update(numpy=numpy.__version__, scipy=scipy.__version__)
    except ImportError:
        pass
    x = out / 'xcheck' / 'cross-check.json'
    rec['cross_machine_check'] = dict(passed=json.loads(x.read_text()).get('passed'), sha256=sha(x)) if x.exists() else None
    if job in ('a2-prep', 'm4', 'm4-analysis'):
        f = out / 'fetch-a2.json'
        p = out / 'extract-a2' / 'miracle-A2-receipt.json'
        rec['a2'] = dict(download=json.loads(f.read_text()) if f.exists() else None,
                         prep_receipt=json.loads(p.read_text()) if p.exists() else None,
                         extract_sha256=sha(out / 'extract-a2' / 'miracle-A2.npz') if (out / 'extract-a2' / 'miracle-A2.npz').exists() else None)
    (out / 'receipts').mkdir(parents=True, exist_ok=True)
    (out / 'receipts' / f'{job}.json').write_text(json.dumps(rec, indent=1, default=str))


def rebuild(work, aco):
    """miracle_prep.py on the downloaded A1.h5 and R2.h5 (in parallel); copy into the study's extract folder on a match."""
    import shutil
    import subprocess
    out, aco = Path(work) / 'out', Path(aco)
    info = json.loads((aco / 'remote' / 'M4-CAPSULE.json').read_text())
    want = info.get('rebuild_on_box') or {}
    dest = out / 'extract-a1r2'; dest.mkdir(parents=True, exist_ok=True)
    res = dict(rebuilt=bool(want), scenarios={})
    if not want:
        res.update(all_equal=True, note='the capsule carries the committed extracts')
    else:
        md5 = {'A1': '44ded79a5129083a9ac61edf8e6009e2', 'R2': '4f2d5a9bf4c2402b0335eb2c1da34ffb'}
        procs = {sc: subprocess.Popen([sys.executable, '-I', '-B', str(aco / 'studies/TB-M3-MIRACLE/miracle_prep.py'), sc,
                                       str(Path(work) / 'data' / f'{sc}.h5'), md5[sc], str(dest)],
                                      stdout=open(dest / f'prep-{sc}.log', 'w'), stderr=subprocess.STDOUT) for sc in md5}
        for sc, p in procs.items():
            rc = p.wait()
            rel = f'studies/TB-M3-MIRACLE/extract/miracle-{sc}.npz'
            f = dest / f'miracle-{sc}.npz'
            got = sha(f) if f.exists() else None
            res['scenarios'][sc] = dict(exit_code=rc, sha256=got, committed_sha256=want[rel], equal=got == want[rel])
            if got == want[rel]:
                shutil.copyfile(f, aco / rel)
                f.unlink()                     # the copy in the study folder is the one the jobs read; no 16 MB duplicate returns
        res['all_equal'] = all(v['equal'] for v in res['scenarios'].values())
    (dest / 'match.json').write_text(json.dumps(res, indent=1))
    print('A1/R2 extracts:', 'equal to the committed bytes' if res['all_equal'] else 'DIFFERENT: ' + json.dumps(res['scenarios']))


def gate(work):
    return complete(Path(work) / 'out', 'crosscheck')


def status(work, final=False):
    out, st = Path(work) / 'out', Path(work) / 'out' / 'state'
    now = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    try:
        load = open('/proc/loadavg').read().split()[:3]
        mem = next(int(l.split()[1]) for l in open('/proc/meminfo') if l.startswith('MemAvailable')) // 2 ** 20
    except (OSError, StopIteration):
        load, mem = ['?'] * 3, '?'
    L = [f'STATUS {now}   cores {os.cpu_count()}   load {" ".join(load)}   MemAvailable {mem} GiB']
    a2 = 'checked' if (st / 'a2-download.done').exists() else 'FAILED' if (st / 'a2-download.failed').exists() else 'downloading'
    size = (Path(work) / 'data' / 'A2.h5').stat().st_size if (Path(work) / 'data' / 'A2.h5').exists() else 0
    L.append(f'A2 download: {a2} ({size / 1e9:.2f} of 1.07 GB)')
    if (st / 'GATE.failed').exists():
        L.append('GATE FAILED: the cross-machine check did not pass; nothing else was started')
    L.append(f'{"job":15s} {"state":10s} {"good":>6s} {"expected":>8s} {"open err":>8s} {"superseded":>10s}  detail')
    failed = []
    for job in ORDER:
        state = next((s for s in ('done', 'failed', 'blocked', 'started') if (st / f'{job}.{s}').exists()), 'waiting')
        if state in ('failed', 'blocked'):
            failed.append(job)
        c = count(out, job) or {}
        det = ''
        if 'per_set' in c:
            det = ' '.join(f'{s}={n}/{JOBS[job]["per_set"][s]}' for s, n in c['per_set'].items())
        if job == 'crosscheck' and (out / JOBS[job]['path']).exists():
            det = 'passed=' + str(json.loads((out / JOBS[job]['path']).read_text()).get('passed'))
        if job == 'm4-dry' and (out / 'm4-dry' / 'dry-compare.json').exists():
            det = 'A1 reproduction passed=' + str(json.loads((out / 'm4-dry' / 'dry-compare.json').read_text()).get('passed'))
        if c.get('comparator_errors'):
            det += f' comparator_errors={c["comparator_errors"]}'
        L.append(f'{job:15s} {state:10s} {c.get("good", "-")!s:>6s} {c.get("expected", "-")!s:>8s} {c.get("errors_open", "-")!s:>8s} '
                 f'{c.get("errors_superseded", "-")!s:>10s}  {det}')
    if final:
        L.append('ALL DONE' + (f' (not complete: {", ".join(failed)})' if failed else ''))
    tmp = out / 'STATUS.txt.tmp'
    tmp.write_text('\n'.join(L) + '\n'); tmp.replace(out / 'STATUS.txt')


def hashes(work):
    out = Path(work) / 'out'
    lines, big = [], []
    for q in sorted(p for p in out.rglob('*') if p.is_file()):
        rel = str(q.relative_to(out))
        if rel in ('RETURN-SHA256.txt', 'STATUS.txt.tmp') or rel.endswith('.lock'):
            continue
        if q.stat().st_size > CAP:
            big.append(rel); continue
        lines.append(f'{sha(q)}  {rel}\n')
    (out / 'RETURN-SHA256.txt').write_text(''.join(lines))
    if big:
        (out / 'RETURN-TOO-BIG.txt').write_text('\n'.join(big) + '\n')


def fake(work, job):
    """M4_DRY only: a complete, synthetic row set with the real key shapes, to exercise the launcher and the counters."""
    assert os.environ.get('M4_DRY') == '1', 'fake rows are written only in M4_DRY=1 mode'
    out = Path(work) / 'out'
    j = JOBS[job]
    if j['kind'] == 'json':
        p = out / j['path']; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dict(passed=True, all_equal=True, dry_mode_fake=True)))
        return
    if j['kind'] == 'prep':
        d = out / j['path']; d.mkdir(parents=True, exist_ok=True)
        (d / 'miracle-A2.npz').write_bytes(b'dry-mode placeholder, not an extract')
        (d / 'miracle-A2-receipt.json').write_text(json.dumps(dict(md5=A2_MD5, dry_mode_fake=True, qualification=dict(passed=True))))
        return
    if job == 'm4':
        rows = [dict(scenario='A2', source=s, band=b, M=M, draw=d, seed=0) for s in range(32) for M in (24, 48, 96, 192)
                for d in (0, 1) for b in (250.0, 500.0, 1000.0)][:j['expect']]
    elif job == 'mfsbl':
        rows = [dict(key=[s, i, 250.0, 24, 0, i]) for s, n in MFSBL_SETS.items() for i in range(n)]
    else:
        rows = [dict(key=dict(kind=job, i=i)) for i in range(j['expect'])]
    if j['kind'] == 'dir':
        (out / j['path']).mkdir(parents=True, exist_ok=False)
    f = rows_file(out, job)
    with open(f, 'a') as fh:
        for r in rows:
            fh.write(json.dumps(dict(r, dry_mode_fake=True)) + '\n')
            time.sleep(float(os.environ.get('M4_DRY_SLEEP', '0')))
    if job == 'm4':
        (out / j['path'] / 'finished.json').write_text(json.dumps(dict(dry_mode_fake=True, tasks=len(rows))))


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'receipt':
        i = args.index('--')
        work, aco, job, rc, started = args[:5]
        receipt(work, aco, job, rc, started, args[i + 1:])
    elif cmd == 'complete':
        sys.exit(0 if complete(Path(args[0]) / 'out', args[1]) else 1)
    elif cmd == 'gate':
        sys.exit(0 if gate(args[0]) else 1)
    elif cmd == 'status':
        status(args[0], final='--final' in args)
    elif cmd == 'hashes':
        hashes(args[0])
    elif cmd == 'rebuild':
        rebuild(args[0], args[1])
    elif cmd == 'fake':
        fake(args[0], args[1])
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
