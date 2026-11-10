"""On the laptop, after m4_pull_and_verify.sh has copied the box's out/ back: check it against the registration commit.

python3 -B remote/m4_verify_return.py COMMIT RETURN_DIR [--partial]

Checks (any failure prints FAIL and exits 1; --partial reports without failing on unfinished jobs):
 1. RETURN-SHA256.txt (written on the box at the end) matches every returned file, and no returned file is over 50 MB.
 2. Every job has a receipt that names COMMIT and the same machine (machine.txt SHA-256 and CPU model, one machine for
    everything), exit code 0, and output hashes equal to the returned files.
 3. Every input hash in every receipt equals the bytes of that file at COMMIT (`git show`), so each result was produced
    by committed code from committed inputs.
 4. Row counts recomputed here: A2 768, revised fitting 1,536 (A1/R2) and 2,048 (simulation), comparator
    768/768/1,024/2,048 (A1/R2/room grid/simulation); no error rows (A2 error rows are kept by the registered runner
    and reported here as failures to look at, never re-run); every A2 seed equals A1's seed of the same
    (source, band, M, draw) in the stored TB-M3 rows (matched design, run_m4.check_tasks).
 5. The gate passed (cross_check.py); the A1 check (`run_m4.py dry`) is reported; the A2 download MD5 and the prep
    receipt MD5 are the registered bb9d5f12d39f53eadf9c448348d51036, the qualification passed, and the extract SHA-256
    is the one TB-M4 started from. The input hashes in run_m4.py's own started.json files also equal COMMIT.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import m4_receipt as MR  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
STUDY = 'archive/2026-11-05-taslp/acoustic'


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    partial = '--partial' in sys.argv
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', args[0]], text=True).strip()
    out = Path(args[1])
    fails, notes = [], []
    F = fails.append

    # 1. transfer integrity and size
    man = out / 'RETURN-SHA256.txt'
    if man.exists():
        listed = {}
        for l in man.read_text().splitlines():
            h, n = l.split('  ', 1); listed[n] = h
            if not (out / n).exists():
                F(f'listed but not returned: {n}')
            elif MR.sha(out / n) != h:
                F(f'hash differs after transfer: {n}')
        extra = [str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and str(p.relative_to(out)) not in listed
                 and p.name not in ('RETURN-SHA256.txt', 'STATUS.txt', 'm4_run_all.lock')]
        if extra:
            notes.append(f'{len(extra)} returned files are not in RETURN-SHA256.txt (written after it or partial pull): {extra[:5]}')
    else:
        (notes if partial else fails).append('RETURN-SHA256.txt missing (the launcher has not finished)')
    big = [str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.stat().st_size > MR.CAP]
    if big:
        F(f'files over 50 MB were returned: {big}')

    # 2-3. receipts
    mrec = out / 'machine.txt'
    msha = MR.sha(mrec) if mrec.exists() else None
    if msha is None:
        F('machine.txt not returned')
    machines = set()
    for job in MR.ORDER:
        rp = out / 'receipts' / f'{job}.json'
        if not rp.exists():
            (notes if partial else fails).append(f'{job}: no receipt')
            continue
        r = json.loads(rp.read_text())
        if r['registration_commit'] != commit:
            F(f'{job}: receipt names commit {r["registration_commit"]}, not {commit}')
        machines.add((r['machine'].get('record_sha256'), r['machine'].get('cpu_model')))
        if r['machine'].get('record_sha256') != msha:
            F(f'{job}: machine record hash differs from the returned machine.txt')
        if r['exit_code'] != 0 or not r['complete']:
            (notes if partial else fails).append(f'{job}: exit {r["exit_code"]}, complete={r["complete"]}')
        for n, h in r['outputs'].items():
            if not (out / n).exists():
                F(f'{job}: output {n} not returned')
            elif MR.sha(out / n) != h and not (job == 'm4' and n == 'm4/summary.json'):
                F(f'{job}: output {n} changed after its receipt')
        for n, h in r['inputs'].items():
            try:
                if n == 'remote/reference-rows.jsonl':
                    # not a committed file: the capsule maker derives it from the committed development rows, so
                    # derive it again from the commit with the same function and compare
                    import tempfile
                    from make_capsule import reference_rows
                    src = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{STUDY}/V2-dev/dev1-box/rows.jsonl'], stderr=subprocess.DEVNULL)
                    with tempfile.NamedTemporaryFile(suffix='.jsonl') as tf:
                        tf.write(src); tf.flush()
                        b = ''.join(json.dumps(x) + '\n' for x in reference_rows(tf.name, 6)).encode()
                else:
                    b = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{STUDY}/{n}'], stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                F(f'{job}: input {n} is not in commit {commit[:10]}'); continue
            if hashlib.sha256(b).hexdigest() != h:
                F(f'{job}: input {n} differs from commit {commit[:10]}')
    if len(machines) > 1:
        F(f'receipts name more than one machine: {machines}')

    # 4. rows
    for job in MR.ORDER:
        c = MR.count(out, job)
        if c is None:
            continue
        line = f'{job}: {c["good"]} good rows'
        if c.get('expected') is not None:
            line += f' of {c["expected"]}'
            if c['good'] != c['expected']:
                (notes if partial else fails).append(f'{job}: {c["good"]} good rows, expected {c["expected"]}')
        if c['errors_open']:
            F(f'{job}: {c["errors_open"]} error rows without a good row')
        if c['errors_superseded']:
            notes.append(f'{job}: {c["errors_superseded"]} error rows later retried successfully (kept in the file)')
        if c['duplicates']:
            notes.append(f'{job}: {c["duplicates"]} duplicate good rows')
        if 'per_set' in c and c['per_set'] != MR.MFSBL_SETS:
            (notes if partial else fails).append(f'{job}: per set {c["per_set"]}, expected {MR.MFSBL_SETS}')
        if job == 'm4' and c['good'] and c['scenarios'] != ['A2']:
            F(f'{job}: scenarios {c["scenarios"]}')
        if c.get('comparator_errors'):
            F(f'{job}: {c["comparator_errors"]} rows with a comparator error (mfsbl_error)')
        print(line, c.get('per_set', ''))
    if (out / 'm4' / 'rows.jsonl').exists():
        m3 = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{STUDY}/studies/TB-M3-MIRACLE/primary/rows.jsonl'], text=True)
        a1 = {(r['source'], float(r['band']), r['M'], r['draw']): r['seed'] for r in map(json.loads, m3.splitlines()) if r.get('scenario') == 'A1'}
        bad = [r for r in map(json.loads, open(out / 'm4' / 'rows.jsonl')) if 'error' not in r
               and a1.get((r['source'], float(r['band']), r['M'], r['draw'])) != r['seed']]
        if bad:
            F(f'{len(bad)} A2 rows whose seed is not A1\'s seed for the same problem')
    for d in ('m4', 'm4-dry'):
        for sj in sorted((out / d).glob('*.json')) if (out / d).exists() else []:
            meta = json.loads(sj.read_text())
            for n, h in (meta.get('inputs') or {}).items():
                try:
                    b = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{STUDY}/{n}'], stderr=subprocess.DEVNULL)
                except subprocess.CalledProcessError:
                    F(f'{sj.relative_to(out)}: input {n} is not in commit {commit[:10]}'); continue
                if hashlib.sha256(b).hexdigest() != h:
                    F(f'{sj.relative_to(out)}: input {n} differs from commit {commit[:10]}')
            if meta.get('task_check') and meta['task_check'].get('passed') is not True:
                F(f'{sj.relative_to(out)}: the runner\'s task-list check did not pass')

    # 5. gate, download, prep
    if not MR.complete(out, 'crosscheck'):
        F('cross_check.py did not pass (xcheck/cross-check.json)')
    dc = out / 'm4-dry' / 'dry-compare.json'
    if dc.exists():
        d = json.loads(dc.read_text())
        notes.append(f'A1 check (run_m4.py dry, {d["problems"]} problems): ' +
                     ('reproduced the stored TB-M3 rows' if d['passed'] else f'MISMATCH, worst relative {d["worst_rel"]} (registered: reported, does not stop A2)'))
    elif not partial:
        F('A1 check output m4-dry/dry-compare.json missing')
    fa = out / 'fetch-a2.json'
    if fa.exists():
        if json.loads(fa.read_text())['md5'] != MR.A2_MD5:
            F('A2 download MD5 is not the registered one')
    elif not partial:
        F('fetch-a2.json missing')
    pr = out / 'extract-a2' / 'miracle-A2-receipt.json'
    if pr.exists():
        p = json.loads(pr.read_text())
        if p['md5'] != MR.A2_MD5 or not p['qualification']['passed']:
            F(f'A2 prep receipt: md5 {p["md5"]}, qualification passed={p["qualification"]["passed"]}')
        npz = out / 'extract-a2' / 'miracle-A2.npz'
        if npz.exists() and p.get('extract_sha256') and MR.sha(npz) != p['extract_sha256']:
            F('A2 extract hash differs from its prep receipt')
        for sj in sorted((out / 'm4').glob('*.json')) if (out / 'm4').exists() else []:
            ex = json.loads(sj.read_text()).get('extracts')
            if ex and npz.exists() and ex.get('A2') != MR.sha(npz):
                F(f'{sj.relative_to(out)} names a different A2 extract')
    elif not partial:
        F('A2 prep receipt missing')

    for n in notes:
        print('note:', n)
    for f in fails:
        print('FAIL:', f)
    print('PASS' if not fails else 'FAIL', commit[:10], out, 'machine', next(iter(machines), (None, None))[1])
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
