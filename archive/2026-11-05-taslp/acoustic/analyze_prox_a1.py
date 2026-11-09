"""TB-PROX amendment 1 (PROTOCOLS/TB-PROX-AMENDMENT-1.md): the registered TB-PROX analysis on the merged rows.

python3 -B analyze_prox_a1.py   ->  studies/TB-PROX/a1-analysis/{rows.jsonl, merge.json, summary.json}

The mo26-prox rows come from the registered TB-PROX run (studies/TB-PROX/primary, its non-error mo26 rows) and the
gsmf-prox rows from the amendment run (studies/TB-PROX/gsmf-a1, all rows, errors included); both must match the SHA-256
in their receipts. The original gsmf-prox error rows stay in studies/TB-PROX/primary and are counted here. The merged
rows are written to studies/TB-PROX/a1-analysis/rows.jsonl and analysed by analyze_prox.py, imported unchanged.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_prox as AP  # noqa: E402

PRIMARY = HERE / 'studies/TB-PROX/primary'
A1 = HERE / 'studies/TB-PROX/gsmf-a1'
OUT = HERE / 'studies/TB-PROX/a1-analysis'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows_checked(d):
    rc = json.loads((d / 'run-receipt.json').read_text())
    if sha(d / 'rows.jsonl') != rc['outputs']['rows.jsonl']:
        raise SystemExit(f'rows of {d} do not match their receipt')
    return [json.loads(l) for l in open(d / 'rows.jsonl')], rc


def variant(r):
    return r['task'][-1] if 'error' in r else r['variant']


def main():
    p_rows, p_rc = rows_checked(PRIMARY)
    a_rows, a_rc = rows_checked(A1)
    mo = [r for r in p_rows if variant(r) == 'mo26' and 'error' not in r]
    gs = [r for r in a_rows if variant(r) == 'gsmf']
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / 'rows.jsonl', 'w') as fh:
        for r in mo + gs:
            fh.write(json.dumps(r, default=float) + '\n')
    merge = dict(primary=dict(protocol_commit=p_rc['protocol_commit'], rows_sha256=sha(PRIMARY / 'rows.jsonl'), status=p_rc['status'],
                              mo26_rows=len(mo), mo26_errors=sum(variant(r) == 'mo26' and 'error' in r for r in p_rows),
                              gsmf_error_rows_superseded=sum(variant(r) == 'gsmf' and 'error' in r for r in p_rows),
                              gsmf_rows_without_error=sum(variant(r) == 'gsmf' and 'error' not in r for r in p_rows),
                              not_run=p_rc.get('n_not_run')),
                 amendment=dict(protocol_commit=a_rc['protocol_commit'], rows_sha256=sha(A1 / 'rows.jsonl'), status=a_rc['status'],
                                gsmf_rows=len(gs), gsmf_errors=sum('error' in r for r in gs), not_run=a_rc.get('n_not_run')),
                 merged_rows_sha256=sha(OUT / 'rows.jsonl'))
    (OUT / 'merge.json').write_text(json.dumps(merge, indent=1))
    print(json.dumps(merge, indent=1))
    sys.argv = [sys.argv[0], str(OUT)]
    AP.main()


if __name__ == '__main__':
    main()
