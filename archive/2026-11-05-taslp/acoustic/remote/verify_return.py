"""On the laptop, after rsync back: check a returned registered study against its registration (draft, 2026-11-05).

python3 -B verify_return.py REGISTRATION_COMMIT RETURNED_STUDY_OUTPUT_DIR

Checks: every frozen-input SHA-256 in the run receipt equals the registration commit's bytes; the receipt names one
machine (machine record hash) for the whole study and the cross-machine check passed on that machine; every output
file matches the SHA-256 in the receipt. Prints PASS or the first failure.
"""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def main():
    commit, out = sys.argv[1], Path(sys.argv[2])
    rec = json.loads((out / 'run-receipt.json').read_text())
    for rel, h in rec['frozen_inputs'].items():
        b = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{rel}'])
        assert hashlib.sha256(b).hexdigest() == h, f'frozen input differs from the registration: {rel}'
    for name, h in rec['outputs'].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == h, f'output changed after the run: {name}'
    m = rec.get('machine', {})
    assert m.get('record_sha256') and m.get('single_machine') is True, 'receipt does not name a single machine'
    assert rec.get('cross_machine_check', {}).get('passed') is True, 'cross-machine check missing or failed'
    print('PASS', commit, out, m.get('cpu_model'), rec.get('status'))


if __name__ == '__main__':
    main()
