"""Build the capsule that goes to the rented box, on the laptop, from the registration commit (draft, 2026-11-05).

python3 -B make_capsule.py REGISTRATION_COMMIT OUT.tar.gz [--reference-tasks N]

Contents, nothing else:
- the study folder `archive/2026-11-05-taslp/acoustic/` at the registration commit, without the version-1 study outputs,
  the reviews, the literature notes and the development rows (`git archive`, so only committed bytes travel);
- the frozen inputs outside the study folder that the known-answer suite reads (from FROZEN-INPUTS.json);
- `remote/reference-rows.jsonl`: the laptop development rows of the cross-machine check tasks;
- `CAPSULE-SHA256.txt`: the SHA-256 of every file, and the registration commit.
No manuscript, no other repository content and no raw dataset files are included; the raw MP-RIR and MeshRIR files
are fetched on the box from Zenodo with their MD5s checked (remote/fetch_data.sh).
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

STUDY = 'archive/2026-11-05-taslp/acoustic'
EXCLUDE = ('studies/TB-S1/primary', 'studies/TB-M1/primary', 'B6-review', 'literature', 'V2-dev/dev1/', 'V2-dev/dev1-box/', 'PHASE-C-PLAN.md')
ROOT = Path(__file__).resolve().parents[4]


def git(*a):
    return subprocess.check_output(['git', '-C', str(ROOT), *a])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('commit'); ap.add_argument('out'); ap.add_argument('--reference-tasks', type=int, default=6)
    ap.add_argument('--reference-rows', default='V2-dev/dev1-box/rows.jsonl')
    a = ap.parse_args()
    files = [f for f in git('ls-tree', '-r', '--name-only', a.commit, STUDY).decode().splitlines()
             if not any(f.startswith(f'{STUDY}/{e}') for e in EXCLUDE)]
    manifest = json.loads(git('show', f'{a.commit}:{STUDY}/FROZEN-INPUTS.json'))
    files += [f for f in manifest['files'] if not f.startswith(STUDY)]
    ref = [json.loads(l) for l in open(ROOT / STUDY / a.reference_rows) if '"error"' not in l]
    pick = []
    for kind, M in (('sim', 48), ('sim', 96), ('sim', 24), ('meas', 48), ('meas', 96), ('meas', 24)):
        cand = [r for r in ref if r['kind'] == kind and r['M_mics'] == M]
        if cand:
            pick.append(cand[0])
    pick = pick[:a.reference_tasks]
    for r in pick:
        r['_task'] = (['sim', r['cell'], r['seed']] if r['kind'] == 'sim' else
                      ['meas', r['source'], r['band'], r['M'], r['draw'], r['seed']])
    sha = {}
    with tarfile.open(a.out, 'w:gz') as tar:
        def add(name, data):
            ti = tarfile.TarInfo(f'capsule/{name}'); ti.size = len(data); ti.mtime = 0; ti.mode = 0o644
            tar.addfile(ti, io.BytesIO(data)); sha[name] = hashlib.sha256(data).hexdigest()
        for f in files:
            add(f, git('show', f'{a.commit}:{f}'))
        add(f'{STUDY}/remote/reference-rows.jsonl', ''.join(json.dumps(r) + '\n' for r in pick).encode())
        listing = f'registration_commit {a.commit}\n' + ''.join(f'{h}  {n}\n' for n, h in sorted(sha.items()))
        add('CAPSULE-SHA256.txt', listing.encode())
    print(len(sha), 'files;', Path(a.out).stat().st_size // 1024, 'KiB;', len(pick), 'cross-check tasks')


if __name__ == '__main__':
    main()
