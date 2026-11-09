"""Release check for the registered version-2 runners, on the laptop (git) or on a rented box (capsule).

git mode (laptop): as version 1. The registration commit must be an ancestor of HEAD, and every frozen input must be
byte-identical to it.
capsule mode (box, no .git):
- the capsule must name the registration commit;
- every file listed in CAPSULE-SHA256.txt must match;
- FROZEN-INPUTS.json must have the SHA-256 recorded on the laptop from the registration commit (passed on the command
  line);
- every frozen input must match its manifest SHA-256.
Both modes then check the dependency versions, the thread caps and nice 10. They return the record that goes into the
run receipt, including the machine record (a registered study runs entirely on one machine).
"""
from __future__ import annotations
import hashlib
import json
import os
import platform
import socket
import subprocess
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def machine_record(work=None):
    cpu = {}
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if ':' in line:
                k, v = (s.strip() for s in line.split(':', 1))
                if k in ('model name', 'flags') and k not in cpu:
                    cpu[k] = v
    except OSError:
        pass
    import numpy as np
    import scipy
    blas = np.show_config(mode='dicts')['Build Dependencies']['blas']
    rec = dict(cpu_model=cpu.get('model name'), cpu_flags_sha256=hashlib.sha256(cpu.get('flags', '').encode()).hexdigest(),
               logical_cpus=os.cpu_count(), kernel=platform.release(), os=platform.platform(),
               host_sha256=hashlib.sha256(socket.gethostname().encode()).hexdigest()[:16],
               python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
               blas=f"{blas.get('name')} {blas.get('version')}",
               env={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                                                   'OPENBLAS_CORETYPE', 'NPY_DISABLE_CPU_FEATURES')})
    mt = Path(work) / 'machine.txt' if work else None
    if mt and mt.exists():
        rec['machine_txt_sha256'] = sha(mt)
    rec['record_sha256'] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
    return rec


def _common_checks(manifest):
    import numpy as np
    import scipy
    actual = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    if actual != manifest['dependency_versions']:
        raise RuntimeError(f'Dependency versions changed: {actual}')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(name) != '1':
            raise RuntimeError(f'Thread cap missing: {name}')
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError('Run requires nice -n 10')


def check(commit, study_dir, root=None, capsule_root=None, frozen_sha256=None, work=None):
    """Returns the release record for the receipt; raises on any mismatch."""
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('A full lowercase registration commit hash is required')
    study_dir = Path(study_dir).resolve()
    rel_study = 'archive/2026-11-05-taslp/acoustic'
    manifest_path = study_dir / 'FROZEN-INPUTS.json'
    manifest = json.loads(manifest_path.read_text())
    if capsule_root is None:                                              # git mode, laptop
        root = Path(root).resolve()
        subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', commit, 'HEAD'], check=True)
        for rel in [f'{rel_study}/FROZEN-INPUTS.json'] + list(manifest['files']):
            frozen = subprocess.check_output(['git', '-C', str(root), 'show', f'{commit}:{rel}'])
            if (root / rel).read_bytes() != frozen:
                raise RuntimeError(f'Unfrozen file: {rel}')
        base, mode, listing_sha = root, 'git', None
    else:                                                                 # capsule mode, box
        base = Path(capsule_root).resolve()
        listing = (base / 'CAPSULE-SHA256.txt').read_text().splitlines()
        if listing[0] != f'registration_commit {commit}':
            raise RuntimeError('Capsule was built from another commit')
        for line in listing[1:]:
            h, name = line.split('  ', 1)
            if sha(base / name) != h:
                raise RuntimeError(f'Capsule file changed: {name}')
        if frozen_sha256 is None or sha(manifest_path) != frozen_sha256:
            raise RuntimeError('FROZEN-INPUTS.json does not match the SHA-256 recorded at the registration commit')
        mode, listing_sha = 'capsule', sha(base / 'CAPSULE-SHA256.txt')
    for rel, expected in manifest['files'].items():
        if sha(base / rel) != expected:
            raise RuntimeError(f'Input SHA mismatch: {rel}')
    _common_checks(manifest)
    return dict(mode=mode, registration_commit=commit, frozen_inputs_json_sha256=sha(manifest_path),
                capsule_listing_sha256=listing_sha, frozen_inputs=manifest['files'],
                machine=dict(machine_record(work), single_machine=True))
