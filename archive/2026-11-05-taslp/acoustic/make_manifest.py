"""Write FROZEN-INPUTS.json: SHA-256 of every input a registered run depends on, plus dependency versions."""
import hashlib, json, platform, sys
from pathlib import Path
import numpy as np, scipy
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FILES = ['archive/2026-10-27-science/acoustic-estimator-robustness/frozen/revision_controls.py',
         'archive/2026-10-28-finish/acoustic-estimator-robustness/common/science.py',
         'archive/2026-10-29-studies/acoustic-estimator-robustness/common/walls.py']
FILES += sorted(str(p.relative_to(ROOT)) for p in (HERE / 'common').glob('*.py'))
FILES += [str((HERE / n).relative_to(ROOT)) for n in ('run_sim.py', 'analyze_sim.py', 'analyze_sim_r1.py', 'run_measured.py', 'analyze_measured.py',
                                                    'measured_prep.py', 'seed_scan.py', 'make_manifest.py', 'SEEDS.json')]
FILES += sorted(str(p.relative_to(ROOT)) for p in (HERE / 'PROTOCOLS').glob('*.md'))
FILES += sorted(str(p.relative_to(ROOT)) for p in (HERE / 'studies').glob('*/README.md'))
FILES += [a for a in sys.argv[1:]]
out = dict(files={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES},
           dependency_versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
           data_inputs='TB-S1: generated deterministically from registered seeds. TB-M1: MP-RIR transfer functions preprocessed into B3-measured/prep (hashed here).')
(HERE / 'FROZEN-INPUTS.json').write_text(json.dumps(out, indent=2, sort_keys=True) + '\n')
print(len(FILES), 'files')
