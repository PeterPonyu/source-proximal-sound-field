"""Code-path test of run_m4 run/--resume/xcheck and the comparator-error path, with R2 data relabelled as A2.
Not a result; A2 is not touched. Task lists are truncated to keep it cheap."""
import json, shutil, sys, types
from pathlib import Path
import numpy as np

ROOT = Path('/home/zeyufu/Desktop/lrl-taslp-acoustic/archive/2026-11-05-taslp/acoustic')
sys.path.insert(0, str(ROOT / 'studies/TB-M4-MIRACLE-A2'))
import run_m4 as R

W = Path(sys.argv[1]); shutil.rmtree(W, ignore_errors=True); W.mkdir(parents=True)
ext = W / 'extract'; ext.mkdir()
shutil.copy(ROOT / 'studies/TB-M3-MIRACLE/extract/miracle-R2.npz', ext / 'miracle-A2.npz')
rec = json.loads((ROOT / 'studies/TB-M3-MIRACLE/extract/miracle-R2-receipt.json').read_text())
rec.update(md5=R.A2_MD5, extract_sha256=R.sha(ext / 'miracle-A2.npz'))
(ext / 'miracle-A2-receipt.json').write_text(json.dumps(rec))

real_tasks = R.RM.tasks
small = sorted(real_tasks(['A2']), key=lambda t: t[3])[:2]
LIMIT = {'n': 1}
R.check_tasks = (lambda f: (lambda: f()))(R.check_tasks)       # real check, computed on the real list
chk = R.check_tasks(); assert chk['passed'], chk
R.check_tasks = lambda: chk
R.RM.tasks = lambda sc: small[:LIMIT['n']] if sc == ['A2'] else real_tasks(sc)

orig_add = R.MF.add_methods
def flaky(band, T, out, info, timing):
    if LIMIT['n'] == 2:                          # fail the resumed task's comparator midway
        out['mfsbl_mackay'] = [0.0]; raise RuntimeError('injected comparator failure')
    return orig_add(band, T, out, info, timing)
R.MF.add_methods = flaky

a = types.SimpleNamespace(extract=str(ext), out=str(W / 'run'), workers=1, resume=False)
R.cmd_run(a)                                    # first task only
LIMIT['n'] = 2
a.resume = True
R.cmd_run(a)                                    # resume: only the second task runs
rows = [json.loads(l) for l in open(W / 'run/rows.jsonl')]
fin = json.loads((W / 'run/finished.json').read_text())
print('rows', len(rows), 'keys', [R.rkey(r) for r in rows])
print('mfsbl_error rows', [('mfsbl_error' in r, sorted(k for k in r['nmse'] if k.startswith('mfsbl'))) for r in rows])
print('finished', fin)
print('seconds', [(r['process_seconds_m3_methods'], r['process_seconds'], r['timing'].get('mfsbl_em')) for r in rows])
LIMIT['n'] = 2
R.MF.add_methods = orig_add
R.RM.tasks = lambda sc: sorted(real_tasks(sc), key=lambda t: t[3])[:2] if sc == ['A2'] else real_tasks(sc)
x = types.SimpleNamespace(extract=str(ext), full=str(W / 'run'), out=str(W / 'xcheck'), workers=1)
R.cmd_xcheck(x)
print(json.dumps({k: v for k, v in json.loads((W / 'xcheck/xcheck-compare.json').read_text()).items() if k != 'cases'}))
