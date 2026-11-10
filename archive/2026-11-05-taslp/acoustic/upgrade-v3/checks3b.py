"""Checks for version 3b (PLAN-3b.md), run before any 3b development output.

    OPENBLAS_NUM_THREADS=4 python3 -B upgrade-v3/checks3b.py OUT.json
"""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
A = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(A / 'common')); sys.path.insert(0, str(A / 'V2-dev'))
import v3  # noqa: E402
import v2  # noqa: E402
import estimators as E  # noqa: E402
import run_dev as RD  # noqa: E402

spec = importlib.util.spec_from_file_location('v3_checked', HERE / 'v3-as-checked.py')
v3c = importlib.util.module_from_spec(spec); sys.modules['v3_checked'] = v3c; spec.loader.exec_module(v3c)


def main():
    RD.init()
    out = {}
    # D1 one-SE rule on known fold errors
    fe = np.array([[10, 5.0, 4.0, 3.9], [12, 6.0, 4.2, 4.1], [11, 5.5, 3.8, 3.9], [9, 5.2, 4.4, 4.0], [10, 5.1, 4.0, 4.1]])
    out['D1_one_se_outside'] = v3.one_se_order(fe) == 3        # argmin 3 (mean 4.00, SE 0.045); K=2 mean 4.08 is outside
    fe2 = fe.copy(); fe2[:, 2] = [4.0, 4.1, 3.9, 4.1, 4.05]
    out['D1_one_se_within'] = v3.one_se_order(fe2) == 2        # K=2 mean 4.03 is within 4.045
    out['D1_one_se_argmin_when_clear'] = v3.one_se_order(np.array([[3, 2, 1.0], [3, 2, 1.0001], [3, 2, 0.9999]])) == 2
    # D2 version 3 options unchanged by the 3b edit: same fit, same CV order, same extension as the checked file
    bs, T, _, _, _ = RD.sim_band('base', 400920)
    a = v3.fit_band(bs, 3); b = v3c.fit_band(bs, 3)
    out['D2_v3_fit_unchanged'] = all(np.array_equal(a[K]['S'], b[K]['S']) and np.array_equal(a[K]['W'], b[K]['W']) for K in range(4))
    folds = E.make_folds(bs.M, 5, 400920)
    out['D2_v3_cv_unchanged'] = v3.cv_order(bs, 3, folds)[0] == v3c.cv_order(bs, 3, folds)[0]
    ra = v3.extend(bs, a[2], 2); rb = v3c.extend(bs, b[2], 2)
    out['D2_v3_extend_unchanged'] = bool(np.array_equal(ra['centres'], rb['centres']) and np.array_equal(ra['lw'], rb['lw']))
    # D3 version-2 replication switch still exact
    out['D3_v2_switch'] = all(np.array_equal(v3.fit_band(bs, 2, opts=v3.V2_OPTS)[K]['S'], v2.fit_band(bs, 2)[K]['S']) for K in range(3))
    # D4 ball restarts never return a worse objective than the single run
    st1, st3 = [], []
    r1 = v3.extend(bs, a[3], 3, opts=v3.V3_OPTS, stats=st1)
    r3 = v3.extend(bs, a[3], 3, opts=v3.V3B_OPTS, stats=st3)
    out['D4_restart_not_worse'] = bool(r3['evidence'] >= r1['evidence'] - 1e-9)
    out['D4_stats'] = dict(single=st1, restarts=st3)
    out['all_pass'] = all(v for k, v in out.items() if isinstance(v, bool))
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == '__main__':
    main()
