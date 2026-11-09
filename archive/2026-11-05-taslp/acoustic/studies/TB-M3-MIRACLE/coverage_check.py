"""Coverage check of the TB-M3-MIRACLE primary run against the registered task design (post hoc, descriptive).

python3 -B coverage_check.py RUN_DIR   ->  RUN_DIR/coverage.json

Added 2026-11-07 after an independent review found that analyze_m3.py builds its condition list and its Bonferroni
family from the rows it receives and does not check them against the plan. This check leaves analyze_m3.py and its
summary unchanged. It confirms, for each scenario, that the rows are exactly the registered task set of run_m3.tasks
(32 sources x 3 bands x 4 microphone counts x 2 draws, each once, with the registered seed), that no row is an error,
and that every method error is a valid band. Exit status 1 if any check fails.

A valid band (tightened 2026-11-08 after backend review rounds 5-6): the row records the registered F = 6, and the
method's entry is a list of exactly six numeric, non-Boolean, finite, positive bin errors whose mean, taken as
analyze_v2.db takes it, is finite and positive. A scalar is not read as a band. Every method present in a row is
checked (analyze_v2.load converts all of them to decibels), and each method of USED must be present. The record names
the SHA-256 of the rows and of this checker, so a stored coverage.json is tied to the rows and the procedure.
"""
from __future__ import annotations
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import analyze_v2 as AV  # noqa: E402
import run_m3  # noqa: E402

USED = sorted(set(AV.GENERIC) | {AV.PRIMARY, 'gsmf', 'mo26_cap5', 'v1', 'two_stage_point', 'two_stage_region',
                                 'ref_point', 'ref_region'})
F_REGISTERED = 6


def valid_band(v):
    """Exactly six numeric, non-Boolean, finite, positive bin errors with a finite positive mean (analyze_v2.db)."""
    if not (isinstance(v, list) and len(v) == F_REGISTERED
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0 for x in v)):
        return False
    with np.errstate(all='ignore'):
        mean = float(np.mean(v))
    return math.isfinite(mean) and mean > 0


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(out_dir):
    """The coverage record of the rows in out_dir; make_numbers_v2.py calls this on the rows it consumes."""
    out_dir = Path(out_dir)
    rows = [json.loads(l) for l in open(out_dir / 'rows.jsonl')]
    planned = {(sc, spk, float(fc), M, d): seed for sc, spk, fc, M, d, seed in run_m3.tasks(('A1', 'R2'))}
    errors = [r for r in rows if 'error' in r]
    ok = [r for r in rows if 'error' not in r]
    keys = Counter((r['scenario'], r['source'], float(r['band']), r['M'], r['draw']) for r in ok)
    dup = [k for k, n in keys.items() if n > 1]
    missing = sorted(set(planned) - set(keys))
    extra = sorted(set(keys) - set(planned))
    seed_bad = [k for k in keys if k in planned and
                next(r['seed'] for r in ok if (r['scenario'], r['source'], float(r['band']), r['M'], r['draw']) == k) != planned[k]]
    f_bad = sum(1 for r in ok if r.get('F') != F_REGISTERED or isinstance(r.get('F'), bool))
    nonfinite = Counter()
    present = set()
    for r in ok:
        nm = r.get('nmse') if isinstance(r.get('nmse'), dict) else {}
        present |= set(nm)
        for m in sorted(set(USED) | set(nm)):
            if not valid_band(nm.get(m)):
                nonfinite[m] += 1
    per = {}
    for sc in ('A1', 'R2'):
        rs = [r for r in ok if r['scenario'] == sc]
        cells = Counter(f"{int(r['band'])}Hz-M{r['M']}" for r in rs)
        units = Counter((f"{int(r['band'])}Hz-M{r['M']}", r['source']) for r in rs)
        per[sc] = dict(problems=len(rs), conditions=len(cells), problems_per_condition=sorted(set(cells.values())),
                       sources=len({r['source'] for r in rs}), draws_per_source_and_condition=sorted(set(units.values())))
    res = dict(status='post hoc coverage check of the registered run; analyze_m3.py and summary.json unchanged',
               rows_sha256=sha256(out_dir / 'rows.jsonl'), checker_sha256=sha256(__file__),
               rows=len(rows), errors=len(errors), duplicates=len(dup), missing=len(missing), extra=len(extra),
               seed_mismatch=len(seed_bad), F_mismatch=f_bad, methods_required=USED, methods_checked=sorted(present | set(USED)),
               nonfinite_by_method=dict(sorted(nonfinite.items())), scenarios=per)
    res['complete'] = bool(not errors and not dup and not missing and not extra and not seed_bad and not f_bad and not nonfinite
                           and all(v['problems'] == 768 and v['conditions'] == 12 and v['problems_per_condition'] == [64]
                                   and v['sources'] == 32 and v['draws_per_source_and_condition'] == [2] for v in per.values()))
    return res


def main():
    out_dir = Path(sys.argv[1])
    res = check(out_dir)
    (out_dir / 'coverage.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'methods_checked'}, indent=1))
    sys.exit(0 if res['complete'] else 1)


if __name__ == '__main__':
    main()
