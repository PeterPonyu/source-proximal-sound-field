"""Released runner for TB-M1 (registered measured arm, C3) on the MP-RIR transfer functions.

Requires the preregistration commit (FROZEN-INPUTS.json byte-identical, including the preprocessed transfer-function
file), one BLAS thread per worker and nice 10. Problems: 3 loudspeakers x 48 frequency bins (12 per octave band at
125, 250, 500, 1000 Hz) x M in {24, 48, 96, 192} x 4 random microphone subsets of the 648-point box; every other box
point is a target. The fitted acoustic centre reaches only the two references.

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python3 -B run_measured.py \
    --protocol-commit <sha> --prep B3-measured/prep --output studies/TB-M1/primary --workers 21
"""
from __future__ import annotations
import argparse
import csv
import datetime
import json
import platform
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'common'))
import numpy as np  # noqa: E402
import estimators as E  # noqa: E402
import methods  # noqa: E402
from run_sim import check_release, sha  # noqa: E402

SOURCES = ('S4', 'S7', 'S8')
COUNTS = (24, 48, 96, 192)
DRAWS = 4
SEED0 = 406700
PREP = {}


def band_of(f):
    return min((125.0, 250.0, 500.0, 1000.0), key=lambda c: abs(np.log(f / c)))


def tasks():
    out, n = [], 0
    for s in SOURCES:
        for b in range(len(PREP['freqs'])):
            for M in COUNTS:
                for d in range(DRAWS):
                    out.append((s, b, M, d, SEED0 + n)); n += 1
    return out


def one(task):
    s, b, M, d, seed = task
    t0 = time.process_time()
    try:
        meta = PREP['meta'][s]
        pos = PREP[f'{s}_positions']; H = PREP[f'{s}_H'][:, b]
        f = float(PREP['freqs'][b])
        rng = np.random.default_rng(np.random.SeedSequence([seed, 8101]))
        mic = np.sort(rng.choice(len(pos), M, replace=False))
        tgt = np.setdiff1d(np.arange(len(pos)), mic)
        scale = np.sqrt(np.mean(np.abs(H[mic]) ** 2))
        prob = E.Problem(pos[mic], H[mic] / scale, pos[tgt], f, c=meta['sound_speed'],
                         box=(np.asarray(meta['box_lo']), np.asarray(meta['box_hi'])))
        centre = np.asarray(meta['acoustic_centre'])[None, :]
        out, info, timing = methods.evaluate(prob, H[tgt] / scale, centre, seed)
        row = dict(source=s, bin=b, f=f, band=band_of(f), M=M, draw=d, seed=seed, **{f'nmse_{k}': v for k, v in out.items()})
        row.update(info)
        row['sel_below_floor'] = bool(any(p[2] < 0 for p in info['sel_theta']))
        row['k1_below_floor'] = bool(info['k1_theta'][2] < 0)
        row['process_seconds'] = round(time.process_time() - t0, 3)
        row['timing'] = {k: round(v, 3) for k, v in timing.items()}
        return row
    except Exception:
        return dict(source=s, bin=b, M=M, draw=d, seed=seed, error=traceback.format_exc())


def load_prep(prep_dir):
    z = np.load(prep_dir / 'measured-prep.npz', allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d['meta'] = json.loads((prep_dir / 'measured-prep.json').read_text())
    return d


def init(prep_dir):
    PREP.update(load_prep(Path(prep_dir)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol-commit', required=True)
    ap.add_argument('--prep', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--workers', type=int, default=21)
    a = ap.parse_args()
    manifest = check_release(a.protocol_commit)
    prep = HERE / a.prep
    out = HERE / a.output
    out.mkdir(parents=True, exist_ok=False)
    init(prep)
    tl = tasks()
    (out / 'started.json').write_text(json.dumps(dict(protocol_commit=a.protocol_commit, problems=len(tl), workers=a.workers,
                                                      utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=1))
    ka = subprocess.run([sys.executable, '-B', str(HERE / 'common/known_answers.py'), str(out / 'known-answer-checks.json')],
                        capture_output=True, text=True)
    if ka.returncode != 0:
        (out / 'failure.json').write_text(json.dumps(dict(stage='known answers', stdout=ka.stdout[-4000:]), indent=1))
        sys.exit(1)
    t0 = time.time(); rows, errors = [], []
    with Pool(a.workers, initializer=init, initargs=(str(prep),)) as pool, open(out / 'partial-rows.jsonl', 'w') as part:
        for i, row in enumerate(pool.imap_unordered(one, tl, chunksize=1)):
            part.write(json.dumps(row, default=float) + '\n'); part.flush()
            (errors if 'error' in row else rows).append(row)
            if (i + 1) % 256 == 0:
                print(f'{i + 1}/{len(tl)} problems, {time.time() - t0:.0f} s', flush=True)
    if errors:
        (out / 'failure.json').write_text(json.dumps(dict(stage='problems', errors=errors[:20], n_errors=len(errors)), indent=1))
    rows.sort(key=lambda r: (r['source'], r['bin'], r['M'], r['draw']))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / 'field-metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    receipt = dict(status='complete' if not errors else 'complete_with_errors', protocol_commit=a.protocol_commit,
                   elapsed_seconds=round(time.time() - t0, 1), n_rows=len(rows), n_errors=len(errors),
                   prep_sha256={p.name: sha(p) for p in sorted(prep.iterdir()) if p.is_file()},
                   outputs={p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'run-receipt.json'},
                   host=platform.node(), frozen_inputs=manifest['files'])
    (out / 'run-receipt.json').write_text(json.dumps(receipt, indent=1))
    print('done', receipt['status'], receipt['elapsed_seconds'])


if __name__ == '__main__':
    main()
