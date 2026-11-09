"""Bounds per problem for the post hoc identifiability check (PLAN.md). Not registered.

Rebuilds every TB-S2 problem and every TB-M2 `mesh` problem exactly as in its registered run and computes, per bin,
the plug-in power ratio and ridge at the reference position, the per-bin bound and the bound with coordinates shared
across the band, plus the two sensitivities of PLAN.md: (a) the bound and its plug-in values at the band fit's first
point; (b) equal power ratio and ridge in every bin (eta = 1, rho = 1e-2) at the reference position. The file it writes
holds no localization distance. As a check that the rebuilt problems are the registered ones, the evidence-tuned sinc
kernel is refitted in every bin and its error compared with the stored value.

    OMP_NUM_THREADS=1 python3 -B posthoc/crb/run_bounds.py --workers 20
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
S = HERE.parents[1]
sys.path.insert(0, str(S / 'common'))
sys.path.insert(0, str(HERE))
import crb  # noqa: E402
import estimators as E  # noqa: E402
import problems_v2 as PB  # noqa: E402

PREP = {}


def rows():
    out = []
    for line in (S / 'studies/TB-S2/primary/rows.jsonl').read_text().splitlines():
        r = json.loads(line)
        out.append(dict(study='TB-S2', cell=r['cell'], seed=r['seed'], M=r['M_mics'], j_k1_S=r['info']['j_k1_S'],
                        sinc_ev=r['nmse']['sinc_ev']))
    for line in (S / 'studies/TB-M2/primary/rows.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['kind'] != 'mesh':
            continue
        out.append(dict(study='TB-M2', source=r['source'], band=r['band'], M=r['M'], draw=r['draw'], seed=r['seed'],
                        c_used=r['c_used'], j_k1_S=r['info']['j_k1_S'], sinc_ev=r['nmse']['sinc_ev']))
    return out


def init():
    z = np.load(S / 'B3-measured/meshrir-prep/meshrir-prep.npz', allow_pickle=False)
    PREP.update({k: z[k] for k in z.files})


def bounds_at(band, s, plug=True):
    J, pl = [], []
    for p in band.probs:
        if plug:
            q = crb.plugin(p.x, p.y, s, p.k)
            pl.append(dict(eta=q['eta'], rho=q['rho'], at_bound=q['at_bound']))
            J.append(crb.info(p.x, s, p.k, q['eta'], q['rho'], K=p.H))
        else:
            J.append(crb.info(p.x, s, p.k, 1.0, 1e-2, K=p.H))
    return [crb.rms_bound(j) for j in J], crb.rms_bound(sum(J)), pl


def work(t):
    t0 = time.process_time()
    if t['study'] == 'TB-S2':
        band, T, _, ref = PB.sim_band(t['cell'], t['seed'])
        key = dict(study=t['study'], cell=t['cell'], seed=t['seed'])
    else:
        band, T, ref, _ = PB.meshrir_band(PREP, t['source'], t['band'], t['M'], t['seed'])
        ref = ref[0]
        key = dict(study=t['study'], source=t['source'], band=t['band'], M=t['M'], draw=t['draw'], seed=t['seed'])
        assert abs(band.c - t['c_used']) < 1e-9, 'sound speed differs from the registered run'
    assert band.M == t['M'] and band.F == 6
    sinc = [E.nmse(E.sinc_fit(p, 'evidence')['pred'], T[:, f]) for f, p in enumerate(band.probs)]
    rebuild = max(abs(a - b) / abs(b) for a, b in zip(sinc, t['sinc_ev']))
    ref = np.asarray(ref, float)
    Bf, BJ, pl = bounds_at(band, ref)
    Bf_a, BJ_a, pl_a = bounds_at(band, np.asarray(t['j_k1_S'], float))
    Bf_b, BJ_b, _ = bounds_at(band, ref, plug=False)
    return dict(key, M=band.M, ref=ref.tolist(), c=band.c, freqs=band.freqs.tolist(), plugin=pl, B_f=Bf, B_J=BJ,
                plugin_a=pl_a, B_f_a=Bf_a, B_J_a=BJ_a, B_f_b=Bf_b, B_J_b=BJ_b, rebuild_rel_diff=rebuild,
                seconds=round(time.process_time() - t0, 1))


def task_key(r):
    return (r['study'], r.get('cell'), r.get('source'), r.get('band'), r['M'] if r['study'] == 'TB-M2' else None, r['seed'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--limit', type=int, default=None, help='first N tasks only (mechanics check)')
    ap.add_argument('--out', default=str(HERE / 'bounds.jsonl'))
    a = ap.parse_args()
    tasks = rows()
    tasks.sort(key=lambda t: t['study'] != 'TB-S2')          # the slow simulated rebuilds first
    if a.limit:
        tasks = tasks[:a.limit]
    out = Path(a.out)
    done = {task_key(json.loads(line)) for line in out.read_text().splitlines()} if out.exists() else set()
    todo = [t for t in tasks if task_key(t) not in done]
    print(f'{len(tasks)} tasks, {len(todo)} to do', flush=True)
    with Pool(a.workers, initializer=init) as pool, out.open('a') as fh:
        for i, r in enumerate(pool.imap_unordered(work, todo, chunksize=1)):
            fh.write(json.dumps(r) + '\n'); fh.flush()
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                print(f'{i + 1}/{len(todo)} {time.strftime("%H:%M:%S")}', flush=True)
    print('BOUNDS_DONE', flush=True)


if __name__ == '__main__':
    main()
