"""Theory curves of the identifiability check (PLAN.md): the bound on synthetic ball geometries, no study data.

A ball of radius a (lengths in units of a; the bound in units of a is scale-invariant at fixed ka) holds M microphones
drawn uniformly; a source lies at distance d from the centre; six bins span the octave about the band-centre ka with
the bin placement of the studies; eta = 1 and rho = 1e-2 in every bin. One bin: the median over the six bins of the
per-bin bound; six bins: coordinates shared. Each value is the median over 8 microphone draws.

    python3 -B posthoc/crb/theory.py      # writes theory.json
"""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import crb

KA = (0.5, 2.0, 8.0)
MICS = (24, 48, 96, 192)
D_OVER_A = (1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0)
ETA, RHO, DRAWS = 1.0, 1e-2, 8


def ball(M, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7311, M]))
    v = rng.normal(size=(M, 3)); v /= np.linalg.norm(v, axis=1)[:, None]
    return v * rng.random(M)[:, None] ** (1 / 3)


def main():
    out = dict(eta=ETA, rho=RHO, draws=DRAWS, bins=6, curves=[])
    for ka in KA:
        ks = ka * 2.0 ** ((np.arange(6) + 0.5) / 6 - 0.5)
        for M in MICS:
            for q in D_OVER_A:
                one, six = [], []
                for dr in range(DRAWS):
                    x = ball(M, dr); s = np.array([q, 0.0, 0.0])
                    Js = [crb.info(x, s, k, ETA, RHO) for k in ks]
                    one.append(np.median([crb.rms_bound(J) for J in Js])); six.append(crb.rms_bound(sum(Js)))
                b1, b6 = float(np.median(one)), float(np.median(six))
                out['curves'].append(dict(ka=ka, M=M, d_over_a=q, one_bin=b1, six_bins=b6, ratio=b6 / b1))
    (Path(__file__).parent / 'theory.json').write_text(json.dumps(out, indent=1) + '\n')
    for c in out['curves']:
        if c['M'] in (24, 48, 192) and c['d_over_a'] in (1.2, 1.5, 2.0, 3.0, 5.0, 10.0):
            print(f"ka {c['ka']:4} M {c['M']:3} d/a {c['d_over_a']:5}: one {c['one_bin']:8.3f} six {c['six_bins']:8.3f} ratio {c['ratio']:.3f}")


if __name__ == '__main__':
    main()
