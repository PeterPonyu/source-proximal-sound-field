"""Post hoc descriptive comparison for the TB-M2-RIB record (not a registered rule): on the 48 problems and the one bin
of the registered follow-up, the per-bin variant of the primary (`ind_cv_ext`: the same order rule and extension, each
bin fitted alone, so it uses the same single-bin information as the official kernel) against the official Ribeiro et
al. 2024 kernel, and the primary against that per-bin variant for scale.

python3 -B studies/TB-M2-RIB/descriptive_perbin.py   ->  studies/TB-M2-RIB/descriptive-perbin.json

The registered functions of analyze_followups.py (TB-M2-RIB branch) are imported unchanged: per-problem error in dB on
the one bin, loudspeakers as units, Student-t intervals with Bonferroni over the 8 cells, pooled 95 % over the 6
loudspeakers. The TB-M2 rows and the TB-M2-RIB rows must match the SHA-256 in their receipts.
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import analyze_followups as AF  # noqa: E402

PAIRS = (('ind_cv_ext', 'ribeiro_official'), ('j_cv_ext', 'ind_cv_ext'), ('j_cv_ext', 'ribeiro_official'))


def checked(d):
    rc = json.loads((d / 'run-receipt.json').read_text())
    if hashlib.sha256((d / 'rows.jsonl').read_bytes()).hexdigest() != rc['outputs']['rows.jsonl']:
        raise SystemExit(f'rows of {d} do not match their receipt')
    return [json.loads(l) for l in open(d / 'rows.jsonl')]


def main():
    reg = {AF.key(r): r for r in checked(ROOT / 'studies/TB-M2/primary') if r.get('kind') == 'mesh' and 'error' not in r}
    new = [r for r in checked(HERE / 'primary') if 'error' not in r]
    cells = defaultdict(list)
    for r in new:
        base = reg[AF.key(r)]; b = r['bin']
        e = {m: AF.err(v, [b]) for m, v in base['nmse'].items()}
        e['ribeiro_official'] = AF.err(r['nmse']['ribeiro_official'])
        cells[f"{int(r['band'])}Hz-M{r['M']}"].append(dict(source=r['source'], e=e))
    out = dict(note='post hoc descriptive, not a registered rule', problems=len(new))
    for a, b in PAIRS:
        out[f'{a}-vs-{b}'] = AF.compare(cells, a, b, len(cells))
    (HERE / 'descriptive-perbin.json').write_text(json.dumps(out, indent=1, default=float))
    for a, b in PAIRS:
        c = out[f'{a}-vs-{b}']
        p = c['pooled']
        print(f"{a} - {b}: pooled {p['mean']:+.2f} [{p['low']:+.2f}, {p['high']:+.2f}], cells {c['wins']}/{c['ties']}/{c['losses']}, "
              f"undefined {c['undefined_problems']}")


if __name__ == '__main__':
    main()
