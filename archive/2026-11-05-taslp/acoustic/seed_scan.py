"""Scan every earlier seed in the repository (and the Phase A pilot folder) and fail on any overlap with SEEDS.json."""
import csv
import json
import re
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]   # archive/2026-11-05-taslp/acoustic -> repository root
csv.field_size_limit(10 ** 9)


def csv_seeds(path):
    seeds = set()
    with path.open(newline='', errors='replace') as f:
        reader = csv.reader(f)
        try:
            head = next(reader)
        except StopIteration:
            return seeds
        cols = [i for i, h in enumerate(head) if 'seed' in h.lower()]
        if not cols:
            return seeds
        for row in reader:
            for i in cols:
                if i < len(row):
                    for tok in re.findall(r'\d+', row[i]):
                        seeds.add(int(tok))
    return seeds


def text_seeds(path):
    seeds = set()
    text = path.read_text(errors='replace')
    for m in re.finditer(r'seeds?\s*(?:range\s*)?[:=]?\s*(\d{3,7})\s*(?:-|–|—|to|\.\.)\s*(\d{3,7})', text, re.I):
        a, b = int(m.group(1)), int(m.group(2))
        if 0 <= b - a <= 200000:
            seeds.update((a, b))
    return seeds


def main():
    declared = json.loads((HERE / 'SEEDS.json').read_text())
    ranges = list(declared['declared'].values()) + [list(r) for r in declared['external_known_reviewer_ranges']]
    names = list(declared['declared']) + ['external'] * len(declared['external_known_reviewer_ranges'])
    spent = [list(r) for r in declared.get('spent_never_reused', [])]
    used, per_file = set(), []
    skip = {'.git', 'build', 'node_modules'}
    for top in ('archive', 'code', 'reports', 'delivery', 'sources', 'tools', 'tests'):
        for path in sorted((ROOT / top).rglob('*')):
            if any(p in skip for p in path.parts) or not path.is_file() or HERE in path.parents:
                continue
            if path.suffix == '.csv' and path.stat().st_size < 600e6:
                found = csv_seeds(path)
            elif path.suffix in ('.md', '.json', '.txt') and path.stat().st_size < 5e6:
                found = text_seeds(path)
            else:
                continue
            if found:
                per_file.append((str(path.relative_to(ROOT)), min(found), max(found), len(found)))
                used |= found
    print(f'files with seeds: {len(per_file)}; distinct integers: {len(used)}; global range {min(used)}-{max(used)}')
    bands = {}
    for _, lo, hi, _ in per_file:
        bands.setdefault(lo // 10000, []).append((lo, hi))
    for band in sorted(bands):
        print(f'  band {band * 10000}: {min(a for a, _ in bands[band])}-{max(b for _, b in bands[band])} ({len(bands[band])} files)')
    failures = []
    for name, (lo, hi) in zip(names, ranges):
        clash = sorted(x for x in used if lo <= x <= hi)
        print(f'  declared {name}: {lo}-{hi}; earlier seeds inside: {len(clash)}')
        if clash and name != 'external':   # reviewer ranges are listed for information; they are expected in review notes
            failures.append((name, clash[:5]))
    phase_a = Path('/home/zeyufu/Desktop/docs/acoustic-taslp-upgrade-2026-11-04')
    for path in sorted(phase_a.rglob('*')):
        if path.is_file() and path.suffix in ('.csv', '.md', '.json', '.py'):
            found = csv_seeds(path) if path.suffix == '.csv' else text_seeds(path)
            if path.suffix == '.py':
                found |= {int(t) for t in re.findall(r'\b(399\d{3})\b', path.read_text(errors='replace'))}
            if found:
                used |= found
                per_file.append((str(path), min(found), max(found), len(found)))
    for lo, hi in spent:
        used |= set(range(lo, hi + 1))
    for name, (lo, hi) in zip(names, ranges):
        if name == 'external':
            continue
        clash = sorted(x for x in used if lo <= x <= hi)
        if clash and (name, clash[:5]) not in failures:
            failures.append((name, clash[:5]))
    mine = [(n, r) for n, r in zip(names, ranges) if n != 'external']
    for i, (n1, r1) in enumerate(mine):
        for n2, r2 in mine[i + 1:]:
            if r1[0] <= r2[1] and r2[0] <= r1[1]:
                failures.append((n1, n2))
    if failures:
        print('OVERLAP', failures)
        sys.exit(1)
    print('no overlap with any earlier seed or declared range')


if __name__ == '__main__':
    main()
