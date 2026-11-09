"""Version-2 seed scan: the version-1 scan (seed_scan.py, unchanged) with one difference. The development block may
contain earlier development seeds (that is its purpose); it is still checked for overlap with every other declared
block. Every registered block must contain no earlier seed and overlap no other block.

python3 -B seed_scan_v2.py
"""
import json
import re
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import seed_scan as S  # noqa: E402


def main():
    declared = json.loads((HERE / 'SEEDS.json').read_text())
    blocks = dict(declared['declared'])
    used = set()
    for top in ('archive', 'code', 'reports', 'delivery', 'sources', 'tools', 'tests'):
        for path in sorted((S.ROOT / top).rglob('*')):
            if any(p in {'.git', 'build', 'node_modules'} for p in path.parts) or not path.is_file() or HERE in path.parents:
                continue
            if path.suffix == '.csv' and path.stat().st_size < 600e6:
                used |= S.csv_seeds(path)
            elif path.suffix in ('.md', '.json', '.txt') and path.stat().st_size < 5e6:
                used |= S.text_seeds(path)
    for path in sorted(Path('/home/zeyufu/Desktop/docs/acoustic-taslp-upgrade-2026-11-04').rglob('*')):
        if path.is_file() and path.suffix in ('.csv', '.md', '.json', '.py'):
            used |= S.csv_seeds(path) if path.suffix == '.csv' else S.text_seeds(path)
            if path.suffix == '.py':
                used |= {int(t) for t in re.findall(r'\b(399\d{3})\b', path.read_text(errors='replace'))}
    for lo, hi in declared.get('spent_never_reused', []):
        used |= set(range(lo, hi + 1))
    fail = []
    for name, (lo, hi) in blocks.items():
        clash = sorted(x for x in used if lo <= x <= hi)
        dev = name.startswith('development')
        print(f'  {name}: {lo}-{hi}; earlier seeds inside: {len(clash)}' + (' (development block: expected)' if dev and clash else ''))
        if clash and not dev:
            fail.append((name, clash[:5]))
    names = list(blocks)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if blocks[a][0] <= blocks[b][1] and blocks[b][0] <= blocks[a][1]:
                fail.append((a, b))
    if fail:
        print('OVERLAP', fail); sys.exit(1)
    print('no registered block contains an earlier seed; no two declared blocks overlap')


if __name__ == '__main__':
    main()
