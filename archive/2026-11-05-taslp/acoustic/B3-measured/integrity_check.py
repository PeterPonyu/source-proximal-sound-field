"""TB-M1 integrity rule: the boxes re-extracted from the MD5-verified dataset files must reproduce, byte for byte, the
SHA-256 values registered in PROTOCOLS/TB-M1.md (computed while the files were still downloading).

python3 -I -B integrity_check.py PREP_JSON OUT_JSON

For each loudspeaker the file bytes are taken from the assembled file if present, otherwise from the complete set of
downloaded chunks in order. The script computes the MD5 of all bytes and compares it with the Zenodo value, parses the
.npy header (shape and dtype), then extracts the box columns directly by byte offset (a code path independent of the
preprocessing reader) and hashes them. A loudspeaker whose bytes are not yet complete is reported as pending.
"""
from __future__ import annotations
import ast
import hashlib
import json
import sys
from pathlib import Path

DATA = Path('/home/zeyufu/Desktop/data/external/acoustic-geometry-20261104/mp-rir')
MD5 = dict(S4='a705e3f028cfab4303511c52f75749ef', S7='f33f021d2550711d1371b6d65c5abbfb', S8='25b297c591162d71059f53b6e955057c')
REGISTERED = dict(S4='cedadb7d95d308819a506216663488bee4f477ae296fec6eac3cd00db8f9f346',
                  S7='a8d10e3dd571096877c6badcbbc3043914fc66aca3acea8ed1b4211f88d6b492',
                  S8='ddf317ca08b59ab1f8462ab4517814983aca991dc27efbe15816a78c7a737834')
SIZE, CH, NCH = 3440099456, 16 * 1024 * 1024, 206


def sources_of_bytes(s):
    full = DATA / f'{s}_Mrir.npy'
    if full.exists() and full.stat().st_size == SIZE:
        return 'assembled file', [full]
    d = DATA / f'.{s}_chunks'
    parts = [d / f'{i:05d}' for i in range(NCH)]
    have = sum(p.exists() and p.stat().st_size == min(CH, SIZE - i * CH) for i, p in enumerate(parts))
    if have == NCH:
        return 'complete downloaded chunks', parts
    return f'pending ({have} of {NCH} chunks)', None


def main():
    prep = json.loads(Path(sys.argv[1]).read_text())
    out = {}
    for s in ('S4', 'S7', 'S8'):
        how, files = sources_of_bytes(s)
        if files is None:
            out[s] = dict(status='pending', source=how)
            continue
        md5 = hashlib.md5()
        head = b''
        for f in files:
            with open(f, 'rb') as fh:
                while True:
                    b = fh.read(1 << 24)
                    if not b:
                        break
                    if len(head) < 128:
                        head += b[:128 - len(head)]
                    md5.update(b)
        md5_ok = md5.hexdigest() == MD5[s]
        hl = int.from_bytes(head[8:10], 'little')
        header = ast.literal_eval(head[10:10 + hl].decode('latin1'))
        off0 = 10 + hl
        n = header['shape'][2]
        col = header['shape'][1] * n * 4
        cols = prep[s]['columns']
        sha = hashlib.sha256()
        sizes = [f.stat().st_size for f in files]
        starts = [sum(sizes[:i]) for i in range(len(files))]

        def read(a, b):
            buf = bytearray()
            for f, st, sz in zip(files, starts, sizes):
                lo, hi = max(a, st), min(b, st + sz)
                if lo < hi:
                    with open(f, 'rb') as fh:
                        fh.seek(lo - st)
                        buf += fh.read(hi - lo)
            return bytes(buf)
        for c in cols:
            sha.update(read(off0 + c * col, off0 + (c + 1) * col))
        out[s] = dict(status='verified' if md5_ok and sha.hexdigest() == REGISTERED[s] else 'MISMATCH', source=how,
                      md5=md5.hexdigest(), md5_matches_zenodo=md5_ok, npy_header=header, header_bytes=off0,
                      box_sha256=sha.hexdigest(), registered_box_sha256=REGISTERED[s], box_matches_registered=sha.hexdigest() == REGISTERED[s])
        print(s, out[s]['status'], how, flush=True)
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
