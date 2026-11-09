"""Byte-range reads of the RAF split-zip central directory and metadata position files (budget 20 MB)."""
import os
import json, struct, subprocess, sys, zlib
from pathlib import Path
B = 'https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/real_acoustic_fields/rir/archived'
OUT = Path(os.environ.get('ACOUSTIC_DATA', 'data')) / 'raf'
BUDGET = 20_000_000
used = 0
def rng(name, a, b):
    global used
    used += b - a + 1
    if used > BUDGET:
        raise SystemExit(f'budget exceeded at {used}')
    return subprocess.run(['curl', '-sS', '--max-time', '120', '-r', f'{a}-{b}', f'{B}/{name}'], check=True, capture_output=True).stdout
SEG = 1073741824
def segname(room, disk, ndisks):
    return f'raf_{room}.zip' if disk == ndisks - 1 else f'raf_{room}.z{disk+1:02d}'
res = {}
for room, last in (('emptyroom', 895583963), ('furnishedroom', 817290211)):
    tail = rng(f'raf_{room}.zip', last - 65536, last - 1)
    i = tail.rfind(b'PK\x05\x06')
    eocd = tail[i:]
    disk, cd_disk, n_here, n_total, cd_size, cd_off = struct.unpack('<HHHHII', eocd[4:20])
    zip64 = tail.rfind(b'PK\x06\x06')
    if zip64 >= 0:
        (_, _, _, disk, cd_disk, n_here, n_total, cd_size, cd_off) = struct.unpack('<QHHIIQQQQ', tail[zip64+4:zip64+56])
    ndisks = disk + 1
    print(room, dict(ndisks=ndisks, cd_disk=cd_disk, entries=n_total, cd_size=cd_size, cd_off=cd_off))
    name = segname(room, cd_disk, ndisks)
    seglen = last if cd_disk == ndisks - 1 else SEG
    cd = rng(name, cd_off, min(cd_off + cd_size, seglen) - 1)
    if cd_off + cd_size > seglen:
        cd += rng(segname(room, cd_disk + 1, ndisks), 0, cd_off + cd_size - seglen - 1)
    (OUT / f'{room}-central-directory.bin').write_bytes(cd)
    entries, p = [], 0
    while p < len(cd) and cd[p:p+4] == b'PK\x01\x02':
        (method, crc, csize, usize, nlen, xlen, clen, dstart, _, _, lho) = struct.unpack('<10xH4xIIIHHHHHII', cd[p:p+46])
        fname = cd[p+46:p+46+nlen].decode('utf8', 'replace')
        extra = cd[p+46+nlen:p+46+nlen+xlen]
        q = 0
        while q < len(extra):
            hid, hl = struct.unpack('<HH', extra[q:q+4])
            if hid == 1:
                vals = extra[q+4:q+4+hl]; k = 0
                if usize == 0xFFFFFFFF: usize = struct.unpack('<Q', vals[k:k+8])[0]; k += 8
                if csize == 0xFFFFFFFF: csize = struct.unpack('<Q', vals[k:k+8])[0]; k += 8
                if lho == 0xFFFFFFFF: lho = struct.unpack('<Q', vals[k:k+8])[0]; k += 8
                if dstart == 0xFFFF: dstart = struct.unpack('<I', vals[k:k+4])[0]
            q += 4 + hl
        entries.append(dict(name=fname, method=method, csize=csize, usize=usize, disk=dstart, offset=lho))
        p += 46 + nlen + xlen + clen
    print(room, 'entries parsed', len(entries))
    meta = [e for e in entries if 'metadata' in e['name'] or e['name'].endswith(('.json', '.csv'))]
    print(room, 'metadata entries', [(e['name'], e['csize'], e['disk']) for e in meta][:20])
    files = {}
    for e in meta:
        if e['usize'] == 0: continue
        nm = segname(room, e['disk'], ndisks)
        hdr = rng(nm, e['offset'], e['offset'] + 29)
        nlen, xlen = struct.unpack('<HH', hdr[26:30])
        start = e['offset'] + 30 + nlen + xlen
        seglen = last if e['disk'] == ndisks - 1 else SEG
        end = start + e['csize'] - 1
        if end < seglen:
            raw = rng(nm, start, end)
        else:
            raw = rng(nm, start, seglen - 1) + rng(segname(room, e['disk'] + 1, ndisks), 0, end - seglen)
        data = zlib.decompress(raw, -15) if e['method'] == 8 else raw
        out = OUT / f"{room}-{e['name'].replace('/', '_')}"
        out.write_bytes(data); files[e['name']] = str(out)
    res[room] = dict(entries=len(entries), metadata=files, n_data_dirs=len({e['name'].split('/')[-2] for e in entries if e['name'].endswith('rir.wav')}))
print('bytes read', used)
(OUT / 'range-read-receipt.json').write_text(json.dumps(dict(bytes_read=used, budget=BUDGET, rooms=res), indent=2))
