"""Furnished room: read only the last 4 KB of the central directory (the metadata entries are its last records,
as in the empty room) and then the two compressed position files; the full 16 MB directory was not read."""
import json, struct, subprocess, zlib
from pathlib import Path
B = 'https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/real_acoustic_fields/rir/archived'
OUT = Path('/home/zeyufu/Desktop/data/external/acoustic-geometry-20261104/raf')
used = 0
def rng(name, a, b):
    global used; used += b - a + 1
    return subprocess.run(['curl', '-sS', '--max-time', '120', '-r', f'{a}-{b}', f'{B}/{name}'], check=True, capture_output=True).stdout
cd_off, cd_size = 801206381, 16083732          # from the end-of-central-directory record read by raf_range_reads.py
tail = rng('raf_furnishedroom.zip', cd_off + cd_size - 4096, cd_off + cd_size - 1)
ents, p = [], 0
while True:
    i = tail.find(b'PK\x01\x02', p)
    if i < 0: break
    nlen, xlen, clen = struct.unpack('<HHH', tail[i + 28:i + 34])
    if i + 46 + nlen > len(tail): break
    method, = struct.unpack('<H', tail[i + 10:i + 12]); csize, usize = struct.unpack('<II', tail[i + 20:i + 28])
    dstart, = struct.unpack('<H', tail[i + 34:i + 36]); lho, = struct.unpack('<I', tail[i + 42:i + 46])
    ents.append(dict(name=tail[i + 46:i + 46 + nlen].decode('utf8', 'replace'), method=method, csize=csize, usize=usize, disk=dstart, offset=lho)); p = i + 4
for e in [e for e in ents if 'metadata/all_' in e['name']]:
    nm = 'raf_furnishedroom.zip' if e['disk'] == 9 else f"raf_furnishedroom.z{e['disk'] + 1:02d}"
    hdr = rng(nm, e['offset'], e['offset'] + 29); nl, xl = struct.unpack('<HH', hdr[26:30]); st = e['offset'] + 30 + nl + xl
    raw = rng(nm, st, st + e['csize'] - 1)
    (OUT / f"furnishedroom-{e['name'].replace('/', '_')}").write_bytes(zlib.decompress(raw, -15) if e['method'] == 8 else raw)
print('furnished bytes', used)
