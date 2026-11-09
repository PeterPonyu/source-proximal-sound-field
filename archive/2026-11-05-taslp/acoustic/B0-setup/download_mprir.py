"""Chunked parallel resumable download of approved MP-RIR files (S4, S7, S8) with Zenodo MD5 verification."""
import hashlib, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
D = Path(os.environ.get('ACOUSTIC_DATA', 'data')) / 'mp-rir'
SIZE = 3440099456
MD5 = dict(S4='a705e3f028cfab4303511c52f75749ef', S7='f33f021d2550711d1371b6d65c5abbfb', S8='25b297c591162d71059f53b6e955057c')
CH = 16 * 1024 * 1024
WORKERS = min(int(sys.argv[1]) if len(sys.argv) > 1 else 4, 4)   # at most 4 connections to Zenodo
import os as _os
LOCK = D / '.downloader.pid'
if LOCK.exists():
    try:
        _os.kill(int(LOCK.read_text()), 0)
        sys.exit(f'another downloader is running (pid {LOCK.read_text()})')
    except (ProcessLookupError, ValueError):
        pass
LOCK.write_text(str(_os.getpid()))
order = sys.argv[2].split(',') if len(sys.argv) > 2 else ['S4', 'S7', 'S8']
NCH = len(range(0, SIZE, CH))
tasks = []
for s in order:
    cd = D / f'.{s}_chunks'
    cd.mkdir(exist_ok=True)
    for i, a in enumerate(range(0, SIZE, CH)):
        tasks.append((s, i, a, min(a + CH, SIZE) - 1, cd / f'{i:05d}'))


import json as _json
PRIORITY = _json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
_pri = {(s_, i_) for s_, lst in PRIORITY.items() for i_ in lst}
tasks.sort(key=lambda t: (0 if (t[0], t[1]) in _pri else 1, t[1] if (t[0], t[1]) in _pri else 0))


def get(t):
    s, i, a, b, f = t
    want = b - a + 1
    for attempt in range(80):
        if f.exists() and f.stat().st_size == want:
            return t
        part = f.with_suffix('.part')
        have = part.stat().st_size if part.exists() else 0
        if have >= want:
            part.unlink()
            have = 0
        hdr = part.with_suffix('.hdr')
        r = subprocess.run(['curl', '-sS', '-L', '--max-time', '1800', '-D', str(hdr), '-r', f'{a + have}-{b}',
                            f'https://zenodo.org/api/records/11148712/files/{s}_Mrir.npy/content'], capture_output=True)
        # Accept only a 206 partial response whose Content-Range starts at the requested byte (2026-11-05 fix: Zenodo's
        # JSON 500 error bodies were previously appended to chunk files as data; see DOWNLOAD-INCIDENT-2.md).
        heads = hdr.read_text(errors='replace') if hdr.exists() else ''
        last = heads.strip().split('\r\n\r\n')[-1] if heads else ''
        ok = last.startswith('HTTP/') and ' 206' in last.splitlines()[0] and f'bytes {a + have}-' in last
        if not ok:
            time.sleep(min(120, 10 * (attempt + 1)))
            continue
        with open(part, 'ab') as fh:
            fh.write(r.stdout)
        if part.stat().st_size == want:
            part.rename(f)
            return t
        time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f'chunk failed {s} {i}')


done, t0 = 0, time.time()
with ThreadPoolExecutor(WORKERS) as ex:
    futs = [ex.submit(get, t) for t in tasks]
    for fu in as_completed(futs):
        fu.result()
        done += 1
        if done % 20 == 0:
            print(f'{done}/{len(tasks)} chunks, {done * CH / (time.time() - t0) / 1e6:.2f} MB/s', flush=True)
for s in order:
    cd = D / f'.{s}_chunks'
    out = D / f'{s}_Mrir.npy'
    h = hashlib.md5()
    with open(out, 'wb') as fo:
        for i in range(NCH):
            data = (cd / f'{i:05d}').read_bytes()
            h.update(data)
            fo.write(data)
    ok = h.hexdigest() == MD5[s]
    print(s, 'MD5', 'OK' if ok else 'MISMATCH', h.hexdigest(), flush=True)
    if ok:
        for f in cd.iterdir():
            f.unlink()
        cd.rmdir()
print('ALL DONE', round(time.time() - t0), 's', flush=True)
