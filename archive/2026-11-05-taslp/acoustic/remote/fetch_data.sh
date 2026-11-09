#!/usr/bin/env bash
# On the box: fetch the raw measured data straight from Zenodo, check every MD5, link the path the frozen modules
# hard-code, and run the version-1 integrity check for S7/S8. Draft, 2026-11-05.
#
#   bash fetch_data.sh /root/taslp            (after setup_box.sh; run inside tmux)
#
# Connection budget: at most 4 to Zenodo at any time (3 MP-RIR files in parallel + MeshRIR on 1).
# curl -f never writes an HTTP error body into a data file (the laptop downloader did; see B0-setup/DOWNLOAD-INCIDENT-2.md),
# and -C - resumes a partial file. A file is accepted only after its MD5 matches the Zenodo record.
set -euo pipefail
WORK=${1:-/root/taslp}
CAP=$WORK/capsule; ST=$CAP/archive/2026-11-05-taslp/acoustic
MP=$WORK/data/mp-rir; MR=$WORK/data/meshrir
mkdir -p "$MP" "$MR"
log() { echo "[fetch $(date -u +%H:%M:%S)] $*" | tee -a "$WORK/fetch.log"; }

get() {  # get RECORD NAME DEST_DIR MD5
  local url="https://zenodo.org/api/records/$1/files/$2/content" out="$3/$2"
  for i in $(seq 1 40); do
    if curl -fsSL -C - --retry 3 --retry-delay 15 --connect-timeout 30 -o "$out" "$url"; then break; fi
    log "retry $i for $2"; sleep 30
  done
  echo "$4  $out" | md5sum -c - | tee -a "$WORK/fetch.log"
}

log "MP-RIR small files"
get 11148712 Mxyz.npy "$MP" d1aad2e13b7b32b6f5484df5b4ac8d93
get 11148712 Setup.npz "$MP" 6ed01cadfdc6b4ed51236dbdd2af3f54
log "MP-RIR S4/S7/S8 (3 connections) and MeshRIR S32-M441 (1 connection), in parallel"
get 11148712 S4_Mrir.npy "$MP" a705e3f028cfab4303511c52f75749ef & p1=$!
get 11148712 S7_Mrir.npy "$MP" f33f021d2550711d1371b6d65c5abbfb & p2=$!
get 11148712 S8_Mrir.npy "$MP" 25b297c591162d71059f53b6e955057c & p3=$!
get 10852693 S32-M441_npy.zip "$MR" 9818fc66b36513590e7abd071243d8e9 & p4=$!
wait $p1 $p2 $p3 $p4
# measured_prep.py reads a verified full file only when its marker holds the Zenodo MD5
for s in S4:a705e3f028cfab4303511c52f75749ef S7:f33f021d2550711d1371b6d65c5abbfb S8:25b297c591162d71059f53b6e955057c; do
  echo "${s#*:}" > "$MP/${s%%:*}_Mrir.md5ok"
done

log "link the laptop path that common/mprir.py and B3-measured/integrity_check.py hard-code"
mkdir -p /home/zeyufu/Desktop/data/external/acoustic-geometry-20261104
ln -sfn "$MP" /home/zeyufu/Desktop/data/external/acoustic-geometry-20261104/mp-rir

log "version-1 integrity rule: boxes re-extracted from the verified files must match the registered SHA-256"
source "$WORK/env.sh"
( cd "$WORK" && nice -n 10 python -I -B "$ST/B3-measured/integrity_check.py" "$ST/B3-measured/prep/measured-prep.json" "$WORK/integrity.json" )
python - "$WORK/integrity.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print({s: (v['status'], v.get('box_matches_registered')) for s, v in d.items()})
assert all(v['status'] == 'verified' for v in d.values()), 'INTEGRITY FAILURE: stop and report before any registered run'
PY
sha256sum "$MR/S32-M441_npy.zip" "$MP"/*.npy "$MP"/Setup.npz | tee "$WORK/data-sha256.txt"
log "done"
