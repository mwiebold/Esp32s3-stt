#!/bin/bash
# usage: run_qemu.sh <tasr> <out_dir> <wav1> "<ref1>" [<wav2> "<ref2>" ...]
set -e
TASR=$1; OUT=$2; shift 2
LMARG=""; if [ -n "$TASR_LM" ]; then LMARG="--lm $TASR_LM"; fi
WAVS=(); REFS=()
while [ $# -gt 0 ]; do WAVS+=("$1"); REFS+=("$2"); shift 2; done
PY=${PYTHON:-$(command -v python3)}          # needs numpy + soundfile (ESP-IDF's own python does not have them)
QEMU=${QEMU:-$(command -v qemu-system-xtensa || echo "$HOME/esp/qemu-dl/qemu/bin/qemu-system-xtensa")}  # Espressif QEMU 9.x
. "${IDF_PATH:-$HOME/esp/esp-idf}/export.sh" > /dev/null 2>&1
cd "$(dirname "$0")/.."
"$PY" tools/mkimages.py "$TASR" "$OUT" --wavs "${WAVS[@]}" --refs "${REFS[@]}" --merge ${TASR_BUILD:-firmware/build} --layout ${TASR_LAYOUT:-tinyasr16} $LMARG > /dev/null
rm -f "$OUT/qemu.txt"
timeout 7200 "$QEMU" -nographic -machine esp32s3 -m 8M \
  -global driver=ssi_psram,property=is_octal,value=true -drive file="$OUT/flash.bin",if=mtd,format=raw \
  -icount shift=0 > "$OUT/qemu.txt" 2>&1 &
QPID=$!
until grep -qE "^DONE|abort|Guru" "$OUT/qemu.txt" 2>/dev/null; do sleep 3; done
kill $QPID 2>/dev/null || true
grep -E "CALIB|tinyasr:|UTT|REF|HYP|TOTAL|MEM|PROF|abort|Guru" "$OUT/qemu.txt"
