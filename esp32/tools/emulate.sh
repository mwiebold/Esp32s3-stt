#!/bin/bash
# Run recordings through the real ESP32-S3 firmware in Espressif's QEMU (dual core, 8 MB PSRAM, 16 MB flash):
# the exact on-chip transcript, the instruction count and an estimated real-time factor for the board.
# usage: tools/emulate.sh clip.wav [clip2.wav ...]     (16 kHz mono; up to ~24 s in total)
#   MODEL=../models/nemo4.tnm tools/emulate.sh clip.wav   # int4 profile; NO_LM=1 for greedy decoding
set -e
cd "$(dirname "$0")/.."
ROOT=$(cd .. && pwd)
MODEL=${MODEL:-$ROOT/models/nemo8.tnm}; MODEL=$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")
PY=${PYTHON:-$(command -v python3)}
ARGS=()
for w in "$@"; do
  f=$(cd "$(dirname "$w")" && pwd)/$(basename "$w")
  out=/tmp/emu_$(basename "$w" | tr -c 'A-Za-z0-9._\n' _).wav   # resample/convert to 16 kHz mono PCM16
  "$PY" -c "
import sys, numpy as np, soundfile as sf
x, sr = sf.read(sys.argv[1], dtype='float32', always_2d=True); x = x.mean(1)
if sr != 16000:
    import scipy.signal as ss; x = ss.resample_poly(x, 16000, sr)
sf.write(sys.argv[2], (np.clip(x, -1, 1) * 32767).astype(np.int16), 16000, subtype='PCM_16')" "$f" "$out"
  ARGS+=("$out" "-")
done
# layout: transducer models need no LM; CTC models use the LM partition when models/nemo_lm.tlm exists
LMF=$ROOT/models/nemo_lm.tlm
case "$MODEL" in
  *rnnt*) unset TASR_LM; L=rnnt16; CFG=sdkconfig.rnnt16 ;;
  *nemo4*) L=nemo4_16; CFG=sdkconfig.nemo4
     if [ -z "$NO_LM" ] && [ -f "$LMF" ]; then export TASR_LM=$LMF; else unset TASR_LM; fi ;;
  *) if [ -z "$NO_LM" ] && [ -f "$LMF" ]; then export TASR_LM=$LMF; L=nemo16lm; CFG=sdkconfig.nemo16lm
     else unset TASR_LM; L=nemo16bench; CFG=sdkconfig.nemo16bench; fi ;;
esac
B=firmware/build_$L
if [ ! -f $B/tinyasr_fw.bin ] || [ -n "$REBUILD" ]; then
  (. "${IDF_PATH:-$HOME/esp/esp-idf}/export.sh" > /dev/null 2>&1; cd firmware
   idf.py -B build_$L -D SDKCONFIG=sdkconfig_$L -D SDKCONFIG_DEFAULTS="sdkconfig.defaults;$CFG" build > /dev/null)
fi
echo "emulating the ESP32-S3 firmware (takes ~2-3 s of wall time per second of audio)..."
TASR_BUILD=$B TASR_LAYOUT=$L tools/run_qemu.sh "$MODEL" /tmp/emu_img "${ARGS[@]}" > /tmp/emu_out.txt
grep -E "^HYP|^UTT" /tmp/emu_out.txt | sed -E 's/^HYP: /  text: /; s/ \| cycles.*//'
# QEMU with -icount: the cycle counter reads 0.04 x instructions summed over both cores; ~55% of them are on the
# dual-core critical path; 1.3-1.6 cycles/instruction at 240 MHz, plus ~0.08 s/s of flash/PSRAM stalls (estimate)
"$PY" - <<'PY'
import re
t = open("/tmp/emu_out.txt").read()
m = re.search(r"TOTAL audio ([\d.]+)s cycles (\d+)", t)
if m:
    dur, cc = float(m.group(1)), int(m.group(2))
    crit = cc * 25 * 0.55
    lo, hi = crit * 1.3 / 240e6 / dur + 0.08, crit * 1.6 / 240e6 / dur + 0.08
    print(f"{cc*25/1e6:.0f} M instructions for {dur:.1f} s of audio -> estimated ESP32-S3 RTF {lo:.2f}-{hi:.2f}")
PY
