#!/bin/bash
# Build + flash firmware, model and LM to an ESP32-S3 N16R8 (16 MB flash, 8 MB octal PSRAM).
# usage: tools/flash.sh <port> <model.tnm|model.tasr> [lm.tlm] [clip.wav "reference" ...]
#   CTC models use models/nemo_lm.tlm by default when it exists; NO_LM=1 flashes greedy decoding only
#   TASR_MODE=mic  (default) live INMP441 microphone: speak, the transcript prints after each pause
#   TASR_MODE=file transcribe the given clips from flash and print WER and the measured real-time factor
#   TASR_USB_HID=1  native USB dictation keyboard; BOOT toggles output (starts paused)
#   TASR_OLED=1    also drive an SSD1306 128x64 I2C OLED (SDA GPIO8, SCL GPIO9) with the live transcript
set -e
if [ "$#" -lt 2 ]; then
    echo "usage: $0 <serial-port> <model.tnm|model.tasr> [lm.tlm] [clip.wav reference ...]" >&2
    exit 2
fi
PORT=$1; MODEL=$(cd "$(dirname "$2")" && pwd)/$(basename "$2"); shift 2
LM=""; if [[ "${1:-}" == *.tlm ]]; then LM=$(cd "$(dirname "$1")" && pwd)/$(basename "$1"); shift; fi
DEFLM="$(cd "$(dirname "$0")/../.." && pwd)/models/nemo_lm.tlm"
[ -z "$LM" ] && [ -z "$NO_LM" ] && [[ "$MODEL" == *.tnm ]] && [ -f "$DEFLM" ] && LM=$DEFLM
[ -n "$NO_LM" ] && LM=""
WAVS=(); REFS=()
while [ $# -gt 1 ]; do WAVS+=("$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"); REFS+=("$2"); shift 2; done
MODE=${TASR_MODE:-mic}
case "$MODE" in mic|file) ;; *) echo "TASR_MODE must be mic or file" >&2; exit 2 ;; esac
if [ "${TASR_USB_HID:-0}" = 1 ] && [ "$MODE" != mic ]; then
    echo "TASR_USB_HID=1 requires TASR_MODE=mic" >&2; exit 2
fi
PY=${PYTHON:-$(command -v python3)}   # needs numpy + soundfile (ESP-IDF's own python does not have them)
cd "$(dirname "$0")/.."
. "${IDF_PATH:-$HOME/esp/esp-idf}/export.sh" > /dev/null
case "$MODEL" in
  *nemo4*.tnm) LAYOUT=nemo4_16; CFG="sdkconfig.defaults;sdkconfig.nemo4" ;;    # int4: 6 MB left for your app
  *rnnt*.tnm) LAYOUT=rnnt16; CFG="sdkconfig.defaults;sdkconfig.rnnt16"; LM="" ;;   # transducer: no external LM
  *.tnm)      LAYOUT=nemo16lm; CFG="sdkconfig.defaults;sdkconfig.nemo16lm" ;;
  *)          LAYOUT=tinyasr16; CFG="sdkconfig.defaults" ;;
esac
[ "$MODE" = mic ] && CFG="$CFG;sdkconfig.mic"
SUF=""; [ -n "$TASR_OLED" ] && CFG="$CFG;sdkconfig.oled" && SUF=_oled
if [ "${TASR_USB_HID:-0}" = 1 ]; then CFG="$CFG;sdkconfig.usb_hid"; SUF="${SUF}_hid"; fi
B=build_${LAYOUT}_${MODE}${SUF}
(cd firmware && idf.py -B $B -D SDKCONFIG=sdkconfig_${LAYOUT}_${MODE}${SUF} -D SDKCONFIG_DEFAULTS="$CFG" build)
OUT=build_images/${LAYOUT}_${MODE}; mkdir -p $OUT
LMARG=""; [ -n "$LM" ] && LMARG="--lm $LM"
"$PY" tools/mkimages.py "$MODEL" $OUT --layout $LAYOUT --wavs "${WAVS[@]}" --refs "${REFS[@]}" $LMARG --merge firmware/$B --flash_mode keep
esptool.py --chip esp32s3 -p "$PORT" -b 921600 write_flash 0x0 $OUT/flash.bin
echo "flashed ($LAYOUT, $MODE). monitor: (cd firmware && idf.py -B $B -p $PORT monitor)"

if [ "${TASR_USB_HID:-0}" = 1 ]; then
    echo 'Disconnect the UART cable, then connect the native USB port to the host.'
    echo 'Select the US keyboard layout and focus a text editor. After model load, press BOOT to arm.'
    echo 'Wait for the transcript to finish typing before pressing BOOT again to pause/cancel.'
fi
