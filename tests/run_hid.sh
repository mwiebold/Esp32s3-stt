#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
CC=${CC:-cc}
FLAGS=(-std=c11 -Wall -Wextra -Werror -Iesp32/firmware/main)
SRC=(tests/test_hid.c esp32/firmware/main/hid_dictation_core.c)
"$CC" "${FLAGS[@]}" -O2 "${SRC[@]}" -o "$OUT/hid"
"$OUT/hid"
"$CC" "${FLAGS[@]}" -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined "${SRC[@]}" -o "$OUT/hid-sanitized"
ASAN_OPTIONS=detect_leaks=1 "$OUT/hid-sanitized"
bash -n esp32/tools/flash.sh
