#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cc=${CC:-cc}
src=(esp32/components/tinyasr/{tinyasr,kernels,tinyasr_lm,tasr_nemo,tasr_seg}.c)
flags=(-O2 -D_DEFAULT_SOURCE -std=c11 -ffp-contract=off -DTASR_PROFILE -Wall -Wextra -Wno-unused-function
       -Iesp32/components/tinyasr/include -Iesp32/components/tinyasr)
"$cc" "${flags[@]}" tests/test_engine.c "${src[@]}" -pthread -lm -o "$work/test_engine"
"$work/test_engine"
if [[ ${SANITIZE:-1} == 1 ]]; then
    "$cc" "${flags[@]}" -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer \
        tests/test_engine.c "${src[@]}" -pthread -lm -o "$work/test_engine_asan"
    ASAN_OPTIONS=detect_leaks=1 "$work/test_engine_asan"
fi
python3 -m unittest discover -s tests -p 'test_*.py' -v
if "$cc" "${flags[@]}" -ffast-math -c esp32/components/tinyasr/kernels.c -o "$work/bad.o" 2> "$work/fast-math.txt"; then
    echo 'FAIL: unsafe fast-math build was accepted' >&2
    exit 1
fi
grep -q 'disable -ffast-math' "$work/fast-math.txt"
echo 'PASS unsafe-fast-math rejection'
