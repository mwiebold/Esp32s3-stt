"""Parse wall-time measurements separately from calibrated QEMU counter estimates."""
from __future__ import annotations
import re

_RECORD = re.compile(
    r"UTT (\d+) \| audio ([\d.]+)s \| cycle_delta32 (\d+) \| elapsed_us (\d+) \| RTF [\d.]+\n"
    r"REF: ([^\n]*)\nHYP: ([^\n]*)"
)


def parse_utterances(output: str) -> list[dict]:
    rows = []
    for idx, audio, cycles, elapsed, ref, hyp in _RECORD.findall(output):
        seconds, us = float(audio), int(elapsed)
        if seconds <= 0:
            raise ValueError('Audio duration must be positive')
        rows.append(dict(index=int(idx), audio_s=seconds, cycle_delta32=int(cycles),
                         elapsed_us=us, rtf=us / 1e6 / seconds, ref=ref, hyp=hyp))
    if not rows:
        raise ValueError('No benchmark records; rebuild firmware with the current timing format')
    if len({r['index'] for r in rows}) != len(rows):
        raise ValueError('Duplicate utterance records')
    return rows


def qemu_instruction_scale(output: str) -> float:
    """Reject physical-board counters; validate the pinned emulator's 25x calibration.

    This is an emulator-specific estimator, not a cycle-accurate silicon measurement.
    Calibration loop/task overhead prevents using the observed ratio as an exact scale.
    """
    match = re.search(r'CALIB 4M nops: cycles (\d+), time (\d+) us', output)
    if not match:
        raise ValueError('Missing QEMU calibration; refusing to infer instructions')
    cycles, us = map(int, match.groups())
    if not 144000 <= cycles <= 176000 or us <= 0:
        raise ValueError('Not the supported QEMU -icount calibration; do not apply the 25x conversion')
    return 25.0


def estimated_rows(output: str, critical: float = .55, cpi: tuple[float, float] = (1.3, 1.6),
                   stall: float = .08, hang: float = .82) -> list[dict]:
    scale = qemu_instruction_scale(output)
    rows = parse_utterances(output)
    match = re.search(r'CALIB 4M nops: cycles (\d+), time (\d+) us', output)
    ticks, calibration_us = map(int, match.groups())
    # Use the observed timer/counter ratio rather than assuming an emulator
    # timer frequency. Reject intervals within 10% of a full counter period.
    safe_counter_us = .9 * (2**32) * calibration_us / ticks
    for row in rows:
        if row['elapsed_us'] >= safe_counter_us:
            raise ValueError('QEMU diagnostic counter interval may have wrapped')
        instructions = row['cycle_delta32'] * scale
        compute = [instructions * critical * c / 240e6 + stall * row['audio_s'] for c in cpi]
        row.update(instr_M=instructions / 1e6, compute_s=compute,
                   text_after_s=[hang + x for x in compute], estimated=True)
    return rows
