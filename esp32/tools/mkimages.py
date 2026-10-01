"""Build model/audio partition images and (optionally) a merged 16 MB flash image for QEMU.

python mkimages.py model.tasr out_dir --wavs a.flac b.flac --refs "text a" "text b" [--merge build_dir]
"""
import argparse, os, struct, subprocess, sys
import numpy as np, soundfile as sf

MODEL_OFF, MODEL_SIZE = 0x190000, 0x780000
LM_OFF, LM_SIZE = 0x910000, 0x180000
AUDIO_OFF, AUDIO_SIZE = 0xA90000, 0x570000


LAYOUTS = {
    "tinyasr16": dict(MODEL=(0x190000, 0x780000), LM=(0x910000, 0x180000), AUDIO=(0xA90000, 0x570000), FLASH="16MB"),
    "nemo16bench": dict(MODEL=(0x90000, 0xE10000), LM=None, AUDIO=(0xEA0000, 0x160000), FLASH="16MB"),
    "nemo16lm": dict(MODEL=(0x90000, 0xD60000), LM=(0xDF0000, 0x150000), AUDIO=(0xF40000, 0xC0000), FLASH="16MB"),
    "nemo4_16": dict(MODEL=(0x610000, 0x850000), LM=(0xE60000, 0x150000), AUDIO=(0xFB0000, 0x50000), FLASH="16MB"),
    "rnnt16": dict(MODEL=(0x70000, 0xEE0000), LM=None, AUDIO=(0xF50000, 0xB0000), FLASH="16MB"),
    "nemo32": dict(MODEL=(0x90000, 0xE10000), LM=(0xEA0000, 0x160000), AUDIO=(0x1000000, 0x800000), FLASH="32MB"),
}


def model_image(tasr):
    blob = open(tasr, "rb").read()
    img = struct.pack("<I", len(blob)) + b"\0" * 12 + blob
    assert len(img) <= MODEL_SIZE, f"model {len(img)} > partition {MODEL_SIZE}"
    return img


def audio_image(wavs, refs):
    out = bytearray(b"AUD0" + struct.pack("<I", len(wavs)))
    for w, r in zip(wavs, refs):
        a, sr = sf.read(w, dtype="int16")
        assert sr == 16000
        rb = r.encode()
        out += struct.pack("<II", len(a), len(rb)) + rb + b"\0" * ((-len(rb)) % 4)
        out += a.tobytes() + (b"\0\0" if len(a) % 2 else b"")
    assert len(out) <= AUDIO_SIZE, f"audio {len(out)} > {AUDIO_SIZE}"
    return bytes(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tasr"); ap.add_argument("out")
    ap.add_argument("--wavs", nargs="*", default=[]); ap.add_argument("--refs", nargs="*", default=[])
    ap.add_argument("--merge", default="")
    ap.add_argument("--lm", default="")
    ap.add_argument("--layout", default="tinyasr16")
    ap.add_argument("--flash_mode", default="dio", help="dio for QEMU; 'keep' (bootloader's QIO) for real boards")
    a = ap.parse_args()
    lay = LAYOUTS[a.layout]
    MODEL_OFF, MODEL_SIZE = lay["MODEL"]; AUDIO_OFF, AUDIO_SIZE = lay["AUDIO"]
    LM_OFF, LM_SIZE = lay["LM"] if lay["LM"] else (None, 0)
    globals().update(MODEL_SIZE=MODEL_SIZE, AUDIO_SIZE=AUDIO_SIZE)
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "model.bin"), "wb").write(model_image(a.tasr))
    open(os.path.join(a.out, "audio.bin"), "wb").write(audio_image(a.wavs, a.refs))
    lmb = open(a.lm, "rb").read() if a.lm else b""
    assert not lmb or len(lmb) + 16 <= LM_SIZE
    open(os.path.join(a.out, "lm.bin"), "wb").write(struct.pack("<I", len(lmb)) + b"\0" * 12 + lmb)
    if a.merge:
        b = a.merge
        cmd = ["esptool.py", "--chip", "esp32s3", "merge_bin", "--fill-flash-size", lay["FLASH"],
               "-o", os.path.join(a.out, "flash.bin"), "--flash_mode", a.flash_mode, "--flash_size", lay["FLASH"], "--flash_freq", "80m",
               "0x0", f"{b}/bootloader/bootloader.bin", "0x8000", f"{b}/partition_table/partition-table.bin",
               "0x10000", f"{b}/tinyasr_fw.bin", hex(MODEL_OFF), os.path.join(a.out, "model.bin"),
               hex(AUDIO_OFF), os.path.join(a.out, "audio.bin")]
        if LM_OFF is not None:
            cmd += [hex(LM_OFF), os.path.join(a.out, "lm.bin")]
        subprocess.check_call(cmd)
    print("images written to", a.out)
