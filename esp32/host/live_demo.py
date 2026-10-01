"""Live demo of the ESP32-S3 recognizer on a laptop: microphone -> the firmware's own VAD/segmenter (tasr_seg.c) ->
the firmware's own Conformer-CTC engine + LM (tasr_nemo.c, tinyasr_lm.c). Same C code and arithmetic as the chip, so
the transcripts are what the board would print; the ESP32-S3 time is estimated from QEMU instruction counts.

python live_demo.py                        # speak; each utterance is transcribed after a 0.8 s pause
python live_demo.py --model fast           # int4 profile
python live_demo.py --wav session.wav      # run a recording through the same path instead of the microphone
python live_demo.py --save clips/          # also keep each utterance as a wav (e.g. to replay it in QEMU)
"""
import argparse, ctypes, os, queue, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "..", "models")
sys.path.insert(0, HERE)
import pytasr  # noqa: E402

lib = pytasr._lib
lib.tasr_seg_init.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.tasr_seg_feed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.tasr_seg_feed.restype = ctypes.c_int
lib.tasr_seg_next.argtypes = [ctypes.c_void_p]

# ESP32-S3 cost model, fitted to QEMU instruction counts of the firmware (int8 model, 4-20 s utterances):
# ccount = A*dur + B*dur^2 (attention grows with length^2); ccount x 25 = instructions summed over both cores,
# ~55% of them on the dual-core critical path; 1.3-1.6 cycles/instruction at 240 MHz; + ~0.08 s/s of flash/PSRAM stalls.
A_GREEDY, A_LM, B = 7.76e6, 8.6e6, 0.124e6


def esp32_seconds(dur, lm, int4):
    cc = (A_LM if lm else A_GREEDY) * dur + B * dur * dur
    if int4:
        cc *= 0.91
    crit = cc * 25 * 0.55
    return crit * 1.3 / 240e6 + 0.08 * dur, crit * 1.6 / 240e6 + 0.08 * dur


class Segmenter:
    def __init__(self, seconds=20):
        self.state = ctypes.create_string_buffer(256)  # tasr_seg_t (40 bytes) with room to spare
        self.buf = np.zeros(16000 * seconds, dtype=np.int16)
        lib.tasr_seg_init(self.state, self.buf.ctypes.data, len(self.buf))

    def feed(self, block):
        block = np.ascontiguousarray(block, dtype=np.int16)
        n = lib.tasr_seg_feed(self.state, block.ctypes.data, len(block))
        if not n:
            return None
        utt = self.buf[:n].copy()
        lib.tasr_seg_next(self.state)
        return utt

    def flush(self):  # end of input: hand over whatever speech is buffered
        n = int.from_bytes(self.state.raw[20:24], "little")  # tasr_seg_t.n
        if not self.in_speech or n < 3200:
            return None
        utt = self.buf[:n].copy()
        lib.tasr_seg_next(self.state)
        return utt

    @property
    def in_speech(self):
        return int.from_bytes(self.state.raw[24:28], "little") != 0  # tasr_seg_t.speech


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="accurate", help="accurate (int8), fast (int4), or a .tnm path")
    ap.add_argument("--no_lm", action="store_true")
    ap.add_argument("--wav", default="")
    ap.add_argument("--save", default="")
    ap.add_argument("--device", default=None, help="input device name/index (see python -m sounddevice)")
    a = ap.parse_args()
    path = {"accurate": os.path.join(MODELS, "nemo8.tnm"), "fast": os.path.join(MODELS, "nemo4.tnm")}.get(a.model, a.model)
    int4 = os.path.getsize(path) < 10_000_000
    lm_path = os.path.join(MODELS, "nemo_lm.tlm")
    if not a.no_lm and not os.path.exists(lm_path):
        print(f"note: {lm_path} not found, decoding greedily without the language model", flush=True)
    lm = pytasr.LM(lm_path) if not a.no_lm and os.path.exists(lm_path) else None
    eng = pytasr.Nemo(path, lm=lm, beam=4 if lm else 0, lm_weight=0.3, token_bonus=0.5)
    seg = Segmenter()
    if a.save:
        os.makedirs(a.save, exist_ok=True)
    print(f"model {os.path.basename(path)} ({'int4' if int4 else 'int8'}{' + LM' if lm else ', greedy'}), "
          f"same engine and VAD as the ESP32-S3 firmware", flush=True)
    k = 0

    def handle(utt):
        nonlocal k
        dur = len(utt) / 16000
        t0 = time.time()
        text = eng.transcribe(utt)
        host = time.time() - t0
        lo, hi = esp32_seconds(dur, lm is not None, int4)
        if sys.stdout.isatty():
            sys.stdout.write("\r" + " " * 40 + "\r")
        print(f"[{dur:4.1f} s] {text or '(nothing recognized)'}")
        print(f"         ESP32-S3 est. {lo:.1f}-{hi:.1f} s compute (RTF {lo/dur:.2f}-{hi/dur:.2f}), "
              f"text ~{0.8 + lo:.1f}-{0.8 + hi:.1f} s after you stop talking | laptop {host*1000:.0f} ms", flush=True)
        if a.save:
            import soundfile as sf
            sf.write(os.path.join(a.save, f"utt{k:03d}.wav"), utt, 16000, subtype="PCM_16")
        k += 1

    if a.wav:
        import soundfile as sf
        x, sr = sf.read(a.wav, dtype="int16")
        if x.ndim > 1:
            x = x[:, 0]
        assert sr == 16000, "need 16 kHz audio"
        for i in range(0, len(x) - 319, 320):
            utt = seg.feed(x[i:i + 320])
            if utt is not None:
                handle(utt)
        utt = seg.flush()
        if utt is not None:
            handle(utt)
        return

    import sounddevice as sd
    q = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=16000, channels=1, dtype="int16", blocksize=320, callback=cb, device=a.device):
        print("listening... speak, pause to get the transcript (Ctrl-C to quit)", flush=True)
        shown = False
        try:
            while True:
                utt = seg.feed(q.get())
                if seg.in_speech and not shown and sys.stdout.isatty():
                    sys.stdout.write("\r  (hearing speech...)")
                    sys.stdout.flush()
                    shown = True
                if utt is not None:
                    handle(utt)  # audio keeps queueing meanwhile, like the firmware's 8 s ring buffer
                    shown = False
        except KeyboardInterrupt:
            print()


if __name__ == "__main__":
    main()
