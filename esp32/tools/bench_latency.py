"""Voice-command latency benchmark: short utterances through the real firmware in QEMU (model + LM on the chip).

python bench_latency.py ../../models/nemo8.tnm [--n 10] [--out ../../research/latency_nemo8.json]
Picks N LibriSpeech test-clean utterances of 1.5-4 s (<= 22 s total, the nemo16lm audio partition), runs them in QEMU
and converts exact instruction counts into estimated ESP32-S3 compute time and time-to-text.
"""
import argparse, json, os, re, subprocess, sys
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "eval"))
from wer_utils import load_librispeech, wer  # noqa: E402

HANG = 0.8          # end-of-utterance silence the VAD waits for (s)
CRIT = 0.55         # share of instructions on the dual-core critical path
CPI = (1.3, 1.6)    # cycles per instruction range
STALL = 0.08        # flash/PSRAM stall allowance, s per s of audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="")
    ap.add_argument("--layout", default="nemo16lm", help="nemo16lm (CTC + LM) or rnnt16 (transducer, no LM)")
    ap.add_argument("--max_total", type=float, default=0)
    a = ap.parse_args()
    its = [(u, p, r) for u, p, r in load_librispeech("test-clean")]
    pick, tot = [], 0.0
    cands = [(u, p, r, sf.info(p).frames / 16000) for u, p, r in its[::7]]
    for u, p, r, d in cands:
        if 1.5 <= d <= 4.0 and tot + d <= (a.max_total or (22.0 if a.layout == "nemo16lm" else 19.0)):
            pick.append((u, p, r, d))
            tot += d
        if len(pick) == a.n:
            break
    args = []
    for u, p, r, d in pick:
        args += [os.path.abspath(p), r.lower()]
    env = dict(os.environ, TASR_BUILD=f"firmware/build_{a.layout}", TASR_LAYOUT=a.layout)
    if a.layout == "nemo16lm":
        env["TASR_LM"] = os.path.abspath(os.path.join(ROOT, "models", "nemo_lm.tlm"))
    else:
        env.pop("TASR_LM", None)
    out = subprocess.run([os.path.join(HERE, "run_qemu.sh"), os.path.abspath(a.model), "/tmp/qlat"] + args,
                         env=env, capture_output=True, text=True).stdout
    utts = re.findall(r"UTT (\d+) \| audio ([\d.]+)s \| cycles (\d+).*?\nREF: (.*)\nHYP: (.*)", out)
    rows = []
    for k, dur, cyc, ref, hyp in utts:
        dur, instr = float(dur), int(cyc) * 25
        comp = [instr * CRIT * c / 240e6 + STALL * dur for c in CPI]
        rows.append(dict(audio_s=dur, instr_M=instr / 1e6, compute_s=comp, text_after_s=[HANG + c for c in comp],
                         ref=ref, hyp=hyp))
    w = wer([r["ref"] for r in rows], [r["hyp"] for r in rows])[0]
    print(f"{os.path.basename(a.model)}: {len(rows)} utterances, {sum(r['audio_s'] for r in rows):.1f} s audio, WER {w:.1f}%")
    print(" audio | M instr | est. compute (s) | text after you stop (s) | hypothesis")
    for r in rows:
        print(f" {r['audio_s']:4.1f}s | {r['instr_M']:7.0f} | {r['compute_s'][0]:.1f}-{r['compute_s'][1]:.1f} | "
              f"{r['text_after_s'][0]:.1f}-{r['text_after_s'][1]:.1f} | {r['hyp'][:60]}")
    lo = sorted(r["text_after_s"][0] for r in rows)
    hi = sorted(r["text_after_s"][1] for r in rows)
    summ = dict(model=os.path.basename(a.model), n=len(rows), wer=w, median_text_after_s=[lo[len(lo) // 2], hi[len(hi) // 2]],
                max_text_after_s=[lo[-1], hi[-1]], rows=rows)
    print(f"median time-to-text {summ['median_text_after_s'][0]:.1f}-{summ['median_text_after_s'][1]:.1f} s, "
          f"worst {summ['max_text_after_s'][0]:.1f}-{summ['max_text_after_s'][1]:.1f} s")
    if a.out:
        json.dump(summ, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
