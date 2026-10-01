"""Full benchmark of the deployed C engine (bit-identical arithmetic to the ESP32 build).

python benchmark.py model.tasr --lm lm.tlm --out results.json [--quick]
Runs: LibriSpeech test-clean/test-other + held-out Common Voice, VoxPopuli, AMI, People's Speech, Earnings-22,
      with greedy CTC and with beam+LM, streaming chunk=32 (1.28 s) / left=4.
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
VAL = os.path.join(HERE, "..", "..", "data", "val")
SETS = ["val_cv", "val_vox", "val_ami", "val_ps", "val_e22"]


def run(model, extra, limit):
    cmd = [sys.executable, os.path.join(HERE, "eval_engine.py"), model, "--sets", "test-clean", "test-other",
           "--manifest"] + [os.path.join(VAL, s + ".local.jsonl") for s in SETS] + extra
    if limit:
        cmd += ["--limit", str(limit)]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE).stdout
    for line in out.splitlines():
        if "WER" in line:
            print("   ", line, flush=True)
    return json.loads(out.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lm", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--quick", action="store_true", help="subsample 300 utts per set")
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--left", type=int, default=2)
    ap.add_argument("--lm_weight", type=float, default=0.5)
    ap.add_argument("--token_bonus", type=float, default=2.0)
    ap.add_argument("--skip_greedy", action="store_true")
    a = ap.parse_args()
    lim = 300 if a.quick else 0
    base = ["--chunk", str(a.chunk), "--left", str(a.left)]
    res = {"model": a.model, "chunk": a.chunk, "left": a.left}
    if not a.skip_greedy:
        print("greedy:", flush=True)
        res["greedy"] = run(a.model, base, lim)
    if a.lm:
        print(f"beam4 + LM ({a.lm_weight}, {a.token_bonus}):", flush=True)
        res["beam_lm"] = run(a.model, base + ["--beam", "4", "--lm", a.lm, "--lm_weight", str(a.lm_weight),
                                              "--token_bonus", str(a.token_bonus)], lim)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
