"""Build the robustness benchmark: LibriSpeech test-clean utterances under real noise (DEMAND, CC-BY-4.0), babble and
simulated room reverb. Writes data/robust/<condition>/*.flac and data/robust/<condition>.jsonl manifests.

python make_robust.py [--n 300]
"""
import argparse, json, os, sys
import numpy as np, soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
sys.path.insert(0, HERE)
from wer_utils import load_librispeech  # noqa: E402

NOISES = {"car": "TCAR", "kitchen": "DKITCHEN", "living": "DLIVING", "cafe": "PCAFETER"}


def rms(x):
    return np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12)


def add_noise(s, n, snr_db):
    return s + n * (rms(s) / rms(n)) * 10 ** (-snr_db / 20)


def room_rir(rt60, rng):
    import pyroomacoustics as pra
    dims = [5.0, 4.0, 3.0]
    e_abs, max_order = pra.inverse_sabine(rt60, dims)
    room = pra.ShoeBox(dims, fs=16000, materials=pra.Material(e_abs), max_order=max_order)
    mic = np.array([1.0 + rng.uniform(0, 0.5), 1.0 + rng.uniform(0, 0.5), 1.0])
    ang = rng.uniform(0, np.pi / 2)
    src = mic + 2.0 * np.array([np.cos(ang), np.sin(ang), 0.0]) * [1.0, 1.0, 0] + [0, 0, 0.5]
    room.add_source(np.clip(src, 0.3, np.array(dims) - 0.3))
    room.add_microphone(mic)
    room.compute_rir()
    return np.array(room.rir[0][0], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    rng = np.random.default_rng(1234)
    its = load_librispeech("test-clean")
    its = its[:: max(1, len(its) // a.n)][: a.n]
    babble_src = load_librispeech("dev-clean")
    noise = {k: sf.read(os.path.join(DATA, "noise", v, "ch01.wav"), dtype="float64")[0] for k, v in NOISES.items()}
    conds = ["clean"] + [f"{k}_{s}db" for k in NOISES for s in (10, 5)] + ["babble_10db", "babble_5db",
                                                                            "reverb_0.4s", "reverb_0.8s", "farfield_kitchen"]
    out = os.path.join(DATA, "robust")
    man = {c: [] for c in conds}
    for c in conds:
        os.makedirs(os.path.join(out, c), exist_ok=True)
    for i, (uid, path, ref) in enumerate(its):
        s, _ = sf.read(path, dtype="float64")
        L = len(s)
        for c in conds:
            if c == "clean":
                y = s
            elif c.startswith("babble"):
                b = np.zeros(L)
                for j in rng.choice(len(babble_src), 4, replace=False):
                    x, _ = sf.read(babble_src[j][1], dtype="float64")
                    x = np.tile(x, int(np.ceil(L / len(x))))[:L]
                    b += x / rms(x)
                y = add_noise(s, b, int(c.split("_")[1][:-2]))
            elif c.startswith("reverb") or c == "farfield_kitchen":
                rt = 0.6 if c == "farfield_kitchen" else float(c.split("_")[1][:-1])
                y = np.convolve(s, room_rir(rt, rng))[:L]
                if c == "farfield_kitchen":
                    n = noise["kitchen"]
                    o = rng.integers(0, len(n) - L)
                    y = add_noise(y, n[o:o + L], 10)
            else:
                k, snr = c.split("_")
                n = noise[k]
                o = rng.integers(0, len(n) - L)
                y = add_noise(s, n[o:o + L], int(snr[:-2]))
            y = y / max(1.0, np.abs(y).max() / 0.95)
            f = os.path.join(out, c, uid + ".flac")
            sf.write(f, y.astype(np.float32), 16000, subtype="PCM_16")
            man[c].append({"id": uid, "wav": os.path.abspath(f), "text": ref.lower(), "duration": L / 16000})
        if i % 50 == 0:
            print(i, flush=True)
    for c in conds:
        with open(os.path.join(out, c + ".jsonl"), "w") as fh:
            for d in man[c]:
                fh.write(json.dumps(d) + "\n")
    print("conditions:", conds)


if __name__ == "__main__":
    main()
