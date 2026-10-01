"""External small-model baselines on our test sets with the same normalization (for context in the report).

python eval_external.py whisper-tiny.en|moonshine-tiny|vosk-small|nemo-small [--limit N] [--manifest a.jsonl b.jsonl ...]
"""
import argparse, json, os, sys, time
import numpy as np, soundfile as sf
from wer_utils import load_librispeech, wer

VAL = os.path.join(os.path.dirname(__file__), "..", "data", "val")


def sets(limit, manifests=None):
    out = []
    if manifests:
        for m in manifests:
            its = [json.loads(l) for l in open(m)]
            its = [(d["id"], d["wav"], d["text"]) for d in its]
            out.append((os.path.basename(m).replace(".jsonl", ""), its[:: max(1, len(its) // limit)][:limit] if limit else its))
        return out
    for s in ["test-clean", "test-other"]:
        its = load_librispeech(s)
        out.append((s, its[:: max(1, len(its) // limit)][:limit] if limit else its))
    for v in ["val_cv", "val_vox", "val_ami", "val_ps", "val_e22"]:
        its = [json.loads(l) for l in open(os.path.join(VAL, v + ".local.jsonl"))]
        its = [(d["id"], d["wav"], d["text"]) for d in its]
        out.append((v, its[:: max(1, len(its) // limit)][:limit] if limit else its))
    return out


_vm = None


def _vosk_init(vdir):
    global _vm
    import vosk
    vosk.SetLogLevel(-1)
    _vm = vosk.Model(vdir)


def _vosk_run(x):
    import vosk
    rec = vosk.KaldiRecognizer(_vm, 16000)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
    for i in range(0, len(pcm), 8000):
        rec.AcceptWaveform(pcm[i:i + 8000])
    return json.loads(rec.FinalResult()).get("text", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--manifest", nargs="*", default=[])
    a = ap.parse_args()
    if a.model.startswith("whisper"):
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        name = "openai/" + a.model
        proc = WhisperProcessor.from_pretrained(name)
        model = WhisperForConditionalGeneration.from_pretrained(name).eval()
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(dev)

        def transcribe(batch):
            feats = proc([b for b in batch], sampling_rate=16000, return_tensors="pt").input_features.to(dev)
            with torch.no_grad():
                ids = model.generate(feats, max_new_tokens=200)
            return proc.batch_decode(ids, skip_special_tokens=True)
        bs = 16
    elif a.model == "nemo-small":
        import sherpa_onnx
        d = os.path.join(os.path.dirname(__file__), "..", "models", "sherpa-onnx-nemo-ctc-en-conformer-small")
        rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(model=f"{d}/model.onnx", tokens=f"{d}/tokens.txt", num_threads=4)

        def transcribe(batch):
            out = []
            for b in batch:
                st = rec.create_stream()
                st.accept_waveform(16000, b)
                rec.decode_stream(st)
                out.append(st.result.text)
            return out
        bs = 1
    elif a.model == "vosk-small":
        import json as _j
        from concurrent.futures import ProcessPoolExecutor
        vdir = os.path.join(os.path.dirname(__file__), "..", "models", "ext", "vosk-model-small-en-us-0.15")
        pool = ProcessPoolExecutor(10, initializer=_vosk_init, initargs=(vdir,))

        def transcribe(batch):
            return list(pool.map(_vosk_run, batch))
        bs = 40
    else:
        from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
        model = MoonshineOnnxModel(model_name="moonshine/tiny")
        tok = load_tokenizer()

        def transcribe(batch):
            return [tok.decode_batch(model.generate(b[None].astype(np.float32)))[0] for b in batch]
        bs = 1
    res = {}
    for name, items in sets(a.limit, a.manifest):
        t = time.time()
        hyps = []
        for i in range(0, len(items), bs):
            chunk = [sf.read(p, dtype="float32")[0] for _, p, _ in items[i:i + bs]]
            chunk = [c[:, 0] if c.ndim > 1 else c for c in chunk]
            hyps += transcribe(chunk)
        w, _ = wer([r for _, _, r in items], hyps)
        res[name] = round(w, 2)
        print(f"{a.model} {name}: WER {w:.2f}% ({len(items)} utts, {time.time()-t:.0f}s)", flush=True)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
