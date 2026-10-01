"""WER of the deployed C engine (host build == ESP32 arithmetic) on LibriSpeech and other test sets.

python eval_engine.py model.tasr [--sets test-clean test-other] [--chunk 32 --left 4] [--limit N] [--manifest x.jsonl]
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np, soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "eval"))
from wer_utils import load_librispeech, wer  # noqa: E402

_stream = None


def _init(path, chunk, left, lm_path, beam, lw, tb):
    global _stream
    import pytasr
    lm = pytasr.LM(lm_path) if lm_path else None
    if path.endswith(".tnm"):  # NeMo conformer-ctc-small utterance engine
        _stream = pytasr.Nemo(path, lm=lm, beam=beam, lm_weight=lw, token_bonus=tb)
    else:
        _stream = pytasr.Stream(pytasr.Model(path), chunk, left, lm=lm, beam=beam, lm_weight=lw, token_bonus=tb)


def _run(item):
    uid, path, ref = item
    a, sr = sf.read(path, dtype="int16")
    if a.ndim > 1:
        a = a[:, 0]
    import pytasr
    if isinstance(_stream, pytasr.Nemo):
        return uid, _stream.transcribe(a), ref
    return uid, _stream.transcribe(a, feed=320), ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--sets", nargs="*", default=["test-clean", "test-other"])
    ap.add_argument("--manifest", nargs="*", default=[])
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--left", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--dump", default="")
    ap.add_argument("--lm", default="")
    ap.add_argument("--beam", type=int, default=0)
    ap.add_argument("--lm_weight", type=float, default=0.3)
    ap.add_argument("--token_bonus", type=float, default=1.0)
    a = ap.parse_args()
    def sub(items):
        return items[:: max(1, len(items) // a.limit)][: a.limit] if a.limit else items
    jobs = [(s, sub(load_librispeech(s))) for s in a.sets]
    for m in a.manifest:
        its = [json.loads(l) for l in open(m)]
        if a.limit:
            its = its[: a.limit]
        jobs.append((os.path.basename(m), [(d.get("id", str(i)), d["wav"], d["text"]) for i, d in enumerate(its)]))
    res = {}
    with Pool(a.procs, initializer=_init, initargs=(a.model, a.chunk, a.left, a.lm, a.beam, a.lm_weight, a.token_bonus)) as pool:
        for name, items in jobs:
            t = time.time()
            out = pool.map(_run, items, chunksize=4)
            w, o = wer([r for _, _, r in out], [h for _, h, _ in out])
            dur = sum(sf.info(p).duration for _, p, _ in items)
            print(f"{name}: WER {w:.2f}% (sub {o.substitutions} del {o.deletions} ins {o.insertions}) "
                  f"{len(items)} utts {dur/3600:.2f}h in {time.time()-t:.0f}s", flush=True)
            res[name] = w
            if a.dump:
                with open(f"{a.dump}.{name}.txt", "w") as f:
                    for uid, h, r in out:
                        f.write(f"{uid}\tREF: {r}\n{uid}\tHYP: {h}\n")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
