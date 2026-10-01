"""NeMo Conformer-Transducer Small (14M): our encoder port (nemo_small.py) + RNN-T prediction/joint networks, greedy
decoding exactly as NeMo (max 5 symbols per frame). Used to validate the model before porting the decoder to C.

python rnnt_small.py ../models/nemo_rnnt --sets test-clean test-other [--limit N]
"""
import argparse, os, sys
import numpy as np, soundfile as sf, torch, torch.nn as nn, sentencepiece as spm
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from wer_utils import load_librispeech, wer  # noqa: E402
from nemo_small import NemoSmall  # noqa: E402

BLANK = 1024


class RnntHead(nn.Module):
    def __init__(self, sd, max_symbols=5):
        super().__init__()
        self.embed = nn.Embedding(1025, 320)
        self.lstm = nn.LSTM(320, 320, 1)
        self.enc = nn.Linear(176, 320)
        self.pred = nn.Linear(320, 320)
        self.out = nn.Linear(320, 1025)
        self.max_symbols = max_symbols
        with torch.no_grad():
            self.embed.weight.copy_(sd["decoder.prediction.embed.weight"])
            for n in ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"]:
                getattr(self.lstm, n).copy_(sd[f"decoder.prediction.dec_rnn.lstm.{n}"])
            for m, k in [(self.enc, "joint.enc"), (self.pred, "joint.pred"), (self.out, "joint.joint_net.2")]:
                m.weight.copy_(sd[k + ".weight"]); m.bias.copy_(sd[k + ".bias"])

    def step(self, y, state):
        x = torch.zeros(1, 1, 320) if y is None else self.embed(torch.tensor([[y]]))  # SOS = zero input
        g, state = self.lstm(x, state)
        return self.pred(g[0, 0]), state

    @torch.no_grad()
    def greedy(self, enc):  # enc: (T, 176)
        f = self.enc(enc)
        gp, state = self.step(None, None)
        out = []
        for t in range(f.shape[0]):
            for _ in range(self.max_symbols):
                k = int(self.out(torch.relu(f[t] + gp)).argmax())
                if k == BLANK:
                    break
                out.append(k)
                gp, state = self.step(k, state)
        return out


def load(d):
    sd = torch.load(os.path.join(d, "model_weights.ckpt"), map_location="cpu")
    sd = dict(sd)
    sd["decoder.decoder_layers.0.weight"] = torch.zeros(1025, 176, 1)  # NemoSmall builds a CTC head; unused here
    sd["decoder.decoder_layers.0.bias"] = torch.zeros(1025)
    enc = NemoSmall(sd).eval()
    tok = [f for f in os.listdir(d) if f.endswith("tokenizer.model")][0]
    return enc, RnntHead(sd).eval(), spm.SentencePieceProcessor(model_file=os.path.join(d, tok))


_m = None


def _init(d):
    global _m
    torch.set_num_threads(1)
    _m = load(d)


def _run(item):
    uid, path, ref = item
    enc, head, sp = _m
    a, _ = sf.read(path, dtype="float32")
    with torch.no_grad():
        _, _, e = enc(torch.from_numpy(a))
    return sp.decode(head.greedy(e)), ref


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--sets", nargs="*", default=["test-clean", "test-other"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    with ProcessPoolExecutor(10, initializer=_init, initargs=(a.model_dir,)) as ex:
        for s in a.sets:
            its = load_librispeech(s)
            if a.limit:
                its = its[:: max(1, len(its) // a.limit)][: a.limit]
            res = list(ex.map(_run, its, chunksize=4))
            print(f"rnnt fp32 {s}: WER {wer([r for _, r in res], [h for h, _ in res])[0]:.2f} ({len(its)} utts)", flush=True)
