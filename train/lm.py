"""Tiny GRU language model over the ASR's BPE-256 tokens (for on-device CTC shallow fusion).

prep:  python lm.py prep --sp bpe256.model --out lm_tokens.npy  (mixes LibriSpeech-LM text + ASR training transcripts)
train: python lm.py train --tokens lm_tokens.npy --out lm.pt
"""
import argparse, gzip, math, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BOS = 256  # also used as end-of-sentence (= vocab size of the ASR tokenizer; overridden by --sp)


class GRULM(nn.Module):
    def __init__(self, vocab=257, emb=256, hid=384):
        super().__init__()
        self.cfg = dict(vocab=vocab, emb=emb, hid=hid)
        self.emb = nn.Embedding(vocab, emb)
        self.gru = nn.GRU(emb, hid, batch_first=True)
        self.out = nn.Linear(hid, vocab)

    def forward(self, x, h=None):
        y, h = self.gru(self.emb(x), h)
        return self.out(y), h


def prep(a):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=a.sp)
    global BOS
    BOS = sp.get_piece_size()
    rng = random.Random(0)
    lines = []
    with gzip.open(a.ls_lm, "rt") as f:
        for i, l in enumerate(f):
            if rng.random() < a.ls_frac:
                lines.append(l.strip().lower())
    n_ls = len(lines)
    for src in [x for x in a.asr_text.split(",") if x]:  # --asr_text "" = LibriSpeech-LM (public domain) only
        op = gzip.open if src.endswith(".gz") else open
        with op(src, "rt") as f:
            for l in f:
                l = l.strip()
                if l:
                    lines.append(l)
    print(f"sentences: {n_ls} LibriSpeech-LM + {len(lines)-n_ls} ASR transcripts", flush=True)
    rng.shuffle(lines)
    out = []
    B = 200000
    for i in range(0, len(lines), B):
        for ids in sp.encode(lines[i:i + B]):
            out.append(BOS)
            out.extend(ids)
    arr = np.array(out + [BOS], dtype=np.int16)
    np.save(a.out, arr)
    print(f"tokens: {len(arr)/1e6:.1f}M -> {a.out}")


def train(a):
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    toks = torch.from_numpy(np.load(a.tokens).astype(np.int64))
    n_val = 2_000_000
    val, tr = toks[:n_val], toks[n_val:]
    V = int(toks.max().item()) + 1  # BOS is the largest id
    m = GRULM(vocab=V, hid=a.hid, emb=a.emb).to(dev)
    print(f"params {sum(p.numel() for p in m.parameters())/1e6:.3f}M on {dev}", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    L, B = a.seq, a.batch
    steps = a.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.03)
    t0 = time.time()

    def batch(src, bs):
        idx = torch.randint(0, len(src) - L - 1, (bs,))
        x = torch.stack([src[i:i + L] for i in idx.tolist()])
        y = torch.stack([src[i + 1:i + L + 1] for i in idx.tolist()])
        return x.to(dev), y.to(dev)

    for step in range(1, steps + 1):
        x, y = batch(tr, B)
        lo, _ = m(x)
        loss = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps:
            with torch.no_grad():
                m.eval()
                vl = 0
                for _ in range(10):
                    xv, yv = batch(val, 64)
                    lv, _ = m(xv)
                    vl += F.cross_entropy(lv.reshape(-1, lv.shape[-1]), yv.reshape(-1)).item()
                m.train()
            print(f"step {step} loss {loss.item():.3f} val {vl/10:.3f} ppl/token {math.exp(vl/10):.2f} "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
            torch.save({"model": m.state_dict(), "cfg": m.cfg}, a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--sp", default="../models/bpe256.model")
    ap.add_argument("--ls_lm", default="../data/lm/librispeech-lm-norm.txt.gz")
    ap.add_argument("--asr_text", default="../data/lm/train_text.txt.gz")
    ap.add_argument("--ls_frac", type=float, default=0.25)
    ap.add_argument("--out", default="../data/lm/lm_tokens.npy")
    ap.add_argument("--tokens", default="../data/lm/lm_tokens.npy")
    ap.add_argument("--hid", type=int, default=384)
    ap.add_argument("--emb", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=30000)
    a = ap.parse_args()
    {"prep": prep, "train": train}[a.cmd](a)
