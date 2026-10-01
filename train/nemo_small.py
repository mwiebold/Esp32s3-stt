"""Standalone PyTorch reimplementation of NVIDIA NeMo stt_en_conformer_ctc_small (13 M params, CC-BY-4.0),
numerically matched to NeMo, used as the reference for the ESP32 port (no NeMo dependency).

Pipeline: preemph 0.97 -> centered STFT (n_fft 512, hop 160, win 400 symmetric Hann, zero pad) -> |X|^2 -> slaney mel
(80) -> log(x + 2^-24) -> per-utterance per-feature mean/std -> striding conv subsampling (2x conv3x3 s2, 176 ch)
-> 16 Conformer layers (rel-pos MHSA with untied biases, conv k=31 + BatchNorm, macaron FF) -> 1x1 conv -> CTC
(blank = last index).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NemoPre(nn.Module):
    def __init__(self, window, fb):
        super().__init__()
        self.register_buffer("window", window.float())      # (400,)
        self.register_buffer("fb", fb.float().reshape(80, 257))

    def forward(self, wav):
        """wav (N,) float -> (T, 80) normalized features, T = floor(N/160) + 1 (last frame zeroed)."""
        x = torch.cat([wav[:1], wav[1:] - 0.97 * wav[:-1]])
        spec = torch.stft(x, n_fft=512, hop_length=160, win_length=400, window=self.window, center=True,
                          pad_mode="constant", return_complex=True)
        p = spec.real.square() + spec.imag.square()          # (257, T)
        mel = torch.log(self.fb @ p + 2.0 ** -24)            # (80, T)
        n = wav.shape[0] // 160
        mean = mel[:, :n].mean(1, keepdim=True)
        std = torch.sqrt(((mel[:, :n] - mean) ** 2).sum(1, keepdim=True) / (n - 1)) + 1e-5
        out = (mel - mean) / std
        out[:, n:] = 0.0
        return out.t()


def rel_pos_emb(T, d):
    pos = torch.arange(T - 1, -T, -1, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * -(math.log(10000.0) / d))
    pe = torch.zeros(2 * T - 1, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # row m <-> relative position (T-1-m)


class Layer(nn.Module):
    def __init__(self, d=176, h=4, ff=704, k=31):
        super().__init__()
        self.h, self.dk = h, d // h
        self.norm_feed_forward1, self.norm_self_att, self.norm_conv = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.norm_feed_forward2, self.norm_out = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff1_l1, self.ff1_l2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.ff2_l1, self.ff2_l2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.linear_q, self.linear_k, self.linear_v, self.linear_out = [nn.Linear(d, d) for _ in range(4)]
        self.linear_pos = nn.Linear(d, d, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(h, self.dk))
        self.pos_bias_v = nn.Parameter(torch.zeros(h, self.dk))
        self.pw1 = nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)   # BatchNorm folded in
        self.pw2 = nn.Linear(d, d)

    def attn(self, x, pe):
        T, d = x.shape
        q = self.linear_q(x).view(T, self.h, self.dk)
        k = self.linear_k(x).view(T, self.h, self.dk).transpose(0, 1)
        v = self.linear_v(x).view(T, self.h, self.dk).transpose(0, 1)
        p = self.linear_pos(pe).view(2 * T - 1, self.h, self.dk).transpose(0, 1)  # (H, 2T-1, dk)
        ac = (q + self.pos_bias_u).transpose(0, 1) @ k.transpose(1, 2)           # (H, T, T)
        bd = (q + self.pos_bias_v).transpose(0, 1) @ p.transpose(1, 2)           # (H, T, 2T-1)
        # rel_shift: score[i, j] uses relative position (i - j) -> pe row (T-1) - (i-j)
        idx = (T - 1) - torch.arange(T)[:, None] + torch.arange(T)[None, :]
        bd = torch.gather(bd, 2, idx[None].expand(self.h, T, T))
        a = torch.softmax((ac + bd) / math.sqrt(self.dk), -1)
        return self.linear_out((a @ v).transpose(0, 1).reshape(T, d))

    def forward(self, x, pe):
        x = x + 0.5 * self.ff1_l2(F.silu(self.ff1_l1(self.norm_feed_forward1(x))))
        x = x + self.attn(self.norm_self_att(x), pe)
        c = F.glu(self.pw1(self.norm_conv(x)), dim=-1)
        c = F.silu(self.dw(c.t()[None])[0].t())
        x = x + self.pw2(c)
        x = x + 0.5 * self.ff2_l2(F.silu(self.ff2_l1(self.norm_feed_forward2(x))))
        return self.norm_out(x)


class NemoSmall(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.pre = NemoPre(sd["preprocessor.featurizer.window"], sd["preprocessor.featurizer.fb"])
        self.conv0 = nn.Conv2d(1, 176, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(176, 176, 3, stride=2, padding=1)
        self.sub_out = nn.Linear(3520, 176)
        self.layers = nn.ModuleList([Layer() for _ in range(16)])
        self.head = nn.Linear(176, 1025)
        self.load_nemo(sd)

    @torch.no_grad()
    def load_nemo(self, sd):
        g = lambda k: sd[k].float()
        self.conv0.weight.copy_(g("encoder.pre_encode.conv.0.weight")); self.conv0.bias.copy_(g("encoder.pre_encode.conv.0.bias"))
        self.conv2.weight.copy_(g("encoder.pre_encode.conv.2.weight")); self.conv2.bias.copy_(g("encoder.pre_encode.conv.2.bias"))
        self.sub_out.weight.copy_(g("encoder.pre_encode.out.weight")); self.sub_out.bias.copy_(g("encoder.pre_encode.out.bias"))
        for i, L in enumerate(self.layers):
            p = f"encoder.layers.{i}."
            for n in ["norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"]:
                getattr(L, n).weight.copy_(g(p + n + ".weight")); getattr(L, n).bias.copy_(g(p + n + ".bias"))
            for a, b in [("ff1_l1", "feed_forward1.linear1"), ("ff1_l2", "feed_forward1.linear2"),
                         ("ff2_l1", "feed_forward2.linear1"), ("ff2_l2", "feed_forward2.linear2"),
                         ("linear_q", "self_attn.linear_q"), ("linear_k", "self_attn.linear_k"),
                         ("linear_v", "self_attn.linear_v"), ("linear_out", "self_attn.linear_out")]:
                getattr(L, a).weight.copy_(g(p + b + ".weight")); getattr(L, a).bias.copy_(g(p + b + ".bias"))
            L.linear_pos.weight.copy_(g(p + "self_attn.linear_pos.weight"))
            L.pos_bias_u.copy_(g(p + "self_attn.pos_bias_u")); L.pos_bias_v.copy_(g(p + "self_attn.pos_bias_v"))
            L.pw1.weight.copy_(g(p + "conv.pointwise_conv1.weight")[:, :, 0]); L.pw1.bias.copy_(g(p + "conv.pointwise_conv1.bias"))
            L.pw2.weight.copy_(g(p + "conv.pointwise_conv2.weight")[:, :, 0]); L.pw2.bias.copy_(g(p + "conv.pointwise_conv2.bias"))
            s = g(p + "conv.batch_norm.weight") / torch.sqrt(g(p + "conv.batch_norm.running_var") + 1e-5)
            L.dw.weight.copy_(g(p + "conv.depthwise_conv.weight") * s[:, None, None])
            L.dw.bias.copy_((g(p + "conv.depthwise_conv.bias") - g(p + "conv.batch_norm.running_mean")) * s
                            + g(p + "conv.batch_norm.bias"))
        self.head.weight.copy_(g("decoder.decoder_layers.0.weight")[:, :, 0]); self.head.bias.copy_(g("decoder.decoder_layers.0.bias"))

    def encode(self, feats):
        """feats (T, 80) -> encoder output (T', 176)."""
        x = F.relu(self.conv0(feats[None, None]))
        x = F.relu(self.conv2(x))                       # (1, 176, T', 20)
        x = self.sub_out(x[0].permute(1, 0, 2).reshape(x.shape[2], -1))
        x = x * math.sqrt(176.0)
        pe = rel_pos_emb(x.shape[0], 176)
        for L in self.layers:
            x = L(x, pe)
        return x

    def forward(self, wav):
        f = self.pre(wav)
        n = wav.shape[0] // 160          # valid frames; the encoder sees all T = n+1 frames (last is zero)
        e = self.encode(f)
        return self.head(e), f, e        # logits (T', 1025), blank = 1024


if __name__ == "__main__":
    import sys, soundfile as sf
    d = sys.argv[1] if len(sys.argv) > 1 else "../models/nemo_small"
    import numpy as np
    sd = torch.load(f"{d}/state_dict_plain.pt", map_location="cpu")
    m = NemoSmall(sd).eval()
    a, _ = sf.read(sys.argv[2] if len(sys.argv) > 2 else
                   "../data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac", dtype="float32")
    with torch.no_grad():
        lg, f, e = m(torch.from_numpy(a))
    import os
    if not os.path.exists(f"{d}/ref_1089.npz"):  # NeMo reference tensors (optional parity check)
        print("frames", tuple(lg.shape), "argmax ids", lg.argmax(-1)[:20].tolist())
        sys.exit(0)
    ref = {k: torch.from_numpy(v) for k, v in np.load(f"{d}/ref_1089.npz").items()}
    rf, re_, rl = ref["feat"][0].t(), ref["enc"][0].t(), ref["logits"][0]
    print("feat", tuple(f.shape), tuple(rf.shape), "max|d|", (f[:rf.shape[0]] - rf).abs().max().item())
    print("enc ", tuple(e.shape), tuple(re_.shape), "max|d|", (e - re_).abs().max().item(), "scale", re_.abs().mean().item())
    print("logit", tuple(lg.shape), "max|d|", (lg.log_softmax(-1) - rl).abs().max().item(),
          "argmax agree", (lg.argmax(-1) == rl.argmax(-1)).float().mean().item())
