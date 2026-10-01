"""Export a trained checkpoint to the TASR binary consumed by the C engine (host + ESP32).

python export.py ckpt.pt bpe256.model out.tasr --bits 8
"""
import argparse, struct
import numpy as np
import sentencepiece as spm
import torch

MAGIC = b"TASR"
VERSION = 2  # v2: int4 layers with N % 16 == 0 use the blocked16 layout


class W:
    def __init__(self):
        self.buf = bytearray()

    def align(self, a=16):
        while len(self.buf) % a:
            self.buf += b"\0"

    def f32(self, t):
        self.align()
        self.buf += np.ascontiguousarray(t.detach().cpu().float().numpy(), dtype="<f4").tobytes()

    def qlin(self, w, b, bits):
        """w (N,K) float -> symmetric per-row quant. Rows padded to Kp (x16 for int8, x32 for int4)."""
        w = w.detach().cpu().float().reshape(w.shape[0], -1)
        n, k = w.shape
        qmax = 2 ** (bits - 1) - 1
        s = w.abs().amax(1).clamp(min=1e-8) / qmax
        q = torch.clamp(torch.round(w / s[:, None]), -qmax, qmax).to(torch.int8)
        blocked = bits == 4 and n % 16 == 0
        blk = 16 if (bits == 8 or blocked) else 32
        kp = (k + blk - 1) // blk * blk
        qp = torch.zeros(n, kp, dtype=torch.int8)
        qp[:, :k] = q
        self.align()
        if bits == 8:
            self.buf += qp.numpy().tobytes()
        elif blocked:
            # per block of 16 outputs, per pair of inputs (k, k+1): byte j = (w[n0+j][k]+8) | (w[n0+j][k+1]+8) << 4
            u = (qp.to(torch.int16) + 8).to(torch.uint8).view(n // 16, 16, kp // 2, 2)   # (B, j, kpair, 2)
            packed = (u[..., 0] | (u[..., 1] << 4)).permute(0, 2, 1).contiguous()     # (B, kpair, j)
            self.buf += packed.numpy().tobytes()
        elif bits == 4:
            u = (qp.to(torch.int16) + 8).to(torch.uint8).view(n, kp // 32, 2, 16)
            packed = (u[:, :, 0, :] | (u[:, :, 1, :] << 4)).to(torch.uint8)  # byte j: lo=w[j], hi=w[j+16]
            self.buf += packed.numpy().tobytes()
        else:
            raise ValueError(bits)
        self.f32(s)
        self.f32(b if b is not None else torch.zeros(n))
        return q.float() * s[:, None]


def export(ckpt, sp_path, out, bits=8, head_bits=8, fe_bits=8, use_ema=True):
    st = torch.load(ckpt, map_location="cpu")
    cfg = st["cfg"]
    sd = st["ema"] if (use_ema and "ema" in st) else st["model"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    sp = spm.SentencePieceProcessor(model_file=sp_path)
    V = cfg["vocab"]
    assert sp.get_piece_size() == V
    w = W()
    hdr = [VERSION, V, cfg["d"], cfg["h"], cfg["ff"], cfg["k"], cfg["n_a"], cfg["n_b"], cfg["n_c"], cfg["frontend_c"], 80,
           bits, head_bits, fe_bits]
    w.buf += MAGIC + struct.pack("<" + "I" * 15, *hdr, 0)
    w.f32(st["fe_mean"]); w.f32(st["fe_inv_std"])
    from features import mel_matrix
    w.f32(mel_matrix().reshape(-1))  # (257, 80) row-major
    # frontend
    w.f32(sd["frontend.conv1.weight"].reshape(-1)); w.f32(sd["frontend.conv1.bias"])
    w.f32(sd["frontend.dw.weight"].reshape(-1)); w.f32(sd["frontend.dw.bias"])
    w.qlin(sd["frontend.pw.weight"].flatten(1), sd["frontend.pw.bias"], fe_bits)
    w.qlin(sd["frontend.out.weight"], sd["frontend.out.bias"], fe_bits)

    def ln(p):
        w.f32(sd[p + ".weight"]); w.f32(sd[p + ".bias"])

    def blk(p):
        ln(p + ".ln_ff1"); w.qlin(sd[p + ".ff1.l1.weight"], sd[p + ".ff1.l1.bias"], bits)
        w.qlin(sd[p + ".ff1.l2.weight"], sd[p + ".ff1.l2.bias"], bits)
        ln(p + ".ln_att"); w.qlin(sd[p + ".att.qkv.weight"], sd[p + ".att.qkv.bias"], bits)
        w.qlin(sd[p + ".att.out.weight"], sd[p + ".att.out.bias"], bits)
        ln(p + ".ln_conv"); w.qlin(sd[p + ".conv.pw1.weight"], sd[p + ".conv.pw1.bias"], bits)
        w.f32(sd[p + ".conv.dw.weight"].reshape(-1)); w.f32(sd[p + ".conv.dw.bias"])
        ln(p + ".conv.norm"); w.qlin(sd[p + ".conv.pw2.weight"], sd[p + ".conv.pw2.bias"], bits)
        ln(p + ".ln_ff2"); w.qlin(sd[p + ".ff2.l1.weight"], sd[p + ".ff2.l1.bias"], bits)
        w.qlin(sd[p + ".ff2.l2.weight"], sd[p + ".ff2.l2.bias"], bits)
        ln(p + ".ln_out")

    for i in range(cfg["n_a"]):
        blk(f"stage_a.{i}")
    w.f32(sd["down"].reshape(-1))
    for i in range(cfg["n_b"]):
        blk(f"stage_b.{i}")
    for i in range(cfg["n_c"]):
        blk(f"stage_c.{i}")
    w.qlin(sd["head.weight"], sd["head.bias"], head_bits)
    # tokens (id -> utf8 with '▁' -> ' ')
    w.align()
    for i in range(V):
        b = sp.id_to_piece(i).replace("▁", " ").encode()
        if sp.is_unknown(i) or sp.is_control(i):
            b = b""
        w.buf += bytes([len(b)]) + b
    open(out, "wb").write(w.buf)
    print(f"wrote {out}: {len(w.buf)/1e6:.3f} MB (bits={bits})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("sp"); ap.add_argument("out")
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--head_bits", type=int, default=8)
    ap.add_argument("--fe_bits", type=int, default=8)
    ap.add_argument("--model", action="store_true", help="use raw model weights instead of EMA")
    a = ap.parse_args()
    export(a.ckpt, a.sp, a.out, a.bits, a.head_bits, a.fe_bits, use_ema=not a.model)
