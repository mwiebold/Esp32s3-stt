"""Export the GRU LM to the TLM1 blob (int8 per-row weights) for the C decoder."""
import struct, sys
import numpy as np, torch


def qlin(buf, w, b):
    w = w.detach().float()
    n, k = w.shape
    kp = (k + 15) // 16 * 16
    s = w.abs().amax(1).clamp(min=1e-8) / 127
    q = torch.clamp(torch.round(w / s[:, None]), -127, 127).to(torch.int8)
    qp = torch.zeros(n, kp, dtype=torch.int8); qp[:, :k] = q
    def align():
        while len(buf) % 16: buf.extend(b"\0")
    align(); buf.extend(qp.numpy().tobytes())
    align(); buf.extend(s.numpy().astype("<f4").tobytes())
    align(); buf.extend((b if b is not None else torch.zeros(n)).detach().float().numpy().astype("<f4").tobytes())


st = torch.load(sys.argv[1], map_location="cpu")
sd, cfg = st["model"], st["cfg"]
buf = bytearray(b"TLM1" + struct.pack("<7I", cfg["vocab"], cfg["emb"], cfg["hid"], 0, 0, 0, 0))
qlin(buf, sd["emb.weight"], None)
qlin(buf, sd["gru.weight_ih_l0"], sd["gru.bias_ih_l0"])
qlin(buf, sd["gru.weight_hh_l0"], sd["gru.bias_hh_l0"])
qlin(buf, sd["out.weight"], sd["out.bias"])
open(sys.argv[2], "wb").write(buf)
print(f"wrote {sys.argv[2]}: {len(buf)/1e6:.3f} MB")
