"""Download NVIDIA's stt_en_conformer_ctc_small (CC-BY-4.0) and unpack it for nemo_small.py / export_nemo.py.

python fetch_nemo_small.py ../models/nemo_small                     # Conformer-CTC Small (Hugging Face, CC-BY-4.0)
python fetch_nemo_small.py ../models/nemo_rnnt --transducer        # Conformer-Transducer Small (NGC, NGC terms of use)
Writes state_dict_plain.pt (plain tensor dict of model_weights.ckpt), tokenizer.model and model_config.yaml.
"""
import io, os, sys, tarfile, urllib.request

import torch

URL = "https://huggingface.co/nvidia/stt_en_conformer_ctc_small/resolve/main/stt_en_conformer_ctc_small.nemo"
URL_RNNT = ("https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_transducer_small/versions/1.6.0/files/"
            "stt_en_conformer_transducer_small.nemo")


def main(out, url=URL):
    os.makedirs(out, exist_ok=True)
    print("downloading", url)
    blob = urllib.request.urlopen(url).read()
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        for m in tar.getmembers():
            name = os.path.basename(m.name)
            data = tar.extractfile(m).read() if m.isfile() else None
            if data is None:
                continue
            if name == "model_weights.ckpt":
                sd = torch.load(io.BytesIO(data), map_location="cpu")
                torch.save({k: v.detach().clone() for k, v in sd.items()}, os.path.join(out, "state_dict_plain.pt"))
            elif name.endswith("tokenizer.model"):
                open(os.path.join(out, "tokenizer.model"), "wb").write(data)
            elif name == "model_config.yaml":
                open(os.path.join(out, name), "wb").write(data)
    print("wrote", sorted(os.listdir(out)))


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    rnnt = "--transducer" in sys.argv
    default = os.path.join(os.path.dirname(__file__), "..", "models", "nemo_rnnt" if rnnt else "nemo_small")
    main(args[0] if args else default, URL_RNNT if rnnt else URL)
