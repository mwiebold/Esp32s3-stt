"""ctypes wrapper around the host build of the tinyasr C engine (identical code to the ESP32 firmware)."""
import ctypes, os
import numpy as np

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libtinyasr.dylib"))
_lib.tasr_model_load.restype = ctypes.c_void_p
_lib.tasr_model_load.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_lib.tasr_model_weight_bytes.restype = ctypes.c_size_t
_lib.tasr_model_weight_bytes.argtypes = [ctypes.c_void_p]
_lib.tasr_stream_create.restype = ctypes.c_void_p
_lib.tasr_stream_create.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_lib.tasr_stream_feed.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
_lib.tasr_stream_finish.argtypes = [ctypes.c_void_p]
_lib.tasr_stream_text.restype = ctypes.c_char_p
_lib.tasr_stream_text.argtypes = [ctypes.c_void_p]
_lib.tasr_stream_reset.argtypes = [ctypes.c_void_p]
_lib.tasr_stream_free.argtypes = [ctypes.c_void_p]
_lib.tasr_stream_set_logit_sink.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


_lib.tasr_lm_load.restype = ctypes.c_void_p
_lib.tasr_lm_load.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_lib.tasr_decoder_create.restype = ctypes.c_void_p
_lib.tasr_decoder_create.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float]
_lib.tasr_stream_set_decoder.argtypes = [ctypes.c_void_p, ctypes.c_void_p]


def _aligned(path):
    raw = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
    buf = np.zeros(len(raw) + 64, dtype=np.uint8)
    off = (-buf.ctypes.data) % 16
    arr = buf[off:off + len(raw)]
    arr[:] = raw
    return buf, arr


class LM:
    def __init__(self, path):
        self._buf, self._arr = _aligned(path)
        self.ptr = _lib.tasr_lm_load(self._arr.ctypes.data, len(self._arr))
        assert self.ptr, "lm load failed"


class Model:
    def __init__(self, path):
        self.blob = ctypes.create_string_buffer(open(path, "rb").read())
        # ensure 16-byte alignment: copy into aligned numpy buffer
        raw = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
        self._buf = np.zeros(len(raw) + 64, dtype=np.uint8)
        off = (-self._buf.ctypes.data) % 16
        self._arr = self._buf[off:off + len(raw)]
        self._arr[:] = raw
        self.ptr = _lib.tasr_model_load(self._arr.ctypes.data, len(raw))
        assert self.ptr, "model load failed"

    def weight_bytes(self):
        return _lib.tasr_model_weight_bytes(self.ptr)


class Stream:
    def __init__(self, model, chunk=32, left=4, lm=None, beam=0, topk=6, lm_weight=0.3, token_bonus=1.0):
        self.model = model
        self.ptr = _lib.tasr_stream_create(model.ptr, chunk, left)
        assert self.ptr
        self.lm = lm
        if beam > 0:
            self.dec = _lib.tasr_decoder_create(lm.ptr if lm else None, 257, beam, topk, lm_weight, token_bonus)
            _lib.tasr_stream_set_decoder(self.ptr, self.dec)

    def transcribe(self, pcm_int16, feed=1600, logits=False, vocab=257):
        _lib.tasr_stream_reset(self.ptr)
        pcm = np.ascontiguousarray(pcm_int16, dtype=np.int16)
        sink = n = None
        if logits:
            sink = np.zeros((len(pcm) // 640 + 64, vocab), dtype=np.float32)
            n = ctypes.c_int(0)
            _lib.tasr_stream_set_logit_sink(self.ptr, sink.ctypes.data, sink.shape[0], ctypes.byref(n))
        for i in range(0, len(pcm), feed):
            seg = pcm[i:i + feed]
            _lib.tasr_stream_feed(self.ptr, seg.ctypes.data, len(seg))
        _lib.tasr_stream_finish(self.ptr)
        txt = _lib.tasr_stream_text(self.ptr).decode()
        if logits:
            _lib.tasr_stream_set_logit_sink(self.ptr, None, 0, ctypes.byref(ctypes.c_int(0)))
            return txt, sink[: n.value]
        return txt

    def __del__(self):
        try:
            _lib.tasr_stream_free(self.ptr)
        except Exception:
            pass


_lib.tasr_nemo_load.restype = ctypes.c_void_p
_lib.tasr_nemo_load.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_lib.tasr_nemo_weight_bytes.restype = ctypes.c_size_t
_lib.tasr_nemo_weight_bytes.argtypes = [ctypes.c_void_p]
_lib.tasr_nemo_transcribe.restype = ctypes.c_int
_lib.tasr_nemo_transcribe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p,
                                      ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]


class Nemo:
    """Utterance-level NeMo Conformer-CTC small engine (same C code as the ESP32 firmware)."""

    def __init__(self, path, lm=None, beam=0, topk=6, lm_weight=0.5, token_bonus=1.0):
        self._buf, self._arr = _aligned(path)
        self.ptr = _lib.tasr_nemo_load(self._arr.ctypes.data, len(self._arr))
        assert self.ptr, "nemo load failed"
        self.lm = lm
        self.dec = _lib.tasr_decoder_create(lm.ptr if lm else None, 1025, beam, topk, lm_weight, token_bonus) if beam else None

    def transcribe(self, pcm_int16, logits=False):
        pcm = np.ascontiguousarray(pcm_int16, dtype=np.int16)
        text = ctypes.create_string_buffer(8192)
        sink = n = None
        if logits:
            sink = np.zeros((len(pcm) // 640 + 8, 1025), dtype=np.float32)
            n = ctypes.c_int(0)
        _lib.tasr_nemo_transcribe(self.ptr, pcm.ctypes.data, len(pcm), self.dec, text, 8192,
                                  sink.ctypes.data if logits else None, sink.shape[0] if logits else 0,
                                  ctypes.byref(n) if logits else None)
        t = text.value.decode()
        return (t, sink[: n.value]) if logits else t
