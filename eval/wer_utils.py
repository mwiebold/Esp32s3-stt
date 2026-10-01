"""Shared WER utilities: LibriSpeech loading + text normalization + scoring."""
import os, re, glob
import jiwer

LS_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "LibriSpeech")


def load_librispeech(split, limit=None):
    """Returns list of (utt_id, flac_path, reference_text)."""
    items = []
    for trans in sorted(glob.glob(os.path.join(LS_ROOT, split, "*", "*", "*.trans.txt"))):
        d = os.path.dirname(trans)
        with open(trans) as f:
            for line in f:
                uid, txt = line.strip().split(" ", 1)
                items.append((uid, os.path.join(d, uid + ".flac"), txt))
    return items[:limit] if limit else items


_num_re = re.compile(r"[^a-z0-9' ]+")


FILLERS = {"uh", "um", "umm", "uhm", "mm", "mmm", "hmm", "hm", "ah", "er", "eh", "erm", "inaudible", "crosstalk",
           "laughter", "noise", "unintelligible"}


def normalize(t):
    t = t.lower().replace("’", "'")
    t = _num_re.sub(" ", t)
    t = " ".join("okay" if w == "ok" else w for w in t.split() if w not in FILLERS)
    return t


def wer(refs, hyps):
    refs = [normalize(r) for r in refs]
    hyps = [normalize(h) for h in hyps]
    o = jiwer.process_words(refs, hyps)
    n = sum(len(r.split()) for r in refs)
    return 100.0 * (o.substitutions + o.deletions + o.insertions) / max(n, 1), o
