"""Qwen3-ForcedAligner worker. Run with a python that has qwen-asr, torch, numpy, soundfile."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _proto import serve  # noqa: E402

STATE = {}


def load(req):
    import torch
    from qwen_asr import Qwen3ForcedAligner

    dev = req.get("device") or ("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    STATE["model"] = Qwen3ForcedAligner.from_pretrained(req["model"], dtype=torch.float32, device_map=dev)
    return {"device": dev, "load_s": round(time.time() - t0, 1)}


def align(req):
    import soundfile as sf

    wav, sr = sf.read(req["audio"], dtype="float32", always_2d=False)
    if getattr(wav, "ndim", 1) == 2:
        wav = wav.mean(axis=1)
    t0 = time.time()
    res = STATE["model"].align(audio=(wav, sr), text=req["text"], language=req.get("language") or "Chinese")[0]
    items = getattr(res, "items", res)
    toks = [{"text": it.text, "start": float(it.start_time), "end": float(it.end_time)} for it in items]
    return {"tokens": toks, "elapsed_s": round(time.time() - t0, 2)}


def counts(req):
    proc = STATE["model"].aligner_processor
    lang = req.get("language") or "Chinese"
    return {"counts": [len(proc.encode_timestamp(t, lang)[0]) for t in req["texts"]]}


def check(req):
    import torch
    import qwen_asr  # noqa: F401

    return {"torch": torch.__version__, "mps": bool(torch.backends.mps.is_available())}


if __name__ == "__main__":
    serve({"load": load, "align": align, "counts": counts, "check": check})
