"""Qwen3-TTS CustomVoice worker. Run with a python that has qwen-tts, torch, numpy, soundfile."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _proto import serve  # noqa: E402

STATE = {}


def load(req):
    import torch
    from qwen_tts import Qwen3TTSModel

    dev = req.get("device") or ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if (req.get("dtype", "float16") == "float16" and dev != "cpu") else torch.float32
    t0 = time.time()
    STATE["model"] = Qwen3TTSModel.from_pretrained(req["model"], device_map=dev, dtype=dtype)
    speakers = None
    try:
        speakers = STATE["model"].get_supported_speakers()
    except Exception:  # noqa: BLE001
        pass
    return {"device": dev, "dtype": str(dtype).replace("torch.", ""), "load_s": round(time.time() - t0, 1),
            "speakers": list(speakers) if speakers else None}


def synth(req):
    import numpy as np
    import soundfile as sf

    t0 = time.time()
    wavs, sr = STATE["model"].generate_custom_voice(
        text=req["text"], speaker=req["speaker"], language=req.get("language") or "Chinese",
        instruct=req.get("instruct") or None)
    wav = np.asarray(wavs[0], dtype=np.float32)
    sf.write(req["out"], wav, sr, subtype="PCM_16", format="WAV")
    return {"sr": int(sr), "duration_s": round(len(wav) / float(sr), 3), "elapsed_s": round(time.time() - t0, 2)}


def check(req):
    import torch
    import qwen_tts  # noqa: F401

    return {"torch": torch.__version__, "mps": bool(torch.backends.mps.is_available())}


if __name__ == "__main__":
    serve({"load": load, "synth": synth, "check": check})
