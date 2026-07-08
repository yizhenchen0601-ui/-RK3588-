import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from pathlib import Path
import time
import numpy as np
from piper.voice import PiperVoice, SynthesisConfig
from config import *

TTS_CFG = SynthesisConfig(length_scale=0.85, noise_scale=0.667, noise_w_scale=0.8)

class PiperTTS:
    def __init__(self, model_path=PIPER_MODEL):
        print("Loading TTS...", end=" ", flush=True)
        t0 = time.time()
        self.voice = PiperVoice.load(model_path, download_dir=Path("/home/elf"))
        list(self.voice.synthesize("warmup.", syn_config=TTS_CFG))
        print(f"{time.time() - t0:.1f}s")

    def synthesize(self, text: str) -> bytes:
        import re
        text = re.sub(r"[^一-鿿\w\s。！？、，　；：—…·]", "", text).strip()
        if not text:
            return b""
        frames_float = []
        try:
            for chunk in self.voice.synthesize(text, syn_config=TTS_CFG):
                frames_float.append(chunk.audio_float_array.copy())
        except Exception as e:
            print(f"\n  [TTS] {e}")
            return b""
        if not frames_float:
            return b""
        audio_f32 = np.concatenate(frames_float)
        sr = TTS_SAMPLE_RATE
        trim = min(len(audio_f32), int(sr * 0.2))
        if trim:
            audio_f32 = audio_f32[:-trim]
        fade = min(len(audio_f32), int(sr * 0.01))
        if fade:
            audio_f32[-fade:] *= np.linspace(1.0, 0.0, fade)
        audio_i16 = np.clip(audio_f32 * 3.0 * 32767, -32768, 32767).astype(np.int16)
        return audio_i16.tobytes()
