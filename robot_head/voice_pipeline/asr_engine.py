"""流式 ASR 引擎 — sherpa-onnx"""

import os
import re
import time
import numpy as np
from config import *

ASR_TAG_RE = re.compile(r"<\|[^|]+\|>")

# sherpa-onnx 模型文件映射（transducer）
ASR_FILES = {
    "14m_fp32": {
        "encoder": "encoder.fp32.onnx",
        "decoder": "decoder.fp32.onnx",
        "joiner": "joiner.fp32.onnx",
    },
    "14m_int8": {
        "encoder": "encoder.int8.onnx",
        "decoder": "decoder.int8.onnx",
        "joiner": "joiner.int8.onnx",
    },
    "xlarge": {
        "encoder": "encoder.int8.onnx",
        "decoder": "decoder.onnx",
        "joiner": "joiner.int8.onnx",
    },
}


class StreamingASR:
    """sherpa-onnx 流式语音识别"""

    def __init__(self, model_type=ASR_MODEL_TYPE, model_dir=ASR_DIR):
        import sherpa_onnx
        print(f"Loading ASR ({model_type})...", end=" ", flush=True)
        t0 = time.time()

        model_path = os.path.join(model_dir, model_type)

        if model_type == "ctc":
            self.rec = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
                tokens=os.path.join(model_path, "tokens.txt"),
                model=os.path.join(model_path, "model.int8.onnx"),
                num_threads=4,
                sample_rate=ASR_SAMPLE_RATE,
                feature_dim=80,
                decoding_method="greedy_search",
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=1.0,
                rule2_min_trailing_silence=1.5,
                rule3_min_utterance_length=3.0,
            )
        else:
            files = ASR_FILES[model_type]
            self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=os.path.join(model_path, "tokens.txt"),
                encoder=os.path.join(model_path, files["encoder"]),
                decoder=os.path.join(model_path, files["decoder"]),
                joiner=os.path.join(model_path, files["joiner"]),
                num_threads=4,
                sample_rate=ASR_SAMPLE_RATE,
                feature_dim=80,
                decoding_method="greedy_search",
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=1.0,
                rule2_min_trailing_silence=1.5,
                rule3_min_utterance_length=3.0,
            )

        self._stream = None
        self.cs = 1600
        print(f"done ({time.time() - t0:.1f}s)")

    def reset(self):
        self._stream = self.rec.create_stream()

    def accept_waveform(self, chunk: np.ndarray):
        if self._stream is None:
            self.reset()
        if len(chunk) < self.cs:
            tmp = np.zeros(self.cs, dtype=np.float32)
            tmp[: len(chunk)] = chunk
            chunk = tmp
        self._stream.accept_waveform(ASR_SAMPLE_RATE, chunk)
        while self.rec.is_ready(self._stream):
            self.rec.decode_stream(self._stream)

    def get_partial(self) -> str:
        if self._stream is None:
            return ""
        return ASR_TAG_RE.sub("", (self.rec.get_result(self._stream) or "")).strip()

    def is_endpoint(self) -> bool:
        if self._stream is None:
            return False
        return self.rec.is_endpoint(self._stream)

    def finalize(self) -> str:
        if self._stream is None:
            return ""
        text = ASR_TAG_RE.sub("", (self.rec.get_result(self._stream) or "")).strip()
        self._stream = None
        return text
