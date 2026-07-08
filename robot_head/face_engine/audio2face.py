"""
Audio2Face 管线：TTS音频 → UniTalk → MLP → 舵机角度
从 tts_unitalk_servo.py 提取的核心逻辑
"""
import os
import sys
import time
import json
import threading
import queue
import subprocess
import wave
import numpy as np
import torch
import torch.nn as nn

# ─── 路径 ───
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNITALK_DIR = os.path.join(BASE, "unitalker")
MODELS_DIR = os.path.join(BASE, "models")

UNITALK_CKPT = os.path.join(MODELS_DIR, "UniTalker-B-D0-D7.pt")
UPPER_MODEL_PATH = os.path.join(MODELS_DIR, "upper_face_bs2angle.pth")
LOWER_MODEL_PATH = os.path.join(MODELS_DIR, "lower_face_bs2angle.pth")
FORWARD_MODEL_PATH = os.path.join(MODELS_DIR, "angle2bs_full.pth")
TEMPLATE_PATH = os.path.join(MODELS_DIR, "inhouse_template.npy")

FPS = 30
VOLUME_GAIN = 3.0

# ─── MLP 模型定义 ───
class UpperFaceBS2Angle(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class LowerFaceBS2Angle(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


def init_mlp(device="cpu", lower_only=False):
    """加载 BS→角度 MLP 模型"""
    fwd_ckpt = torch.load(FORWARD_MODEL_PATH, map_location="cpu")
    fwd_bs_keys = fwd_ckpt["bs_keys"]

    upper_model = None
    upper_idx, upper_motor_ids = [], []
    if not lower_only:
        upper_ckpt = torch.load(UPPER_MODEL_PATH, map_location="cpu")
        upper_model = UpperFaceBS2Angle(
            input_dim=len(upper_ckpt["upper_bs_keys"]),
            output_dim=len(upper_ckpt["upper_motor_ids"]),
        ).to(device)
        upper_model.load_state_dict(upper_ckpt["model_state_dict"])
        upper_model.eval()
        upper_motor_ids = upper_ckpt["upper_motor_ids"]
        motor_ranges = upper_ckpt["motor_ranges"]
        used_motors = upper_ckpt["used_motors_full"]
        unitalk_names = fwd_bs_keys[1:] + ["tongueOut"]
        upper_idx = [unitalk_names.index(k) for k in upper_ckpt["upper_bs_keys"]]
    else:
        motor_ranges = None

    lower_ckpt = torch.load(LOWER_MODEL_PATH, map_location="cpu")
    lower_model = LowerFaceBS2Angle(
        input_dim=len(lower_ckpt["lower_bs_idx"]),
        output_dim=len(lower_ckpt["lower_motor_ids"]),
    ).to(device)
    lower_model.load_state_dict(lower_ckpt["model_state_dict"])
    lower_model.eval()

    if lower_only:
        motor_ranges = lower_ckpt["motor_ranges"]
        used_motors = lower_ckpt["lower_motor_ids"]
    lower_bs_idx = lower_ckpt["lower_bs_idx"]
    lower_motor_ids = lower_ckpt["lower_motor_ids"]

    unitalk_names = fwd_bs_keys[1:] + ["tongueOut"]
    lower_bs_keys = [fwd_bs_keys[i] for i in lower_bs_idx]
    lower_idx = [unitalk_names.index(k) for k in lower_bs_keys]

    default_angles = {}
    for mid in used_motors:
        minv, maxv = motor_ranges[mid]
        default_angles[mid] = (minv + maxv) / 2.0

    mode_str = "下脸" if lower_only else "全脸"
    print(f"  MLP: {mode_str} ({len(used_motors)}舵机)")
    return (upper_model, lower_model, upper_idx, lower_idx,
            motor_ranges, upper_motor_ids, lower_motor_ids,
            used_motors, default_angles)


def compute_angles(upper_model, lower_model, bs_52,
                   upper_idx, lower_idx,
                   motor_ranges, upper_motor_ids, lower_motor_ids,
                   default_angles, device="cpu", lower_only=False):
    """52维BS序列 → 每帧的舵机角度字典列表"""
    N = bs_52.shape[0]
    all_angles = []
    for f in range(N):
        frame = bs_52[f]
        angles = default_angles.copy()
        if not lower_only and upper_model is not None:
            ui = torch.tensor(frame[upper_idx], dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pn = upper_model(ui).cpu().numpy()[0]
            for i, mid in enumerate(upper_motor_ids):
                mv = motor_ranges[mid]
                angles[mid] = round(float(pn[i] * (mv[1] - mv[0]) + mv[0]), 2)
        li = torch.tensor(frame[lower_idx], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pn = lower_model(li).cpu().numpy()[0]
        for i, mid in enumerate(lower_motor_ids):
            mv = motor_ranges[mid]
            angles[mid] = round(float(pn[i] * (mv[1] - mv[0]) + mv[0]), 2)
        all_angles.append(angles)
    return all_angles


def init_unitalk(device="cpu"):
    """加载 UniTalker 模型（音频→52BS）"""
    sys.path.insert(0, UNITALK_DIR)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import Wav2Vec2FeatureExtractor, WavLMConfig
    from models.wavlm import WavLMModel as CustomWavLMModel
    from models.unitalker import UniTalker
    from infer_bs import WAVLM_CONFIG, inject_wavlm_from_ckpt

    t0 = time.perf_counter()
    ckpt = torch.load(UNITALK_CKPT, map_location="cpu", weights_only=True)

    class Args:
        dataset = ["D6"]
        data_root = UNITALK_DIR
        duplicate_list = "1"
        use_pca = False
        pca_dim = 512
        audio_encoder_repo = "microsoft/wavlm-base-plus"
        freeze_wav2vec = False
        interpolate_pos = 1
        decoder_dimension = 256
        decoder_type = "conv"
        period = 30
        headlayer = 1
        identity_num = len(ckpt["decoder.learnable_style_emb.weight"])
        audio_encoder_feature_dim = 768

    args = Args()
    wavlm_config = WavLMConfig(**WAVLM_CONFIG)
    CustomWavLMModel.from_pretrained = lambda *a, **kw: CustomWavLMModel(wavlm_config)
    model = UniTalker(args)
    inject_wavlm_from_ckpt(model, ckpt, device)
    model.load_state_dict(ckpt, strict=False)
    model.eval().to(device)

    processor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=16000,
        padding_value=0.0, do_normalize=True, return_attention_mask=False,
    )
    condition_id = torch.tensor([915 + 4], device=device)
    template = torch.zeros((1, 51), device=device)

    print(f"  UniTalk: {time.perf_counter()-t0:.1f}s")
    return model, processor, condition_id, template


def unitalk_infer(model, processor, condition_id, template, wav_path, device="cpu"):
    """WAV音频 → 52维BS序列"""
    sys.path.insert(0, UNITALK_DIR)
    from infer_bs import split_long_audio, bs51_to_render_order

    t0 = time.perf_counter()
    import wave as _wave
    with _wave.open(wav_path, "rb") as wf:
        sr_wav = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if sr_wav != 16000:
        orig = np.arange(len(audio)) / sr_wav
        target = np.arange(0, len(audio) / sr_wav, 1 / 16000)
        audio = np.interp(target, orig, audio).astype(np.float32)

    splits = split_long_audio(audio, processor)
    out_list = []
    with torch.no_grad():
        for seg in splits:
            t = torch.FloatTensor(seg[None]).to(device)
            motion, _, _ = model(t, template, None, condition_id,
                                 "inhouse_blendshape_weight", FPS)
            out_list.append(motion.cpu().numpy().squeeze(0))
    if len(out_list) == 1:
        bs_51 = out_list[0]
    else:
        from infer_bs import merge_out_list
        bs_51 = merge_out_list(out_list, FPS)
    bs_52 = bs51_to_render_order(bs_51)

    t_model = time.perf_counter() - t0
    print(f"  UniTalk: {bs_52.shape[0]}帧 ({bs_52.shape[0]/FPS:.1f}s) {t_model:.2f}s")
    return bs_52


# ─── 完整 Audio2Face 管线 ───
class Audio2Face:
    """TTS → UniTalk → MLP → 舵机 一站式"""

    def __init__(self, device="cpu", lower_only=False):
        self.device = device
        self.lower_only = lower_only
        self.initialized = False

    def initialize(self):
        print("  Audio2Face 初始化...")
        t0 = time.perf_counter()
        (self.upper_model, self.lower_model, self.upper_idx, self.lower_idx,
         self.motor_ranges, self.upper_motor_ids, self.lower_motor_ids,
         self.used_motors, self.default_angles) = init_mlp(self.device, self.lower_only)
        self.unitalk_model, self.processor, self.cond_id, self.template =             init_unitalk(self.device)
        self.initialized = True
        print(f"  Audio2Face 就绪 ({time.perf_counter()-t0:.1f}s)")

    def process(self, wav_path):
        """WAV → BS序列 → 舵机角度"""
        if not self.initialized:
            raise RuntimeError("Audio2Face not initialized")
        bs_52 = unitalk_infer(
            self.unitalk_model, self.processor, self.cond_id,
            self.template, wav_path, self.device,
        )
        return bs_52  # numpy array (N, 52)
