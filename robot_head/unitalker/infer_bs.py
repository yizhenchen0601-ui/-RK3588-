"""
UniTalker 推理：绕开 HuggingFace 在线下载，直接从 checkpoint 加载 WavLM 权重
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import librosa
from transformers import Wav2Vec2FeatureExtractor, WavLMModel as HfWavLMModel
from models.unitalker import UniTalker
from utils.utils import get_audio_encoder_dim

# ========== D6 51-dim inhouse ARKit → render.py 52 的映射 ==========
# D6 使用 inhouse_blendshape_weight (51-dim)，顺序由 arkit_flame.npz 定义
# render.py 使用不同的 52-dim 顺序（含 tongueOut）
REMAP_INHOUSE_51_TO_RENDER_52 = [
    14, 15, 16, 17, 18,   #  0-4: browDown L/R, browInnerUp, browOuterUp L/R
    48, 49, 50,            #  5-7: cheekPuff, cheekSquint L/R
    0, 1,                  #  8-9: eyeBlink L/R
    4, 5,                  # 10-11: eyeLookDown L/R
    6, 7,                  # 12-13: eyeLookIn L/R
    10, 11,                # 14-15: eyeLookOut L/R
    12, 13,                # 16-17: eyeLookUp L/R
    2, 3,                  # 18-19: eyeSquint L/R
    8, 9,                  # 20-21: eyeWide L/R
    23, 21, 19, 22,        # 22-25: jawForward, jawLeft, jawOpen, jawRight
    20,                    # 26: mouthClose
    32, 33,                # 27-28: mouthDimple L/R
    36, 37,                # 29-30: mouthFrown L/R
    41,                    # 31: mouthFunnel
    42,                    # 32: mouthLeft
    26, 27,                # 33-34: mouthLowerDown L/R
    38, 39,                # 35-36: mouthPress L/R
    40,                    # 37: mouthPucker
    43,                    # 38: mouthRight
    29, 28,                # 39-40: mouthRollLower, mouthRollUpper
    44, 45,                # 41-42: mouthShrugLower, mouthShrugUpper
    30, 31,                # 43-44: mouthSmile L/R
    34, 35,                # 45-46: mouthStretch L/R
    24, 25,                # 47-48: mouthUpperUp L/R
    46, 47,                # 49-50: noseSneer L/R
    # 51: tongueOut → 不在 51-dim 中，设为 0
]

# Use defaults matching microsoft/wavlm-base-plus (inferred from checkpoint keys)
WAVLM_CONFIG = {
    "vocab_size": 32,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
    "conv_dim": [512, 512, 512, 512, 512, 512, 512],
    "conv_stride": [5, 2, 2, 2, 2, 2, 2],
    "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
    "conv_bias": False,              # matches checkpoint (no conv.bias keys)
    "feat_extract_norm": "group",    # matches checkpoint (layer_norm only on layer 0)
    "feat_extract_activation": "gelu",
    "do_stable_layer_norm": False,
    "num_conv_pos_embeddings": 128,
    "num_conv_pos_embedding_groups": 16,
    "apply_spec_augment": True,
    "mask_time_prob": 0.05,
    "mask_time_length": 10,
    "mask_time_min_masks": 2,
    "num_codevectors_per_group": 320,
    "num_codevector_groups": 2,
    "contrastive_logits_temperature": 0.1,
    "codevector_dim": 256,
    "proj_codevector_dim": 256,
    "num_negatives": 100,
    "diversity_loss_weight": 0.1,
    "model_type": "wavlm",
}


def split_long_audio(audio, processor):
    a, b = 25, 5
    sr = 16000
    total_length = len(audio) / sr
    reps = max(0, int(np.ceil((total_length - a) / (a - b)))) + 1
    in_list = []
    start, end = 0, int(a * sr)
    step = int((a - b) * sr)
    for _ in range(reps):
        seg = audio[start:end]
        seg = np.squeeze(processor(seg, sampling_rate=sr).input_values)
        in_list.append(seg)
        start += step
        end += step
    return in_list


def merge_out_list(out_list, fps):
    if len(out_list) == 1:
        return out_list[0]
    a, b = 25, 5
    lw = np.linspace(1, 0, b * fps)[:, np.newaxis]
    rw = 1 - lw
    a = a * fps
    b = b * fps
    offset = a - b
    out_len = len(out_list[-1]) + offset * (len(out_list) - 1)
    merged = np.empty((out_len, out_list[-1].shape[-1]), dtype=out_list[-1].dtype)
    merged[:a] = out_list[0]
    for piece in out_list[1:]:
        merged[a - b:a] = lw * merged[a - b:a] + rw * piece[:b]
        merged[a:a + offset] = piece[b:]
        a += offset
    return merged


class Args:
    dataset = ["D6"]
    demo_dataset = ["D6"]
    data_root = "./"
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
    identity_num = None
    audio_encoder_feature_dim = 768


def inject_wavlm_from_ckpt(model, ckpt, device):
    """从 checkpoint 中提取 WavLM 权重注入模型"""
    wavlm_state = {}
    for k, v in ckpt.items():
        if k.startswith("audio_encoder."):
            # 去掉 "audio_encoder." 前缀 → WavLM 自己的 key
            wavlm_key = k[len("audio_encoder."):]
            wavlm_state[wavlm_key] = v
    model.audio_encoder.load_state_dict(wavlm_state, strict=False)
    model.audio_encoder.to(device)
    print(f"  Injected WavLM weights from checkpoint ({len(wavlm_state)} keys)")


def infer_unitalker(wav_path, checkpoint_path, device="cuda:0"):
    args = Args()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    args.identity_num = len(ckpt["decoder.learnable_style_emb.weight"])

    # 创建 UniTalker，patch 掉 from_pretrained 改用本地初始化
    from transformers import WavLMConfig
    from models.wavlm import WavLMModel as CustomWavLMModel
    wavlm_config = WavLMConfig(**WAVLM_CONFIG)

    # 预创建 WavLM 实例，避免 UniTalker.__init__ 时联网下载
    wavlm_instance = CustomWavLMModel(wavlm_config)

    def offline_from_pretrained(*args, **kwargs):
        return wavlm_instance

    CustomWavLMModel.from_pretrained = offline_from_pretrained

    model = UniTalker(args)

    # 从 checkpoint 注入 WavLM + 其余权重
    inject_wavlm_from_ckpt(model, ckpt, device)
    model.load_state_dict(ckpt, strict=False)
    model.eval().to(device)
    print(f"Model loaded. identity_num={args.identity_num}")

    # 加载音频 processor（用本地缓存的 Wav2Vec2 config，不需要联网）
    try:
        # 尝试用本地缓存
        processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
            local_files_only=True,
        )
    except:
        # 手动创建 Wav2Vec2FeatureExtractor
        processor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False,
        )

    audio_data, _ = librosa.load(wav_path, sr=16000)
    audio_data = np.squeeze(processor(audio_data, sampling_rate=16000).input_values)

    audio_splits = split_long_audio(audio_data, processor)

    # D6 head 推理
    annot_type = "inhouse_blendshape_weight"
    fps = 30
    condition_id = torch.tensor([915 + 4], device=device)  # D6 offset=915, local_idx=4
    template = torch.zeros((1, 51), device=device)

    out_list = []
    with torch.no_grad():
        for seg in audio_splits:
            audio_tensor = torch.FloatTensor(seg[None]).to(device)
            out_motion, _, _ = model(
                audio_tensor, template,
                face_motion=None, style_idx=condition_id,
                annot_type=annot_type, fps=fps,
            )
            out_list.append(out_motion.cpu().numpy().squeeze(0))

    bs_51 = merge_out_list(out_list, fps)
    print(f"Raw 51-dim: {bs_51.shape}, range=[{bs_51.min():.4f}, {bs_51.max():.4f}]")
    return bs_51


def bs51_to_render_order(bs_51):
    bs_52 = np.zeros((bs_51.shape[0], 52), dtype=np.float32)
    for render_i, inhouse_i in enumerate(REMAP_INHOUSE_51_TO_RENDER_52):
        if inhouse_i >= 0:
            bs_52[:, render_i] = bs_51[:, inhouse_i]
    # tongueOut (idx 51) stays 0
    bs_52 = np.clip(bs_52, 0.0, 1.0)
    return bs_52


def main():
    parser = argparse.ArgumentParser(description="UniTalker → 52-dim BS (render.py order)")
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument("--out", type=str, default="./result/unitalker_output.npy")
    parser.add_argument("--ckpt", type=str,
                        default="./pretrained_models/UniTalker-B-D0-D7.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    bs_51 = infer_unitalker(args.wav, args.ckpt, args.device)
    bs_52 = bs51_to_render_order(bs_51)

    np.save(args.out, bs_52)
    print(f"\nSaved: {args.out}")
    print(f"Shape: {bs_52.shape}, range=[{bs_52.min():.4f}, {bs_52.max():.4f}]")
    print(f"Frames: {bs_52.shape[0]} @ 30fps = {bs_52.shape[0]/30:.1f}s")


if __name__ == "__main__":
    main()
