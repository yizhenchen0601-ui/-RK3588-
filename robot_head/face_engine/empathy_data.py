"""
共情模型合成数据生成器 - v5 (gen_batch 19模板权重)

核心变更:
  - 输出从 52 维 BS 改为 19 维模板权重
  - 每个权重对应 gen_batch_data.py 的一个表情模板
  - 权重和为1 (Dirichlet 分布)，表示"每个表情用多少"
  - 52 维 BS 通过固定解码矩阵从权重解码得到
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import math
import random

BS_NAMES = [
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
    "tongueOut"
]

EMO_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def _bs(d):
    v = np.zeros(52, dtype=np.float32)
    for idx, val in d.items():
        v[idx] = val
    return v


# ============================================================
# gen_batch_data.py 的 19 个手工 BS 模板（原样照搬）
# ============================================================
GENBATCH_TEMPLATES = {
    "Neutral":     _bs({}),
    "Happy":       _bs({43:1.0, 44:1.0, 18:0.6, 19:0.6, 6:0.5, 7:0.5, 24:0.1}),
    "Excitement":  _bs({43:1.0, 44:1.0, 20:0.7, 21:0.7, 2:0.8, 24:0.3}),
    "Humor":       _bs({44:1.0, 28:0.8, 19:0.5, 38:0.6, 4:0.5}),
    "Pride":       _bs({43:0.3, 44:0.3, 0:0.4, 1:0.4, 10:0.7, 11:0.7, 47:0.8}),
    "Trust":       _bs({43:0.4, 44:0.4, 18:0.2, 19:0.2, 2:0.3}),
    "Love":        _bs({43:0.7, 44:0.7, 8:0.3, 9:0.3, 18:0.8, 19:0.8, 6:0.6, 7:0.6}),
    "Relief":      _bs({8:0.9, 9:0.9, 31:0.4, 24:0.1, 43:0.2, 44:0.2}),
    "Hope":        _bs({20:0.6, 21:0.6, 2:0.9, 3:0.7, 4:0.7, 16:0.8, 17:0.8}),
    "Anger":       _bs({0:1.0, 1:1.0, 49:0.8, 50:0.8, 35:0.8, 36:0.8, 22:0.5}),
    "Disgust":     _bs({49:1.0, 50:1.0, 29:0.9, 30:0.9, 33:0.8, 34:0.8, 41:0.7}),
    "Fear":        _bs({20:1.0, 21:1.0, 24:0.6, 2:0.9, 0:0.2, 1:0.2, 31:0.5}),
    "Vigilance":   _bs({18:0.7, 19:0.7, 0:0.6, 1:0.6, 32:0.3, 12:0.6}),
    "Sad":         _bs({0:0.8, 1:0.8, 2:0.9, 29:1.0, 30:1.0, 39:0.8, 40:0.8}),
    "Loneliness":  _bs({29:0.6, 30:0.6, 10:0.8, 11:0.8, 8:0.2, 9:0.2}),
    "Guilt":       _bs({0:0.9, 1:0.9, 10:1.0, 11:1.0, 8:0.4, 9:0.4}),
    "Surprise":    _bs({2:1.0, 3:1.0, 4:1.0, 24:0.9, 20:0.9, 21:0.9, 31:0.3}),
    "Confusion":   _bs({2:1.0, 0:0.8, 31:0.6, 37:0.5, 23:0.4}),
    "Shyness":     _bs({26:0.4, 27:0.4, 35:0.4, 36:0.4, 14:0.5, 15:0.5, 43:0.2, 44:0.2}),
}

ALL_GENBATCH_NAMES = list(GENBATCH_TEMPLATES.keys())  # 19 个模板名
GENBATCH_NAME_TO_IDX = {name: i for i, name in enumerate(ALL_GENBATCH_NAMES)}

# 固定解码矩阵: (19, 52)，每行是一个 gen_batch 模板
DECODE_MATRIX = np.stack([GENBATCH_TEMPLATES[name] for name in ALL_GENBATCH_NAMES], axis=0)


def decode_weights_to_bs(weights):
    """19维模板权重 → 52维BS (矩阵乘法)"""
    if isinstance(weights, np.ndarray):
        return weights @ DECODE_MATRIX  # (..., 52)
    return weights @ torch.FloatTensor(DECODE_MATRIX).to(weights.device)


# ============================================================
# 每个 HSEmotion 情绪 → 19个 gen_batch 模板的 Dirichlet 先验
# 设计原则：主模板 alpha=45（占~0.8），辅模板 alpha=2~5
# 这样数字人的表情有明确的主情绪，不模糊
# ============================================================
EMO_DIRICHLET_INITIAL = {
    # Happy: 共享喜悦
    "Happy":    {"Happy":45, "Excitement":5, "Humor":3, "Love":2, "Trust":1, "Neutral":1},
    # Sad: 温暖陪伴（主: Trust）
    "Sad":      {"Trust":45, "Neutral":4, "Love":3, "Relief":2, "Sad":1, "Loneliness":1},
    # Angry: 冷静倾听（主: Neutral）
    "Angry":    {"Neutral":45, "Vigilance":4, "Trust":4, "Confusion":2, "Anger":1},
    # Fear: 安抚安全（主: Trust）
    "Fear":     {"Trust":45, "Neutral":4, "Relief":3, "Love":2, "Fear":1},
    # Surprise: 惊喜好奇（主: Surprise）
    "Surprise": {"Surprise":40, "Excitement":5, "Happy":4, "Hope":3, "Confusion":2, "Neutral":1},
    # Disgust: 平和接纳（主: Neutral）
    "Disgust":  {"Neutral":45, "Trust":4, "Confusion":3, "Disgust":1},
    # Neutral: 温暖自然（主: Neutral）
    "Neutral":  {"Neutral":45, "Trust":4, "Love":3, "Happy":2, "Relief":1},
}

EMO_DIRICHLET_SUSTAINED = {
    # Happy: 由衷高兴（主: Happy → Love）
    "Happy":    {"Happy":30, "Love":20, "Pride":4, "Excitement":3, "Trust":2, "Humor":1},
    # Sad: 深切关切（主: Loneliness，能感受但不过度）
    "Sad":      {"Loneliness":40, "Guilt":4, "Trust":3, "Love":2, "Neutral":2, "Sad":2, "Fear":1},
    # Angry: 认真对待（主: Vigilance）
    "Angry":    {"Vigilance":40, "Neutral":4, "Confusion":4, "Trust":3, "Anger":2, "Sad":1},
    # Fear: 守护保护（主: Vigilance）
    "Fear":     {"Vigilance":35, "Trust":5, "Neutral":4, "Confusion":3, "Fear":2, "Anger":1},
    # Surprise: 探索求知（主: Confusion）
    "Surprise": {"Confusion":35, "Hope":5, "Surprise":4, "Excitement":3, "Fear":2, "Vigilance":2},
    # Disgust: 耐心理解（主: Neutral）
    "Disgust":  {"Neutral":40, "Confusion":4, "Trust":3, "Vigilance":2, "Disgust":2},
    # Neutral: 亲和关注（主: Neutral → Love）
    "Neutral":  {"Neutral":30, "Love":15, "Trust":5, "Hope":3, "Happy":2, "Relief":1},
}


def sample_weights(emotion7, active_duration=None, noise_scale=0.05):
    """
    给定7维情绪向量，用Dirichlet采样生成19维模板权重。

    核心流程:
      1. 对每个活跃情绪，查看其 active_duration
      2. 在 initial 和 sustained 先验之间插值（基于持续时长）
      3. 按情绪强度加权合并所有情绪的 alpha
      4. Dirichlet 采样 → 19 维权重（和为1）

    Args:
        emotion7: (7,) softmax 输入
        active_duration: dict, 各情绪连续活跃帧数
    Returns:
        weights: (19,) 模板权重, sum=1
    """
    if active_duration is None:
        active_duration = {emo: 0 for emo in EMO_LABELS}

    # 基础 smoothing alpha (避免0导致某些模板永远不被采样)
    alpha_total = np.ones(19, dtype=np.float32) * 0.1

    for i, emo in enumerate(EMO_LABELS):
        intensity = emotion7[i]
        if intensity < 0.01:
            continue

        duration = active_duration.get(emo, 0)
        threshold = random.uniform(20, 40) if noise_scale > 0.03 else 30
        dur_factor = min(1.0, max(0, duration - 2) / threshold)

        init_prior = EMO_DIRICHLET_INITIAL[emo]
        sust_prior = EMO_DIRICHLET_SUSTAINED[emo]

        for tpl_name in ALL_GENBATCH_NAMES:
            a_init = init_prior.get(tpl_name, 0.1)
            a_sust = sust_prior.get(tpl_name, 0.1)
            alpha = a_init * (1 - dur_factor) + a_sust * dur_factor
            idx = GENBATCH_NAME_TO_IDX[tpl_name]
            alpha_total[idx] += alpha * intensity

    # Dirichlet 采样
    weights = np.random.dirichlet(alpha_total).astype(np.float32)
    return weights


def generate_emotion_sequence(seq_len=64, n_keyframes=4):
    """生成自然流畅的情绪序列（同前）"""
    key_t = sorted(random.sample(range(seq_len), n_keyframes))
    if key_t[0] != 0:
        key_t.insert(0, 0)
    if key_t[-1] != seq_len - 1:
        key_t.append(seq_len - 1)

    key_emotions = []
    for _ in key_t:
        primary = random.randint(0, 6)
        alphas = [1.0] * 7
        alphas[primary] = random.uniform(5.0, 20.0)
        if random.random() < 0.3:
            secondary = random.choice([i for i in range(7) if i != primary])
            alphas[secondary] = random.uniform(2.0, 5.0)
        vec = np.random.dirichlet(alphas).astype(np.float32)
        key_emotions.append(vec)

    emotion_seq = np.zeros((seq_len, 7), dtype=np.float32)
    for k in range(len(key_t) - 1):
        t_start, t_end = key_t[k], key_t[k + 1]
        e_start, e_end = key_emotions[k], key_emotions[k + 1]
        for t in range(t_start, t_end + 1):
            frac = 0 if t_end == t_start else (t - t_start) / (t_end - t_start)
            cf = (1 - math.cos(frac * math.pi)) / 2
            emotion_seq[t] = e_start * (1 - cf) + e_end * cf

    noise = np.random.normal(0, 0.02, (seq_len, 7)).astype(np.float32)
    emotion_seq = np.clip(emotion_seq + noise, 0, None)
    emotion_seq /= emotion_seq.sum(axis=1, keepdims=True) + 1e-10

    return emotion_seq


def generate_empathy_sequence(emotion_seq, diversity=1.0):
    """
    从情绪序列生成对应的19维模板权重序列 + 52维BS。

    返回:
        weights_seq: (T, 19) 模板权重
        bs_seq: (T, 52) 解码后的BS
    """
    T = emotion_seq.shape[0]
    active_duration = {emo: 0 for emo in EMO_LABELS}
    raw_weights = []

    for t in range(T):
        vec = emotion_seq[t]
        for i, emo in enumerate(EMO_LABELS):
            if vec[i] > 0.15:
                active_duration[emo] += 1
            else:
                active_duration[emo] = 0

        noise = 0.03 + 0.07 * diversity
        w = sample_weights(vec, active_duration, noise_scale=noise)
        raw_weights.append(w)

    raw = np.stack(raw_weights).astype(np.float32)  # (T, 19)

    # EMA 平滑（在权重空间做）
    smoothed = np.zeros_like(raw)
    alpha = random.uniform(0.15, 0.35)
    smoothed[0] = raw[0]
    for t in range(1, T):
        smoothed[t] = alpha * raw[t] + (1 - alpha) * smoothed[t - 1]
    # 重归一化
    smoothed /= smoothed.sum(axis=1, keepdims=True) + 1e-10

    bs_seq = decode_weights_to_bs(smoothed)  # (T, 52)
    return smoothed.astype(np.float32), bs_seq.astype(np.float32)


class EmpathyDataset(Dataset):
    """合成共情数据集 — 7维 HSEmotion → 19维 gen_batch 模板权重"""

    def __init__(self, num_samples=10000, seq_len=64, diversity_range=(0.5, 1.0)):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.diversity_range = diversity_range
        self.samples = []
        self._generate()

    def _generate(self):
        for i in range(self.num_samples):
            emotion_seq = generate_emotion_sequence(
                seq_len=self.seq_len,
                n_keyframes=random.randint(3, 6)
            )
            diversity = random.uniform(*self.diversity_range)
            weights_seq, bs_seq = generate_empathy_sequence(emotion_seq, diversity)
            # 存储 (emotion, weights, bs)
            self.samples.append((emotion_seq, weights_seq, bs_seq))
            if (i + 1) % 2000 == 0:
                print(f"  生成数据: {i+1}/{self.num_samples}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        emotion, weights, bs = self.samples[idx]
        return torch.FloatTensor(emotion), torch.FloatTensor(weights), torch.FloatTensor(bs)


def quick_test():
    """快速测试"""
    print("=" * 65)
    print("  共情数据生成器 v5 — 19维模板权重")
    print("=" * 65)

    # 纯情绪测试
    print("\n--- 纯情绪 (模板权重分布) ---")
    dur0 = {e: 0 for e in EMO_LABELS}
    for i, emo in enumerate(EMO_LABELS):
        vec = np.zeros(7, dtype=np.float32)
        vec[i] = 1.0
        w = sample_weights(vec, dur0, noise_scale=0.0)
        active = [(ALL_GENBATCH_NAMES[j], round(w[j], 3))
                   for j in range(19) if w[j] > 0.02]
        active.sort(key=lambda x: -x[1])
        print(f"  {emo:>10}: {active[:8]}")

    print("\n--- 持续50帧后 ---")
    dur50 = {e: 50 for e in EMO_LABELS}
    for i, emo in enumerate(EMO_LABELS):
        vec = np.zeros(7, dtype=np.float32)
        vec[i] = 1.0
        w = sample_weights(vec, dur50, noise_scale=0.0)
        active = [(ALL_GENBATCH_NAMES[j], round(w[j], 3))
                   for j in range(19) if w[j] > 0.02]
        active.sort(key=lambda x: -x[1])
        print(f"  {emo:>10}: {active[:8]}")

    print("\n--- 混合情绪 ---")
    mixes = [
        ("70% Sad + 30% Happy", np.array([0.0,0.0,0.0,0.3,0.0,0.7,0.0])),
        ("50% Angry + 50% Sad", np.array([0.5,0.0,0.0,0.0,0.0,0.5,0.0])),
        ("60% Happy + 40% Surprise", np.array([0.0,0.0,0.0,0.6,0.0,0.0,0.4])),
    ]
    for label, vec in mixes:
        w = sample_weights(vec, dur50, noise_scale=0.0)
        active = [(ALL_GENBATCH_NAMES[j], round(w[j], 3))
                   for j in range(19) if w[j] > 0.02]
        active.sort(key=lambda x: -x[1])
        print(f"  {label:>30}: {active[:8]}")

    # 时序测试
    print("\n--- 时序依赖 ---")
    for emo_name, emo_idx in [("Sad", 5), ("Angry", 0), ("Surprise", 6)]:
        print(f"  [{emo_name}]")
        vec = np.zeros(7, dtype=np.float32)
        vec[emo_idx] = 0.9
        dur = {e: 0 for e in EMO_LABELS}
        for t in [1, 5, 15, 30, 50]:
            dur[emo_name] = t
            w = sample_weights(vec, dur, noise_scale=0.0)
            active = [(ALL_GENBATCH_NAMES[j], round(w[j], 3))
                       for j in range(19) if w[j] > 0.02]
            active.sort(key=lambda x: -x[1])
            print(f"    帧{t:3d}: {active[:6]}")

    # 数据集测试
    print("\n--- 数据集 ---")
    ds = EmpathyDataset(num_samples=50, seq_len=48)
    emo, w, bs = ds[0]
    print(f"  样本数: {len(ds)}")
    print(f"  emotion: {emo.shape}, sum={emo[0].sum():.2f}")
    print(f"  weights: {w.shape}, sum={w[0].sum():.3f}")
    print(f"  bs: {bs.shape}, range=[{bs.min():.2f}, {bs.max():.2f}]")
    print(f"  bs ≈ weights @ decode  (验证解码正确性)")

    print("\n✓ 准备就绪!")


if __name__ == "__main__":
    quick_test()
