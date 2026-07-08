"""
共情模型 v5 — gen_batch 模板权重预测

架构:
  - 输入 7 维 softmax → TransformerEncoder → 帧级特征
  - Cross-Attention 风格查询 → 全局风格向量
  - MLP(128→64→32→19) → Softmax → 19 维模板权重
  - 固定解码矩阵 (19×52) 权重 → 52 维 BS

核心 design:
  - 输出是 19 个 gen_batch 模板的权重（和为1），可解释性强
  - "这个情绪要用 60% Happy + 30% Excitement + 10% Love"
  - 解码矩阵来自 gen_batch_data.py，不参与训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


def _build_template_matrix():
    """从 empathy_data 构建固定的 19×52 解码矩阵"""
    from empathy_data import GENBATCH_TEMPLATES, ALL_GENBATCH_NAMES
    rows = []
    for name in ALL_GENBATCH_NAMES:
        rows.append(torch.FloatTensor(GENBATCH_TEMPLATES[name]))
    return torch.stack(rows)  # (19, 52)


class EmpathyModel(nn.Module):
    """
    情绪条件共情模型 — 预测 gen_batch 模板权重

    输入: emotion_seq (B, T, 7)
    输出: weights (B, T, 19) + bs (B, T, 52)
    """
    def __init__(self, input_dim=7, output_dim=19, d_model=64,
                 nhead=4, num_layers=2, max_seq_len=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 情绪编码器 (保持不变)
        self.emo_embed = nn.Linear(input_dim, d_model)
        self.emo_pos = PositionalEncoding(d_model, max_seq_len)
        emo_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=False, norm_first=True
        )
        self.emotion_encoder = nn.TransformerEncoder(emo_layer, num_layers)

        # 风格查询 (保持不变)
        self.style_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.1)
        self.style_cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )
        self.style_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # 解码器: 输出 19 维模板权重 (原来是 52 维 BS)
        self.decoder_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 19),
        )

        # 固定解码矩阵: (19, 52)，不参与训练
        template_matrix = _build_template_matrix()
        self.register_buffer('template_matrix', template_matrix)

    def encode_emotion(self, emotion_seq):
        x = self.emo_embed(emotion_seq) * math.sqrt(self.d_model)
        x = x.permute(1, 0, 2)
        x = self.emo_pos(x)
        return self.emotion_encoder(x)

    def decode_weights_to_bs(self, weights):
        """19维权重 → 52维BS (固定矩阵乘法)"""
        return weights @ self.template_matrix  # (..., 52)

    def forward(self, emotion_seq):
        """
        前向传播

        Returns:
            weights: (B, T, 19) softmax 模板权重
            bs: (B, T, 52) 解码后的 BS
        """
        B, T, _ = emotion_seq.shape
        emo_feat = self.encode_emotion(emotion_seq)  # (T, B, D)

        # Cross-attention 提取全局风格
        query = self.style_query.expand(1, B, -1)
        style_feat, _ = self.style_cross_attn(query, emo_feat, emo_feat)
        style_feat = self.style_mlp(style_feat.squeeze(0))  # (B, D)

        # 风格广播到每帧
        style_expanded = style_feat.unsqueeze(0).expand(T, B, -1)
        cat_feat = torch.cat([emo_feat, style_expanded], dim=-1)
        cat_feat = cat_feat.permute(1, 0, 2)  # (B, T, 2D)

        logits = self.decoder_mlp(cat_feat)  # (B, T, 19)
        weights = F.softmax(logits, dim=-1)
        bs = self.decode_weights_to_bs(weights)

        return weights, bs

    @torch.no_grad()
    def generate(self, emotion_seq):
        """推理接口 (向后兼容，只返回 BS)"""
        self.eval()
        _, bs = self.forward(emotion_seq)
        return bs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=== 共情模型 v5 (模板权重版) ===\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmpathyModel(d_model=64, nhead=4, num_layers=2).to(device)
    print(f"参数量: {count_parameters(model):,}")
    print(f"template_matrix: {model.template_matrix.shape} (固定, 不参与训练)")

    B, T = 4, 32
    emo = torch.from_numpy(np.random.dirichlet(np.ones(7), (B, T))).float().to(device)
    weights, bs = model(emo)
    print(f"\n输入: {emo.shape}")
    print(f"权重: {weights.shape}, 行和={weights[0,0].sum():.3f}")
    print(f"BS:   {bs.shape}, range=[{bs.min():.3f}, {bs.max():.3f}]")

    # 测试 generate (向后兼容)
    bs_gen = model.generate(emo)
    print(f"generate BS: {bs_gen.shape}, 与 forward BS 一致: {torch.allclose(bs, bs_gen)}")

    print("\n7 种情绪测试:")
    from empathy_data import EMO_LABELS, BS_NAMES, ALL_GENBATCH_NAMES
    for i, emo_name in enumerate(EMO_LABELS):
        vec = torch.zeros(1, T, 7).to(device)
        vec[0, :, i] = 1.0
        w, bs = model.generate_with_weights(vec) if hasattr(model, 'generate_with_weights') else (None, model.generate(vec))
        # 取最后一帧的权重
        w_last = model.forward(vec)[0][0, -1].cpu().numpy()
        top5 = np.argsort(w_last)[-5:][::-1]
        s = ' | '.join(f'{ALL_GENBATCH_NAMES[j]}={w_last[j]:.3f}' for j in top5)
        print(f"  {emo_name:>10}: {s}")

    print("\n测试通过!")
