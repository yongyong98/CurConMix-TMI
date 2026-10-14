import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 1D Sine Positional Encoding (for temporal tokens) ----
class SinePositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, C]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # [T, C]

    def forward(self, x):  # x: [B, T, C]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0).to(x.dtype)  # [1,T,C]

# ---- Temporal Transformer Encoder (4 heads, 3 layers) ----
class TemporalTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, num_layers: int = 3, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pos = SinePositionalEncoding1D(d_model)

    def forward(self, x):  # x: [B, T, C]
        x = self.pos(x)
        return self.encoder(x)  # [B, T, C]

# ---- Multi-resolution temporal path (k in {4,6,9}) ----
class TemporalPathway(nn.Module):
    """
    시간축 평균풀링(kernel=stride=k)으로 길이를 줄여 Transformer 처리 후,
    원 길이로 반복 복원(repeat_interleave). 마지막에 분류기 통과.
    """
    def __init__(self, d_model: int, num_classes: int, k: int, nhead: int = 4, num_layers: int = 3):
        super().__init__()
        self.k = k
        self.temporal = TemporalTransformer(d_model, nhead=nhead, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):  # x: [B, T, C]
        B, T, C = x.shape
        # pad to multiple of k
        pad = (self.k - (T % self.k)) % self.k
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))  # pad time dimension at end

        # pool: [B, T', C], T' = T//k
        T_pooled = x.shape[1] // self.k
        x = x.view(B, T_pooled, self.k, C).mean(dim=2)

        # temporal encoder
        h = self.temporal(x)  # [B, T_pooled, C]

        # upsample by repeating along time back to original length
        h_up = h.repeat_interleave(self.k, dim=1)  # [B, T_pooled*k, C]
        if pad > 0:
            h_up = h_up[:, :T, :]  # trim to original T

        logits = self.classifier(h_up)  # [B, T, num_classes]
        return logits

# ---- Wrapper: Multi-resolution + spatial averaging ----
class MultiResTemporalModel(nn.Module):
    """
    입력: 영상 프레임 배치 (또는 사전 추출 임베딩)
    - spatial_encoder: [B,T, ...] -> per-frame embedding [B,T,C] 제공하는 모듈
      (예: Swin-Base/Large의 글로벌 풀링 벡터; 여기서는 외부에서 제공한다고 가정)
    - spatial_head: [C] -> num_classes (프레임 단 분류기)
    최종 로짓 = (spatial_logits + mean(temporal_logits_list)) / 2
    """
    def __init__(self, embed_dim: int, num_classes: int, ks=(4,6,9), use_spatial_avg=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.ks = ks
        self.use_spatial_avg = use_spatial_avg

        # 공간 인코더(백본)는 외부에서 주입 받는 설계로 유지
        # -> forward에서 embeddings 인자로 받습니다.
        # 프레임 단 분류기(Spatial head)
        self.spatial_head = nn.Linear(embed_dim, num_classes)

        # 세 개의 temporal pathway
        self.temporal_paths = nn.ModuleList(
            [TemporalPathway(embed_dim, num_classes, k=k, nhead=4, num_layers=3) for k in ks]
        )

    def forward(self, frame_embeddings):
        """
        frame_embeddings: [B, T, C]  — Swin-Base/Large 등에서 추출된 per-frame 임베딩
        반환: logits_final [B, T, num_classes] (프레임 단 예측)
        """
        B, T, C = frame_embeddings.shape
        assert C == self.embed_dim, f"Expected embedding dim {self.embed_dim}, got {C}"

        # (i) Spatial per-frame logits
        spatial_logits = self.spatial_head(frame_embeddings)  # [B,T,num_classes]

        # (ii) Temporal decoders at k ∈ {4,6,9}
        temporal_logits_list = [tp(frame_embeddings) for tp in self.temporal_paths]  # list of [B,T,num_classes]
        temporal_mean = torch.stack(temporal_logits_list, dim=0).mean(dim=0)  # [B,T,num_classes]

        # Final averaging
        if self.use_spatial_avg:
            logits = (spatial_logits + temporal_mean) / 2.0
        else:
            logits = temporal_mean

        return logits, temporal_mean


# ----------------- Example usage -----------------
if __name__ == "__main__":
    embed_dim = 1024  # or 1536
    num_classes = 7   # 예: phase recognition 7 classes
    B, T = 2, 64

    model = MultiResTemporalModel(embed_dim=embed_dim, num_classes=num_classes, ks=(4,6,9))
    dummy_embeddings = torch.randn(B, T, embed_dim)  # 실제로는 Swin에서 뽑은 per-frame embedding
    logits, parts = model(dummy_embeddings)
    print(logits.shape)  # [2, 64, 7]