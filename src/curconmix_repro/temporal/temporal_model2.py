import math
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------- 1D Sine Positional Encoding -------------------------
class SinePositionalEncoding1D(nn.Module):
    """Odd/even d_model 모두 안전, [1,T,C]로 등록해 broadcast/디바이스 안전."""
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        even_idx = torch.arange(0, d_model, 2, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * torch.exp(even_idx * (-math.log(10000.0) / d_model)))

        odd_idx = torch.arange(1, d_model, 2, dtype=torch.float32)
        if odd_idx.numel() > 0:
            pe[:, 1::2] = torch.cos(pos * torch.exp((odd_idx - 1) * (-math.log(10000.0) / d_model)))

        self.register_buffer("pe", pe.unsqueeze(0))  # [1,T,C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :].to(dtype=x.dtype)


# ----------------------------- Temporal Transformer ----------------------------
class TemporalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pos = SinePositionalEncoding1D(d_model)

    def forward(
        self,
        x: torch.Tensor,                           # [B,T',C]
        key_padding_mask: Optional[torch.Tensor]   # [B,T'] (True = pad)
    ) -> torch.Tensor:
        x = self.pos(x)
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


# ------------------------------ Masked window mean -----------------------------
def windowed_masked_mean(
    x: torch.Tensor,        # [B, T_pad, C]
    mask: torch.Tensor,     # [B, T_pad] (True=valid)
    k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T_pad, C = x.shape
    assert (T_pad % k) == 0
    T_pooled = T_pad // k

    x_ = x.view(B, T_pooled, k, C)                # [B,T',k,C]
    m_ = mask.view(B, T_pooled, k).unsqueeze(-1)  # [B,T',k,1] bool

    m_float = m_.float()
    denom = m_float.sum(dim=2).clamp_min(1.0)     # [B,T',1]
    num = (x_ * m_float).sum(dim=2)               # [B,T',C]
    pooled = num / denom                          # [B,T',C]

    pooled_mask = (denom.squeeze(-1) > 0)         # [B,T'] True if any valid
    return pooled, pooled_mask


def masked_temporal_mean(
    x: torch.Tensor,                               # [B,T,C]
    valid_len: Optional[torch.Tensor] = None,      # [B]
) -> torch.Tensor:
    if valid_len is None:
        return x.mean(dim=1)

    B, T, _ = x.shape
    ar = torch.arange(T, device=x.device).unsqueeze(0)
    mask = (ar < valid_len.unsqueeze(1)).unsqueeze(-1)       # [B,T,1]
    denom = mask.sum(dim=1).clamp_min(1)                     # [B,1]
    return (x * mask.to(dtype=x.dtype)).sum(dim=1) / denom.to(dtype=x.dtype)


# ------------------------------ Temporal Pathway --------------------------------
class TemporalPathway(nn.Module):
    """
    k단위 평균풀링(마스크드) → Transformer → (다운샘플 해상도에서) Linear 분류 → logits를 최근접 업샘플.
    """
    def __init__(
        self,
        d_model: int,
        num_classes: int,
        k: int,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.k = k
        self.temporal = TemporalTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, valid_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        # per-frame 유효 마스크
        if valid_len is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
        else:
            ar = torch.arange(T, device=x.device).unsqueeze(0)
            mask = ar < valid_len.unsqueeze(1)  # [B,T] True=valid

        # 끝에 패딩 추가(0) & 마스크는 False로
        pad = (self.k - (T % self.k)) % self.k
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))            # [B,T_pad,C]
            mask = F.pad(mask, (0, pad), value=False)

        T_pad = x.shape[1]
        assert (T_pad % self.k) == 0

        # 마스크드 평균 다운샘플
        x_pool, pooled_mask = windowed_masked_mean(x, mask, self.k)          # [B,T',C], [B,T']
        key_padding_mask = ~pooled_mask                                       # True=pad
        h = self.temporal(x_pool, key_padding_mask=key_padding_mask)          # [B,T',C]

        # 다운샘플 해상도에서 분류 → 최근접 업샘플 (repeat_interleave와 동일 효과)
        logits_pool = self.classifier(h)                                      # [B,T',num_classes]
        logits_up = F.interpolate(
            logits_pool.permute(0, 2, 1), size=T_pad, mode="nearest"
        ).permute(0, 2, 1)                                                    # [B,T_pad,num_classes]

        if pad > 0:
            logits_up = logits_up[:, :T, :]                                   # 원래 길이로 트림
        return logits_up


# ------------------------- Multi-resolution wrapper ----------------------------
class MultiResTemporalModel2(nn.Module):
    """
    옵션:
      - fusion: "average" | "learned" | "video_gate"
      - use_spatial_norm: spatial head 앞에 LayerNorm
      - spatial_dropout: spatial head 입력 드롭아웃
      - ks_weights: True면 k-경로 가중치(softmax) 학습
    """
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        ks: Tuple[int, ...] = (4, 6, 9),
        fusion: str = "average",
        use_spatial_branch: bool = True,
        use_spatial_norm: bool = False,
        spatial_dropout: float = 0.0,
        ks_weights: bool = False,
        temporal_cfg: Optional[Dict] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.ks = ks
        self.use_spatial_branch = use_spatial_branch
        self.fusion = fusion
        self.ks_weights = ks_weights

        self.spatial_norm = nn.LayerNorm(embed_dim) if use_spatial_norm else nn.Identity()
        self.spatial_drop = nn.Dropout(p=spatial_dropout) if spatial_dropout > 0 else nn.Identity()
        self.spatial_head = nn.Linear(embed_dim, num_classes)

        temporal_cfg = temporal_cfg or {}
        self.temporal_paths = nn.ModuleList([
            TemporalPathway(
                d_model=embed_dim,
                num_classes=num_classes,
                k=k,
                nhead=temporal_cfg.get("nhead", 4),
                num_layers=temporal_cfg.get("num_layers", 3),
                dim_feedforward=temporal_cfg.get("dim_feedforward", 2048),
                dropout=temporal_cfg.get("dropout", 0.1),
            )
            for k in ks
        ])

        if fusion == "learned":
            self.fusion_logit = nn.Parameter(torch.tensor(0.0))  # alpha = sigmoid(0)=0.5
        elif fusion == "video_gate":
            # Lightweight sequence-level gate from pooled spatial/temporal logits.
            self.video_gate_proj = nn.Linear(num_classes * 2, 1)

        if ks_weights:
            self.ks_logits = nn.Parameter(torch.zeros(len(ks)))   # softmax로 가중 합

    def forward(
        self,
        frame_embeddings: torch.Tensor,               # [B,T,C]
        valid_len: Optional[torch.Tensor] = None,     # [B]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, C = frame_embeddings.shape
        assert C == self.embed_dim

        parts: Dict[str, torch.Tensor | List[torch.Tensor]] = {}

        # Spatial branch
        spatial_in = self.spatial_norm(frame_embeddings)
        spatial_in = self.spatial_drop(spatial_in)
        spatial_logits = self.spatial_head(spatial_in)         # [B,T,num_classes]
        parts["spatial"] = spatial_logits

        # Temporal branches
        temporal_logits_list = [tp(frame_embeddings, valid_len=valid_len) for tp in self.temporal_paths]
        parts["temporal_list"] = temporal_logits_list

        if self.ks_weights:
            w = torch.softmax(self.ks_logits, dim=0)           # [K]
            temporal_mix = torch.stack(temporal_logits_list, dim=0)  # [K,B,T,C]
            temporal_mean = (w.view(-1, 1, 1, 1) * temporal_mix).sum(dim=0)
            parts["ks_weights"] = w
        else:
            temporal_mean = torch.stack(temporal_logits_list, dim=0).mean(dim=0)

        parts["temporal_mean"] = temporal_mean

        # Fusion
        if self.use_spatial_branch:
            if self.fusion == "learned":
                alpha = torch.sigmoid(self.fusion_logit)       # scalar
                logits = alpha * spatial_logits + (1 - alpha) * temporal_mean
                parts["alpha"] = alpha
            elif self.fusion == "video_gate":
                pooled_spatial = masked_temporal_mean(spatial_logits, valid_len=valid_len)   # [B,Ccls]
                pooled_temporal = masked_temporal_mean(temporal_mean, valid_len=valid_len)   # [B,Ccls]
                gate_input = torch.cat([pooled_spatial, pooled_temporal], dim=-1)            # [B,2*Ccls]
                video_gate = torch.sigmoid(self.video_gate_proj(gate_input)).view(B, 1, 1)
                logits = video_gate * spatial_logits + (1 - video_gate) * temporal_mean
                parts["video_gate"] = video_gate.view(B)
            else:
                logits = 0.5 * (spatial_logits + temporal_mean)
        else:
            logits = temporal_mean

        return logits, parts
