"""
transformer.py
--------------
Multi-Task Siamese Transformer built from scratch in PyTorch.

Architecture
────────────
  SharedTextEncoder
      ├── Embedding layer  (trainable)
      ├── Positional Encoding  (sinusoidal, fixed)
      ├── N × TransformerEncoderLayer  (custom, no nn.Transformer dependency)
      └── Mean Pooling  →  (B, hidden_dim)

  SiameseOnboardingModel
      ├── SharedTextEncoder  (weights shared between resume & JD branches)
      ├── SimilarityHead     →  cosine similarity scalar
      └── SkillExtractionHead →  multi-label logits  (B, NUM_SKILLS)

All hyper-parameters are passed via a ModelConfig dataclass.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.skill_vocab import NUM_SKILLS


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    vocab_size:   int   = 10_004   # tokenizer vocab + 4 special tokens
    hidden_dim:   int   = 256      # transformer hidden / d_model
    ffn_dim:      int   = 1024     # feed-forward inner dimension (4× hidden)
    num_layers:   int   = 3        # number of encoder layers
    num_heads:    int   = 4        # attention heads (hidden_dim % num_heads == 0)
    max_len:      int   = 256      # maximum token sequence length
    dropout:      float = 0.1
    num_skills:   int   = NUM_SKILLS  # output size of skill head


# ─────────────────────────────────────────────────────────────────────────────
# 2. POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al. 2017).
    Adds positional signal to token embeddings; not learnable.
    """

    def __init__(self, hidden_dim: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build (1, max_len, hidden_dim) buffer
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / hidden_dim)
        )  # (hidden_dim/2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:hidden_dim // 2])

        pe = pe.unsqueeze(0)  # (1, max_len, hidden_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, hidden_dim)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CUSTOM MULTI-HEAD SELF-ATTENTION
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """
    Scaled dot-product multi-head self-attention (from scratch).
    Supports an optional boolean key_padding_mask (True = ignore).
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim  = hidden_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                        # (B, L, D)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, L) True=pad
    ) -> torch.Tensor:
        B, L, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        # Project and reshape to (B, H, L, Dh)
        def _project(linear, t):
            return linear(t).view(B, L, H, Dh).transpose(1, 2)

        Q = _project(self.q_proj, x)
        K = _project(self.k_proj, x)
        V = _project(self.v_proj, x)

        # Scaled dot-product attention: (B, H, L, L)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Mask padding positions with -inf before softmax
        if key_padding_mask is not None:
            # (B, 1, 1, L) → broadcast over heads and queries
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum and reshape
        out = torch.matmul(attn, V)          # (B, H, L, Dh)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)


# ─────────────────────────────────────────────────────────────────────────────
# 4. CUSTOM TRANSFORMER ENCODER LAYER
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoderLayer(nn.Module):
    """
    Pre-LN Transformer encoder layer (more stable than original post-LN):
      x → LayerNorm → MultiHeadSelfAttention → residual
      x → LayerNorm → FFN → residual
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)

        # Position-wise feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LN self-attention block
        x = x + self.self_attn(self.norm1(x), key_padding_mask)
        # Pre-LN FFN block
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 5. SHARED TEXT ENCODER
# ─────────────────────────────────────────────────────────────────────────────

class SharedTextEncoder(nn.Module):
    """
    Shared encoder branch used for BOTH resume and JD.

    Pipeline:
        token_ids (B, L)
          → Embedding (B, L, D)
          → PositionalEncoding
          → N × TransformerEncoderLayer
          → mean pooling over non-padding positions
          → (B, D)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embedding  = nn.Embedding(cfg.vocab_size, cfg.hidden_dim, padding_idx=0)
        self.pos_enc    = SinusoidalPositionalEncoding(cfg.hidden_dim, cfg.max_len, cfg.dropout)
        self.layers     = nn.ModuleList([
            TransformerEncoderLayer(cfg.hidden_dim, cfg.ffn_dim, cfg.num_heads, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        token_ids: torch.Tensor,               # (B, L)
        padding_mask: Optional[torch.Tensor] = None,  # (B, L) True=pad
    ) -> torch.Tensor:
        # Embed + positional encode
        x = self.embedding(token_ids)          # (B, L, D)
        x = self.pos_enc(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)                       # (B, L, D)

        # ── Mean pooling (ignore PAD positions) ───────────────────────────
        if padding_mask is not None:
            # non_pad: (B, L, 1), True where token is NOT padding
            non_pad = (~padding_mask).unsqueeze(-1).float()
            x = (x * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1e-9)
        else:
            x = x.mean(dim=1)                  # (B, D)

        return x                               # (B, D)


# ─────────────────────────────────────────────────────────────────────────────
# 6. TASK HEADS
# ─────────────────────────────────────────────────────────────────────────────

class SimilarityHead(nn.Module):
    """
    Computes cosine similarity between resume and JD embeddings.
    Returns a scalar in [-1, 1] per pair.
    """

    def forward(
        self,
        resume_emb: torch.Tensor,  # (B, D)
        jd_emb:     torch.Tensor,  # (B, D)
    ) -> torch.Tensor:             # (B,)
        return F.cosine_similarity(resume_emb, jd_emb, dim=-1)


class SkillExtractionHead(nn.Module):
    """
    Multi-label classification head.

    Input:  (B, D) embedding
    Output: (B, NUM_SKILLS) raw logits  (use sigmoid for probabilities)
    """

    def __init__(self, hidden_dim: int, num_skills: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_skills),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)  # (B, num_skills) – raw logits


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class SiameseOnboardingModel(nn.Module):
    """
    Multi-task Siamese Transformer for Resume–JD matching.

    Forward returns a dict with:
        similarity:    (B,)             cosine similarity score
        resume_skills: (B, NUM_SKILLS)  logits for resume skill extraction
        jd_skills:     (B, NUM_SKILLS)  logits for JD skill extraction
        resume_emb:    (B, D)           resume embedding
        jd_emb:        (B, D)           JD embedding
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # Shared encoder (weights tied between both branches)
        self.encoder = SharedTextEncoder(cfg)
        # Task heads
        self.sim_head   = SimilarityHead()
        self.skill_head = SkillExtractionHead(cfg.hidden_dim, cfg.num_skills, cfg.dropout)

    def encode(
        self,
        token_ids: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a single sequence → (B, D)."""
        return self.encoder(token_ids, padding_mask)

    def forward(
        self,
        resume_ids:    torch.Tensor,                    # (B, L)
        jd_ids:        torch.Tensor,                    # (B, L)
        resume_mask:   Optional[torch.Tensor] = None,   # (B, L)
        jd_mask:       Optional[torch.Tensor] = None,   # (B, L)
    ) -> dict[str, torch.Tensor]:

        # ── Encode both branches with shared weights ───────────────────────
        resume_emb = self.encoder(resume_ids, resume_mask)  # (B, D)
        jd_emb     = self.encoder(jd_ids,     jd_mask)      # (B, D)

        # ── Similarity head ────────────────────────────────────────────────
        similarity = self.sim_head(resume_emb, jd_emb)      # (B,)

        # ── Skill extraction heads ─────────────────────────────────────────
        resume_skill_logits = self.skill_head(resume_emb)   # (B, NUM_SKILLS)
        jd_skill_logits     = self.skill_head(jd_emb)       # (B, NUM_SKILLS)

        return {
            "similarity":    similarity,
            "resume_skills": resume_skill_logits,
            "jd_skills":     jd_skill_logits,
            "resume_emb":    resume_emb,
            "jd_emb":        jd_emb,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 8. QUICK SHAPE CHECK
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = ModelConfig()
    model = SiameseOnboardingModel(cfg)
    print(f"Parameters: {model.count_parameters():,}")

    B, L = 4, 64
    r_ids = torch.randint(4, cfg.vocab_size, (B, L))
    j_ids = torch.randint(4, cfg.vocab_size, (B, L))
    r_mask = r_ids == 0
    j_mask = j_ids == 0

    out = model(r_ids, j_ids, r_mask, j_mask)
    print("similarity:   ", out["similarity"].shape)
    print("resume_skills:", out["resume_skills"].shape)
    print("jd_skills:    ", out["jd_skills"].shape)
    print("resume_emb:   ", out["resume_emb"].shape)
