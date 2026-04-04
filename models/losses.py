"""
losses.py
---------
Multi-task loss for the Siamese Onboarding Model.

Tasks
─────
1. Similarity loss  – CosineEmbeddingLoss (label ∈ {+1, -1})
2. Skill loss       – BCEWithLogitsLoss on multi-hot skill vectors (for both
                      resume branch and JD branch)

The combined loss is a weighted sum:
    L = λ_sim * L_sim  +  λ_skill * (L_skill_resume + L_skill_jd) / 2

Weights are configurable via LossConfig.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOSS CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LossConfig:
    lambda_sim:   float = 1.0    # weight for similarity loss
    lambda_skill: float = 0.5    # weight for skill extraction loss
    sim_margin:   float = 0.0    # margin for CosineEmbeddingLoss
    pos_weight:   float = 5.0    # class imbalance weight for rare skills


# ─────────────────────────────────────────────────────────────────────────────
# 2. INDIVIDUAL LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

class SimilarityLoss(nn.Module):
    """
    CosineEmbeddingLoss:
        L = 1 - cos(e1, e2)          if y == +1
        L = max(0, cos(e1,e2) - margin) if y == -1
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin, reduction="mean")

    def forward(
        self,
        resume_emb: torch.Tensor,   # (B, D)
        jd_emb:     torch.Tensor,   # (B, D)
        labels:     torch.Tensor,   # (B,)  values ∈ {+1, -1}
    ) -> torch.Tensor:
        return self.loss_fn(resume_emb, jd_emb, labels)


class SkillExtractionLoss(nn.Module):
    """
    Binary cross-entropy with logits for multi-label skill prediction.
    pos_weight upweights the positive class to combat label sparsity.
    """

    def __init__(self, num_skills: int, pos_weight: float = 5.0, device: str = "cpu"):
        super().__init__()
        pw = torch.ones(num_skills) * pos_weight
        self.loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=pw.to(device), reduction="mean"
        )

    def forward(
        self,
        logits: torch.Tensor,   # (B, num_skills) – raw logits
        targets: torch.Tensor,  # (B, num_skills) – multi-hot float {0,1}
    ) -> torch.Tensor:
        return self.loss_fn(logits, targets)


# ─────────────────────────────────────────────────────────────────────────────
# 3. COMBINED MULTI-TASK LOSS
# ─────────────────────────────────────────────────────────────────────────────

class MultiTaskLoss(nn.Module):
    """
    Weighted combination of similarity and skill extraction losses.

    forward() returns a dict:
        total         : combined scalar loss (used for .backward())
        sim_loss      : similarity component
        skill_loss    : average skill component (resume + JD)
    """

    def __init__(self, cfg: LossConfig, num_skills: int, device: str = "cpu"):
        super().__init__()
        self.cfg        = cfg
        self.sim_loss   = SimilarityLoss(margin=cfg.sim_margin)
        self.skill_loss = SkillExtractionLoss(num_skills, cfg.pos_weight, device)

    def forward(
        self,
        resume_emb:           torch.Tensor,  # (B, D)
        jd_emb:               torch.Tensor,  # (B, D)
        labels:               torch.Tensor,  # (B,)  +1 / -1
        resume_skill_logits:  torch.Tensor,  # (B, num_skills)
        jd_skill_logits:      torch.Tensor,  # (B, num_skills)
        resume_skill_targets: torch.Tensor,  # (B, num_skills)
        jd_skill_targets:     torch.Tensor,  # (B, num_skills)
    ) -> dict[str, torch.Tensor]:

        # ── Task 1: Similarity ────────────────────────────────────────────
        l_sim = self.sim_loss(resume_emb, jd_emb, labels)

        # ── Task 2: Skill extraction  (both branches) ─────────────────────
        l_skill_resume = self.skill_loss(resume_skill_logits, resume_skill_targets)
        l_skill_jd     = self.skill_loss(jd_skill_logits,     jd_skill_targets)
        l_skill        = (l_skill_resume + l_skill_jd) / 2.0

        # ── Combined ──────────────────────────────────────────────────────
        total = self.cfg.lambda_sim * l_sim + self.cfg.lambda_skill * l_skill

        return {
            "total":      total,
            "sim_loss":   l_sim,
            "skill_loss": l_skill,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 4. METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_similarity_accuracy(
    similarity_scores: torch.Tensor,  # (B,)
    labels: torch.Tensor,             # (B,)  +1 / -1
    threshold: float = 0.5,
) -> float:
    """
    Binary accuracy: predict +1 if cosine_sim >= threshold, else -1.
    Returns accuracy in [0, 1].
    """
    preds = (similarity_scores >= threshold).float() * 2 - 1  # to {-1, +1}
    correct = (preds == labels).float().mean().item()
    return correct


def compute_skill_f1(
    logits: torch.Tensor,    # (B, num_skills)
    targets: torch.Tensor,   # (B, num_skills)
    threshold: float = 0.5,
) -> float:
    """
    Macro-averaged F1 score for multi-label skill prediction.
    """
    preds = (torch.sigmoid(logits) >= threshold).float()

    # per-sample TP, FP, FN
    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return f1.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data.skill_vocab import NUM_SKILLS
    B, D = 8, 256
    cfg = LossConfig()
    loss_fn = MultiTaskLoss(cfg, NUM_SKILLS)

    r_emb = torch.randn(B, D)
    j_emb = torch.randn(B, D)
    lbl   = torch.ones(B).float()
    lbl[4:] = -1.0
    r_log = torch.randn(B, NUM_SKILLS)
    j_log = torch.randn(B, NUM_SKILLS)
    r_tgt = (torch.rand(B, NUM_SKILLS) > 0.85).float()
    j_tgt = (torch.rand(B, NUM_SKILLS) > 0.85).float()

    out = loss_fn(r_emb, j_emb, lbl, r_log, j_log, r_tgt, j_tgt)
    print("total:     ", out["total"].item())
    print("sim_loss:  ", out["sim_loss"].item())
    print("skill_loss:", out["skill_loss"].item())
