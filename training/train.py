"""
train.py
--------
Kaggle-ready training script for the AI Adaptive Onboarding Engine.

Features
────────
• Multi-GPU via nn.DataParallel (works on 2× T4 Kaggle GPUs out-of-the-box)
• Mixed-precision (fp16) with torch.cuda.amp
• Configurable via TrainConfig dataclass
• Checkpoint saving (best model by val loss)
• CSV + console logging
• Reproducible (global seed)

Usage (Kaggle notebook)
───────────────────────
    %run training/train.py

Usage (CLI)
───────────
    python training/train.py --n_samples 8000 --epochs 15 --batch_size 32
"""

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# ── project imports ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.dataset   import build_dataset
from data.skill_vocab import NUM_SKILLS
from models.transformer import SiameseOnboardingModel, ModelConfig
from models.losses      import MultiTaskLoss, LossConfig, compute_similarity_accuracy, compute_skill_f1
from utils.tokenizer    import WordTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 1. TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Data
    n_samples:    int   = 10_000
    val_ratio:    float = 0.15
    seed:         int   = 42
    max_len:      int   = 256

    # Model
    vocab_size:   int   = 10_004
    hidden_dim:   int   = 256
    ffn_dim:      int   = 1024
    num_layers:   int   = 3
    num_heads:    int   = 4
    dropout:      float = 0.1

    # Training
    epochs:       int   = 20
    batch_size:   int   = 32      # per GPU
    lr:           float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int   = 200
    clip_grad:    float = 1.0
    fp16:         bool  = True

    # Loss weights
    lambda_sim:   float = 1.0
    lambda_skill: float = 0.5
    sim_margin:   float = 0.0
    pos_weight:   float = 5.0

    # I/O
    save_dir:     str   = "checkpoints"
    log_interval: int   = 50     # steps


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEEDING
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# 3. TOKENISATION COLLATE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def make_collate_fn(tokenizer: WordTokenizer, max_len: int):
    """
    Factory: returns a collate function that tokenises text fields
    and stacks tensors into a batch dict.
    """

    def collate_fn(samples: list[dict]) -> dict:
        resume_texts = [s["resume_text"] for s in samples]
        jd_texts     = [s["jd_text"]     for s in samples]

        r_ids = tokenizer.encode_batch(resume_texts, max_len)
        j_ids = tokenizer.encode_batch(jd_texts,     max_len)

        r_mask = tokenizer.make_padding_mask(r_ids)
        j_mask = tokenizer.make_padding_mask(j_ids)

        labels        = torch.stack([s["label"]         for s in samples])
        resume_skills = torch.stack([s["resume_skills"] for s in samples])
        jd_skills     = torch.stack([s["jd_skills"]     for s in samples])

        return {
            "resume_ids":    r_ids,
            "jd_ids":        j_ids,
            "resume_mask":   r_mask,
            "jd_mask":       j_mask,
            "labels":        labels,
            "resume_skills": resume_skills,
            "jd_skills":     jd_skills,
        }

    return collate_fn


# ─────────────────────────────────────────────────────────────────────────────
# 4. WARMUP + COSINE LR SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

def get_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup → cosine decay."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        import math
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN / EVAL LOOPS
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, scheduler, scaler, loss_fn,
    device, epoch, cfg: TrainConfig,
) -> dict:
    model.train()
    total_loss = sim_loss = skill_loss = sim_acc = skill_f1 = 0.0
    n_batches = 0

    for step, batch in enumerate(loader):
        # Move tensors to device
        r_ids  = batch["resume_ids"].to(device)
        j_ids  = batch["jd_ids"].to(device)
        r_mask = batch["resume_mask"].to(device)
        j_mask = batch["jd_mask"].to(device)
        lbl    = batch["labels"].to(device)
        r_tgt  = batch["resume_skills"].to(device)
        j_tgt  = batch["jd_skills"].to(device)

        optimizer.zero_grad()

        with autocast(enabled=cfg.fp16):
            out = model(r_ids, j_ids, r_mask, j_mask)
            losses = loss_fn(
                out["resume_emb"], out["jd_emb"], lbl,
                out["resume_skills"], out["jd_skills"],
                r_tgt, j_tgt,
            )

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # ── Accumulate metrics ────────────────────────────────────────────
        total_loss  += losses["total"].item()
        sim_loss    += losses["sim_loss"].item()
        skill_loss  += losses["skill_loss"].item()
        sim_acc     += compute_similarity_accuracy(
                            out["similarity"].detach(), lbl.detach()
                        )
        skill_f1    += compute_skill_f1(
                            out["resume_skills"].detach(), r_tgt.detach()
                        )
        n_batches   += 1

        if (step + 1) % cfg.log_interval == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                f"  Epoch {epoch:02d} | Step {step+1:04d} "
                f"| loss={total_loss/n_batches:.4f} "
                f"| sim_acc={sim_acc/n_batches:.3f} "
                f"| skill_f1={skill_f1/n_batches:.3f} "
                f"| lr={lr:.2e}"
            )

    return {
        "loss":      total_loss  / max(n_batches, 1),
        "sim_loss":  sim_loss    / max(n_batches, 1),
        "skill_loss": skill_loss / max(n_batches, 1),
        "sim_acc":   sim_acc     / max(n_batches, 1),
        "skill_f1":  skill_f1   / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, cfg: TrainConfig) -> dict:
    model.eval()
    total_loss = sim_loss = skill_loss = sim_acc = skill_f1 = 0.0
    n_batches = 0

    for batch in loader:
        r_ids  = batch["resume_ids"].to(device)
        j_ids  = batch["jd_ids"].to(device)
        r_mask = batch["resume_mask"].to(device)
        j_mask = batch["jd_mask"].to(device)
        lbl    = batch["labels"].to(device)
        r_tgt  = batch["resume_skills"].to(device)
        j_tgt  = batch["jd_skills"].to(device)

        with autocast(enabled=cfg.fp16):
            out = model(r_ids, j_ids, r_mask, j_mask)
            losses = loss_fn(
                out["resume_emb"], out["jd_emb"], lbl,
                out["resume_skills"], out["jd_skills"],
                r_tgt, j_tgt,
            )

        total_loss  += losses["total"].item()
        sim_loss    += losses["sim_loss"].item()
        skill_loss  += losses["skill_loss"].item()
        sim_acc     += compute_similarity_accuracy(out["similarity"], lbl)
        skill_f1    += compute_skill_f1(out["resume_skills"], r_tgt)
        n_batches   += 1

    return {
        "loss":       total_loss  / max(n_batches, 1),
        "sim_loss":   sim_loss    / max(n_batches, 1),
        "skill_loss": skill_loss  / max(n_batches, 1),
        "sim_acc":    sim_acc     / max(n_batches, 1),
        "skill_f1":   skill_f1   / max(n_batches, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model, tokenizer, cfg, epoch, metrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unwrap DataParallel if needed
    raw_model = model.module if hasattr(model, "module") else model
    torch.save({
        "epoch":       epoch,
        "model_state": raw_model.state_dict(),
        "model_cfg":   asdict(raw_model.cfg),
        "metrics":     metrics,
        "train_cfg":   asdict(cfg),
    }, path)
    print(f"  ✓ Checkpoint saved → {path}")


def load_checkpoint(path: Path, device: str = "cpu"):
    """Load a checkpoint and return (model, tokenizer_path, epoch, metrics)."""
    ckpt = torch.load(path, map_location=device)
    model_cfg = ModelConfig(**ckpt["model_cfg"])
    model = SiameseOnboardingModel(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt["epoch"], ckpt["metrics"]


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Device setup ──────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"[Train] Device: {device} | GPUs available: {n_gpus}")

    # ── Build dataset ─────────────────────────────────────────────────────
    print(f"[Train] Generating {cfg.n_samples} synthetic samples …")
    train_ds, val_ds = build_dataset(cfg.n_samples, cfg.val_ratio, cfg.seed)
    print(f"[Train] Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Build & fit tokenizer ─────────────────────────────────────────────
    print("[Train] Building tokenizer …")
    tokenizer = WordTokenizer(max_vocab=cfg.vocab_size - 4)
    all_texts = (
        [s["resume_text"] for s in train_ds.samples]
        + [s["jd_text"]   for s in train_ds.samples]
    )
    tokenizer.build(all_texts, min_freq=1)
    cfg.vocab_size = tokenizer.vocab_size  # update to actual size
    tok_path = save_dir / "tokenizer.json"
    tokenizer.save(tok_path)
    print(f"[Train] Tokenizer vocab size: {tokenizer.vocab_size}")

    # ── DataLoaders ───────────────────────────────────────────────────────
    collate = make_collate_fn(tokenizer, cfg.max_len)
    eff_batch = cfg.batch_size * max(n_gpus, 1)
    train_loader = DataLoader(
        train_ds, batch_size=eff_batch, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=eff_batch, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=collate,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=cfg.hidden_dim,
        ffn_dim=cfg.ffn_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        max_len=cfg.max_len,
        dropout=cfg.dropout,
    )
    model = SiameseOnboardingModel(model_cfg).to(device)
    if n_gpus > 1:
        print(f"[Train] Using nn.DataParallel across {n_gpus} GPUs")
        model = nn.DataParallel(model)
    raw_model = model.module if hasattr(model, "module") else model
    print(f"[Train] Parameters: {raw_model.count_parameters():,}")

    # ── Optimizer, scheduler, scaler ──────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    scheduler = get_scheduler(optimizer, cfg.warmup_steps, total_steps)
    scaler = GradScaler(enabled=cfg.fp16)

    # ── Loss ──────────────────────────────────────────────────────────────
    loss_cfg = LossConfig(
        lambda_sim=cfg.lambda_sim,
        lambda_skill=cfg.lambda_skill,
        sim_margin=cfg.sim_margin,
        pos_weight=cfg.pos_weight,
    )
    loss_fn = MultiTaskLoss(loss_cfg, NUM_SKILLS, device=str(device))

    # ── CSV logger ────────────────────────────────────────────────────────
    log_path = save_dir / "training_log.csv"
    csv_file = open(log_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "epoch", "train_loss", "val_loss",
        "train_sim_acc", "val_sim_acc",
        "train_skill_f1", "val_skill_f1",
    ])
    csv_writer.writeheader()

    # ── Training loop ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    print(f"\n[Train] Starting training for {cfg.epochs} epochs …\n{'─'*60}")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            loss_fn, device, epoch, cfg,
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device, cfg)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | {elapsed:.1f}s | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_sim_acc={val_metrics['sim_acc']:.3f} | "
            f"val_skill_f1={val_metrics['skill_f1']:.3f}"
        )

        # ── Logging ───────────────────────────────────────────────────────
        csv_writer.writerow({
            "epoch":          epoch,
            "train_loss":     round(train_metrics["loss"],  4),
            "val_loss":       round(val_metrics["loss"],    4),
            "train_sim_acc":  round(train_metrics["sim_acc"],  4),
            "val_sim_acc":    round(val_metrics["sim_acc"],    4),
            "train_skill_f1": round(train_metrics["skill_f1"], 4),
            "val_skill_f1":   round(val_metrics["skill_f1"],   4),
        })
        csv_file.flush()

        # ── Checkpoint (best) ─────────────────────────────────────────────
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(
                model, tokenizer, cfg, epoch, val_metrics,
                save_dir / "best_model.pt",
            )

        # ── Checkpoint (last) ─────────────────────────────────────────────
        save_checkpoint(
            model, tokenizer, cfg, epoch, val_metrics,
            save_dir / "last_model.pt",
        )

    csv_file.close()
    print(f"\n[Train] Done! Best val loss: {best_val_loss:.4f}")
    print(f"[Train] Checkpoints saved in: {save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLI ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train AI Onboarding Engine")
    cfg = TrainConfig()
    for field_name, field_val in asdict(cfg).items():
        t = type(field_val)
        if t == bool:
            parser.add_argument(f"--{field_name}", default=field_val,
                                type=lambda x: x.lower() == "true")
        else:
            parser.add_argument(f"--{field_name}", default=field_val, type=t)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
