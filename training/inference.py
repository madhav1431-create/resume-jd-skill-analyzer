"""
inference.py
------------
Inference Engine for the AI Adaptive Onboarding Engine.

Provides a single high-level class `OnboardingEngine` that:
  1. Loads the trained model + tokenizer from a checkpoint
  2. Accepts raw text **or** PDF paths for resume & JD
  3. Predicts match score, skill sets, gaps, and roadmap
  4. Falls back to purely rule-based analysis when no checkpoint is available
     or when the model fails at runtime

Usage
─────
    from training.inference import OnboardingEngine

    engine = OnboardingEngine(checkpoint="checkpoints/best_model.pt")
    result = engine.analyze(resume_text="...", jd_text="...")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F

# ── project imports ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.skill_vocab   import NUM_SKILLS, IDX2SKILL
from data.dataset       import extract_skills
from models.transformer import SiameseOnboardingModel, ModelConfig
from utils.tokenizer    import WordTokenizer
from utils.text_utils   import extract_text_from_pdf, normalize_for_model
from utils.roadmap      import (
    compute_skill_gap,
    generate_roadmap,
    extract_skills_with_model,
)


# ─────────────────────────────────────────────────────────────────────────────
# ONBOARDING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class OnboardingEngine:
    """
    High-level inference wrapper.

    Parameters
    ----------
    checkpoint  : path to .pt checkpoint file (optional)
    tokenizer   : path to tokenizer JSON (optional; auto-inferred from ckpt dir)
    device      : "cuda" | "cpu" | "auto"
    threshold   : probability threshold for skill prediction  [0, 1]
    """

    def __init__(
        self,
        checkpoint: Optional[Union[str, Path]] = None,
        tokenizer:  Optional[Union[str, Path]] = None,
        device:     str   = "auto",
        threshold:  float = 0.45,
    ):
        self.threshold = threshold
        self.model:     Optional[SiameseOnboardingModel] = None
        self.tokenizer: Optional[WordTokenizer] = None

        # ── Device ────────────────────────────────────────────────────────
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # ── Load checkpoint ───────────────────────────────────────────────
        if checkpoint and Path(checkpoint).exists():
            self._load_model(Path(checkpoint), tokenizer)
            print(f"[Engine] Model loaded from {checkpoint} on {self.device}")
        else:
            print("[Engine] No checkpoint found → rule-based analysis only")

    @property
    def has_model(self) -> bool:
        """True when both model weights and tokenizer are available."""
        return self.model is not None and self.tokenizer is not None

    # ── Model Loading ─────────────────────────────────────────────────────

    def _load_model(
        self,
        ckpt_path: Path,
        tok_path:  Optional[Union[str, Path]],
    ) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device)

        # Rebuild model from saved config
        model_cfg  = ModelConfig(**ckpt["model_cfg"])
        self.model = SiameseOnboardingModel(model_cfg)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

        # Load tokenizer (default: sibling file next to checkpoint)
        if tok_path is None:
            tok_path = ckpt_path.parent / "tokenizer.json"
        if Path(tok_path).exists():
            self.tokenizer = WordTokenizer.load(tok_path)
        else:
            print("[Engine] Tokenizer not found – skill extraction falls back to rules")

    # ── Text Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _text_from_pdf(pdf_path: Union[str, Path]) -> str:
        """Extract and clean text from a PDF file."""
        return extract_text_from_pdf(pdf_path)

    @staticmethod
    def _resolve_text(
        text: Optional[str],
        pdf:  Optional[Union[str, Path]],
        label: str,
    ) -> str:
        """
        Return usable text from either a raw string or a PDF path.

        Priority: text > pdf.  Raises ValueError if both are empty.
        """
        if text and text.strip():
            return text.strip()
        if pdf and Path(pdf).exists():
            return extract_text_from_pdf(pdf)
        raise ValueError(f"Provide either {label}_text or {label}_pdf.")

    # ── Core Analysis ─────────────────────────────────────────────────────

    def _encode(self, text: str) -> torch.Tensor:
        """Encode a single text to a (1, D) embedding tensor."""
        ids  = self.tokenizer.encode_batch([text], max_len=256).to(self.device)
        mask = self.tokenizer.make_padding_mask(ids).to(self.device)
        with torch.no_grad():
            emb = self.model.encode(ids, mask)   # (1, D)
        return emb

    def _model_similarity(self, resume: str, jd: str) -> float:
        """Cosine similarity in [0, 1] (mapped from [-1, 1])."""
        r_emb = self._encode(resume)
        j_emb = self._encode(jd)
        cos   = F.cosine_similarity(r_emb, j_emb, dim=-1).item()
        return round((cos + 1.0) / 2.0, 3)

    def _model_skills(self, text: str) -> list[str]:
        """Model head + rule-based extraction for higher recall."""
        if not self.has_model:
            return extract_skills(text)
        return extract_skills_with_model(
            text, self.model, self.tokenizer, self.device, self.threshold
        )

    # ── Public API ────────────────────────────────────────────────────────

    def analyze(
        self,
        resume_text: Optional[str]              = None,
        resume_pdf:  Optional[Union[str, Path]] = None,
        jd_text:     Optional[str]              = None,
        jd_pdf:      Optional[Union[str, Path]] = None,
        include_prerequisites: bool             = True,
    ) -> dict:
        """
        Full analysis pipeline.

        Parameters
        ----------
        resume_text : raw resume string   (takes priority over resume_pdf)
        resume_pdf  : path to resume PDF  (used only if resume_text is empty)
        jd_text     : raw JD string       (takes priority over jd_pdf)
        jd_pdf      : path to JD PDF      (used only if jd_text is empty)

        Returns
        -------
        {
            "match_score":    float  [0, 1]
            "resume_skills":  list[str]
            "jd_skills":      list[str]
            "missing_skills": list[str]
            "gap_analysis":   dict
            "roadmap":        dict
            "method":         "model" | "rule-based"
        }
        """
        # 1. Resolve raw text
        resume = self._resolve_text(resume_text, resume_pdf, "resume")
        jd     = self._resolve_text(jd_text,     jd_pdf,     "jd")

        # 2. Normalise for model consumption
        resume_norm = normalize_for_model(resume)
        jd_norm     = normalize_for_model(jd)

        # 3. Skill extraction + similarity (model → rule-based fallback)
        method = "rule-based"
        if self.has_model:
            try:
                resume_skills = self._model_skills(resume_norm)
                jd_skills     = self._model_skills(jd_norm)
                match_score   = self._model_similarity(resume_norm, jd_norm)
                method = "model"
            except Exception as exc:
                print(f"[Engine] Model inference failed ({exc}); falling back to rules")
                resume_skills, jd_skills, match_score = self._rule_based(resume, jd)
        else:
            resume_skills, jd_skills, match_score = self._rule_based(resume, jd)

        # 4. Skill gap
        gap_analysis = compute_skill_gap(resume_skills, jd_skills)
        gap_analysis["match_score"] = match_score  # prefer model score

        # 5. Roadmap
        roadmap = generate_roadmap(
            resume_skills, jd_skills, include_prerequisites
        )

        return {
            "match_score":   match_score,
            "resume_skills": sorted(resume_skills),
            "jd_skills":     sorted(jd_skills),
            "missing_skills": gap_analysis.get("missing", []),
            "gap_analysis":  gap_analysis,
            "roadmap":       roadmap,
            "method":        method,
        }

    # ── Rule-based fallback (extracted for clarity) ───────────────────────

    @staticmethod
    def _rule_based(resume: str, jd: str):
        """Return (resume_skills, jd_skills, match_score) via keyword matching."""
        resume_skills = extract_skills(resume)
        jd_skills     = extract_skills(jd)
        gap           = compute_skill_gap(resume_skills, jd_skills)
        return resume_skills, jd_skills, gap["match_score"]


# ─────────────────────────────────────────────────────────────────────────────
# QUICK TEST (no checkpoint needed)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = OnboardingEngine()   # rule-based mode

    resume = """
    Senior Software Engineer with 6 years experience.
    Expert in Python, Django, REST API, SQL, PostgreSQL, git, Linux.
    Some exposure to Docker and AWS.
    """
    jd = """
    Seeking a Backend ML Engineer. Requirements:
    Python, PyTorch, machine learning, deep learning, Docker,
    Kubernetes, AWS, mlflow, CI/CD, PostgreSQL.
    """

    result = engine.analyze(resume_text=resume, jd_text=jd)

    print(f"Match Score : {result['match_score']:.0%}")
    print(f"Method      : {result['method']}")
    print(f"Resume skills ({len(result['resume_skills'])}):", result['resume_skills'])
    print(f"JD skills     ({len(result['jd_skills'])}):    ", result['jd_skills'])
    print(f"Missing       : {result['missing_skills']}")
    print(f"Total hours   : {result['roadmap']['total_hours']}")
    print("\nPhases:")
    for phase, skills in result["roadmap"]["phases"].items():
        print(f"  {phase}: {skills}")
