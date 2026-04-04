"""
dataset.py
----------
Synthetic dataset creation pipeline.

Generates:
  • Positive pairs  – (resume, matching JD), label = +1
  • Negative pairs  – (resume, random JD),   label = -1
  • Hard-negative   – (resume, similar but wrong JD), label = -1

Also produces per-sample skill vectors (multi-hot, shape = NUM_SKILLS)
for the skill-extraction head.

Usage:
    from data.dataset import build_dataset, ResumeJDDataset
    train_ds, val_ds = build_dataset(n_samples=5000)
"""

import random
import re
from typing import Optional

import torch
from torch.utils.data import Dataset

# project imports
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.skill_vocab import SKILLS, SKILL2IDX, ALIASES, NUM_SKILLS


# ─────────────────────────────────────────────────────────────────────────────
# 1. TEXT TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

_RESUME_TEMPLATES = [
    (
        "Experienced {role} with {years} years of experience. "
        "Proficient in {skills_str}. "
        "Worked on {project}. "
        "Holds a degree in {degree}."
    ),
    (
        "I am a {role} skilled in {skills_str}. "
        "I have {years} years of professional experience. "
        "My recent project involved {project}. "
        "Education: {degree}."
    ),
    (
        "{role} | {years} YOE | {skills_str}. "
        "Projects: {project}. Background: {degree}."
    ),
]

_JD_TEMPLATES = [
    (
        "We are looking for a {role}. "
        "Requirements: {skills_str}. "
        "You will work on {project}. "
        "{years}+ years of experience required."
    ),
    (
        "Job Title: {role}. "
        "Must have: {skills_str}. "
        "Experience: {years} years minimum. "
        "Responsibilities include {project}."
    ),
    (
        "Hiring {role}. Needed skills: {skills_str}. "
        "Expected experience: {years} years. Role: {project}."
    ),
]

_ROLES = [
    "Data Scientist", "Machine Learning Engineer", "Backend Engineer",
    "Full Stack Developer", "Data Engineer", "MLOps Engineer",
    "NLP Engineer", "Computer Vision Engineer", "Software Engineer",
    "DevOps Engineer", "Cloud Architect", "AI Researcher",
]

_PROJECTS = [
    "building recommendation systems",
    "developing RESTful APIs at scale",
    "designing NLP pipelines for text classification",
    "automating CI/CD workflows",
    "creating real-time data streaming pipelines",
    "training and deploying transformer models",
    "building data warehouses on cloud platforms",
    "developing computer vision systems for object detection",
    "designing microservices architectures",
    "conducting A/B experiments and statistical analysis",
]

_DEGREES = [
    "Computer Science", "Data Science", "Electrical Engineering",
    "Mathematics", "Statistics", "Information Technology",
    "Artificial Intelligence", "Software Engineering",
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. SKILL EXTRACTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase and strip punctuation for matching."""
    return re.sub(r"[^a-z0-9 #+./]", " ", text.lower())


def extract_skills(text: str) -> list[str]:
    """
    Rule-based skill extractor.
    Returns a de-duplicated list of canonical skill names found in `text`.
    """
    norm = _normalize(text)
    found = set()

    # 1. Direct canonical match (longest first to avoid substring shadowing)
    for skill in sorted(SKILLS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(skill) + r"\b", norm):
            found.add(skill)

    # 2. Alias match
    for alias, canonical in ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", norm):
            found.add(canonical)

    return sorted(found)


def skills_to_vector(skills: list[str]) -> torch.Tensor:
    """Convert a list of skill names to a multi-hot float tensor."""
    vec = torch.zeros(NUM_SKILLS, dtype=torch.float32)
    for s in skills:
        if s in SKILL2IDX:
            vec[SKILL2IDX[s]] = 1.0
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# 3. SAMPLE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _sample_skill_set(n_skills: int = None) -> list[str]:
    n = n_skills or random.randint(4, 12)
    return random.sample(SKILLS, min(n, len(SKILLS)))


def _fill_template(templates: list[str], skills: list[str]) -> str:
    tmpl = random.choice(templates)
    return tmpl.format(
        role=random.choice(_ROLES),
        years=random.randint(1, 10),
        skills_str=", ".join(skills),
        project=random.choice(_PROJECTS),
        degree=random.choice(_DEGREES),
    )


def generate_sample(pair_type: str = "positive") -> dict:
    """
    Generate one (resume, JD, label, resume_skills, jd_skills) sample.

    pair_type:
        "positive"  – resume and JD share most skills → label +1
        "negative"  – resume and JD have disjoint skill sets → label -1
        "hard_neg"  – overlap ~30 %, still labelled -1
    """
    resume_skills = _sample_skill_set(random.randint(6, 14))

    if pair_type == "positive":
        # JD uses same core + a few extras
        extra = _sample_skill_set(random.randint(0, 4))
        jd_skills = list(set(resume_skills[:max(1, len(resume_skills) - 2)] + extra))
        label = 1
    elif pair_type == "hard_neg":
        # ~30 % overlap
        overlap_n = max(1, len(resume_skills) // 3)
        jd_skills = resume_skills[:overlap_n] + _sample_skill_set(random.randint(4, 8))
        jd_skills = list(set(jd_skills))
        label = -1
    else:  # "negative"
        # Completely different pool
        all_other = [s for s in SKILLS if s not in resume_skills]
        jd_skills = random.sample(all_other, min(random.randint(4, 10), len(all_other)))
        label = -1

    resume_text = _fill_template(_RESUME_TEMPLATES, resume_skills)
    jd_text = _fill_template(_JD_TEMPLATES, jd_skills)

    # Re-extract skills from generated text (simulates real extraction)
    extracted_resume_skills = extract_skills(resume_text)
    extracted_jd_skills = extract_skills(jd_text)

    # Fall back to ground-truth lists if extraction misses too much
    if len(extracted_resume_skills) < 2:
        extracted_resume_skills = resume_skills
    if len(extracted_jd_skills) < 2:
        extracted_jd_skills = jd_skills

    return {
        "resume_text":   resume_text,
        "jd_text":       jd_text,
        "label":         label,               # +1 or -1 for CosineEmbeddingLoss
        "resume_skills": extracted_resume_skills,
        "jd_skills":     extracted_jd_skills,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class ResumeJDDataset(Dataset):
    """
    PyTorch Dataset for (resume, JD) pairs.

    Each __getitem__ returns a dict with raw text strings and skill vectors.
    The collate / tokenisation is handled separately inside the training loop
    so that the Tokenizer can live on the training side.
    """

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "resume_text":   s["resume_text"],
            "jd_text":       s["jd_text"],
            "label":         torch.tensor(s["label"], dtype=torch.float32),
            "resume_skills": skills_to_vector(s["resume_skills"]),
            "jd_skills":     skills_to_vector(s["jd_skills"]),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. DATASET BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    n_samples: int = 5000,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[ResumeJDDataset, ResumeJDDataset]:
    """
    Build synthetic train / validation datasets.

    Distribution: 40 % positive, 30 % hard_neg, 30 % negative.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    n_pos  = int(n_samples * 0.40)
    n_hard = int(n_samples * 0.30)
    n_neg  = n_samples - n_pos - n_hard

    samples = (
        [generate_sample("positive")  for _ in range(n_pos)]
        + [generate_sample("hard_neg") for _ in range(n_hard)]
        + [generate_sample("negative") for _ in range(n_neg)]
    )
    random.shuffle(samples)

    split = int(len(samples) * (1 - val_ratio))
    return ResumeJDDataset(samples[:split]), ResumeJDDataset(samples[split:])


# ─────────────────────────────────────────────────────────────────────────────
# 6. QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_ds, val_ds = build_dataset(n_samples=200)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    sample = train_ds[0]
    print("Resume:", sample["resume_text"][:80])
    print("JD:    ", sample["jd_text"][:80])
    print("Label: ", sample["label"].item())
    print("Resume skills:", sample["resume_skills"].sum().int().item(), "active")
    print("JD skills:    ", sample["jd_skills"].sum().int().item(), "active")
