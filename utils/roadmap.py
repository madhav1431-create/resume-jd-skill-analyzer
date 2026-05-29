"""
roadmap.py
----------
Skill Gap Analysis + Intelligent Learning Roadmap Generator.

Pipeline
────────
1. Extract skills from resume and JD (rule-based or model-based).
2. Compute skill gap: missing = JD_skills − Resume_skills.
3. Auto-infer prerequisites via SKILL_LEVELS + SKILL_DOMAINS.
4. Sort by level to produce an ordered learning path.
5. Classify into phases (Foundation → Core → Advanced).
6. Format roadmap as a structured dict (consumed by the Streamlit UI).

Dependency Inference
────────────────────
Instead of a manually curated DEPENDENCIES dict, dependencies are
AUTOMATICALLY INFERRED:

    infer_dependencies(skill, candidate_pool)

Logic:
  1. Look up the skill's level (1/2/3) and domains (ml/data/web/…)
  2. Return all candidates that share ≥1 domain AND have a LOWER level.

Example:
  deep learning (L3, domain=ml) → depends on [python, numpy, …(L1),
  machine learning, pytorch, …(L2)] if they are in the candidate pool.

Phase Classification
────────────────────
Phases are assigned by skill level, NOT by iterating a dependency graph:
  Level 1 → Phase 1 — Foundations
  Level 2 → Phase 2 — Core Skills
  Level 3 → Phase 3 — Advanced

This guarantees correct separation — no cascading-into-one-phase bugs.
"""

from __future__ import annotations

import math
import sys
import os
from collections import defaultdict, deque
from typing import Optional

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.skill_vocab import (
    SKILLS, SKILL2IDX, IDX2SKILL, NUM_SKILLS,
    DEPENDENCIES,                         # legacy fallback
    SKILL_LEVELS, SKILL_DOMAINS, SKILL_TO_DOMAINS,
)
from data.dataset import extract_skills, skills_to_vector

import torch


# ─────────────────────────────────────────────────────────────────────────────
# 1. SKILL GAP COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_skill_gap(
    resume_skills: list[str],
    jd_skills:     list[str],
) -> dict:
    """
    Identify missing skills and compute a gap vector.

    Returns
    -------
    {
        "resume_skills":  list of skills found in resume
        "jd_skills":      list of skills required by JD
        "matched":        skills present in both
        "missing":        skills in JD but not in resume
        "extra":          skills in resume but not required
        "gap_ratio":      fraction of JD skills missing  [0, 1]
        "match_score":    keyword-based overlap score     [0, 1]
    }
    """
    resume_set = set(resume_skills)
    jd_set     = set(jd_skills)

    matched = sorted(resume_set & jd_set)
    missing = sorted(jd_set - resume_set)
    extra   = sorted(resume_set - jd_set)

    gap_ratio    = len(missing) / max(len(jd_set), 1)
    match_score  = len(matched) / max(len(jd_set), 1)

    return {
        "resume_skills": sorted(resume_skills),
        "jd_skills":     sorted(jd_skills),
        "matched":       matched,
        "missing":       missing,
        "extra":         extra,
        "gap_ratio":     round(gap_ratio, 3),
        "match_score":   round(match_score, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. AUTOMATIC DEPENDENCY INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def infer_dependencies(
    skill: str,
    candidate_pool: set[str],
) -> list[str]:
    """
    Automatically infer which skills from `candidate_pool` are
    prerequisites of `skill`.

    Algorithm
    ---------
    1. Look up skill's level (1/2/3) and domain set.
    2. Return every candidate that:
       a) shares at least ONE domain with `skill`, AND
       b) has a strictly LOWER level.

    If the skill is unknown (not in SKILL_LEVELS), falls back to the
    legacy DEPENDENCIES graph from skill_vocab.py.

    Parameters
    ----------
    skill           : the skill whose dependencies we want
    candidate_pool  : set of skills to search within (usually all_to_learn)

    Returns
    -------
    list of prerequisite skill names (sorted for determinism)
    """
    level = SKILL_LEVELS.get(skill)

    # Fallback: skill not in the level system → use legacy graph
    if level is None:
        legacy_deps = DEPENDENCIES.get(skill, [])
        return sorted(d for d in legacy_deps if d in candidate_pool)

    # Level 1 skills have NO prerequisites
    if level <= 1:
        return []

    domains = SKILL_TO_DOMAINS.get(skill, set())
    if not domains:
        return []

    deps: list[str] = []
    for candidate in candidate_pool:
        if candidate == skill:
            continue
        cand_level = SKILL_LEVELS.get(candidate, 1)
        if cand_level >= level:
            continue  # only lower-level skills
        cand_domains = SKILL_TO_DOMAINS.get(candidate, set())
        if cand_domains & domains:  # share at least one domain
            deps.append(candidate)

    return sorted(deps)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREREQUISITE EXPANSION (auto-inferred)
# ─────────────────────────────────────────────────────────────────────────────

def _expand_prerequisites(missing: list[str], known: set[str]) -> set[str]:
    """
    Discover transitive prerequisites of `missing` skills that the
    candidate does NOT already have (`known`).

    Uses BFS over inferred dependencies (not the manual graph).
    """
    # All skills in SKILL_LEVELS are potential prereqs
    all_skills_set = set(SKILL_LEVELS.keys())

    to_learn: set[str] = set()
    queue = deque(missing)
    visited: set[str] = set(missing) | known

    while queue:
        skill = queue.popleft()
        prereqs = infer_dependencies(skill, all_skills_set - known)
        for prereq in prereqs:
            if prereq not in visited:
                visited.add(prereq)
                to_learn.add(prereq)
                queue.append(prereq)

    return to_learn


# ─────────────────────────────────────────────────────────────────────────────
# 4. ROADMAP GENERATION
# ─────────────────────────────────────────────────────────────────────────────

# Rough time estimates (hours) per skill
_TIME_ESTIMATES: dict[str, int] = {
    # languages
    "python": 40, "java": 50, "javascript": 40, "typescript": 20,
    "c++": 60, "rust": 60, "go": 40, "r": 30, "scala": 40,
    # frameworks
    "react": 30, "django": 25, "flask": 15, "fastapi": 12,
    "nodejs": 25, "spring boot": 35,
    # data
    "sql": 25, "postgresql": 15, "mongodb": 15, "redis": 10,
    "elasticsearch": 20, "spark": 30, "kafka": 20, "airflow": 20,
    "dbt": 15, "snowflake": 15, "bigquery": 10, "hadoop": 25,
    # ML
    "machine learning": 50, "deep learning": 60, "neural networks": 40,
    "nlp": 40, "computer vision": 40, "reinforcement learning": 50,
    "transformers": 35, "large language models": 30, "llm": 30,
    "pytorch": 35, "tensorflow": 35, "scikit-learn": 20,
    "xgboost": 10, "lightgbm": 10, "huggingface": 20, "langchain": 15,
    "rag": 20, "fine-tuning": 20, "mlops": 30, "mlflow": 15, "wandb": 10,
    # stats
    "statistics": 30, "probability": 20, "linear algebra": 25,
    "bayesian inference": 25, "hypothesis testing": 15, "a/b testing": 10,
    "time series": 20, "calculus": 20,
    # devops
    "docker": 20, "kubernetes": 30, "linux": 25, "bash": 15, "git": 10,
    "aws": 40, "gcp": 35, "azure": 35, "cicd": 20, "terraform": 25,
}

_DEFAULT_HOURS = 15

_RESOURCES: dict[str, list[dict]] = {
    "python":           [{"title": "Python.org Tutorial",        "url": "https://docs.python.org/3/tutorial/"}],
    "machine learning": [{"title": "Andrew Ng — ML Specialization", "url": "https://www.coursera.org/specializations/machine-learning-introduction"}],
    "deep learning":    [{"title": "Fast.ai Practical Deep Learning", "url": "https://course.fast.ai/"}],
    "pytorch":          [{"title": "PyTorch Official Tutorials",   "url": "https://pytorch.org/tutorials/"}],
    "docker":           [{"title": "Docker Get Started",           "url": "https://docs.docker.com/get-started/"}],
    "kubernetes":       [{"title": "Kubernetes Basics",            "url": "https://kubernetes.io/docs/tutorials/kubernetes-basics/"}],
    "sql":              [{"title": "SQLZoo",                       "url": "https://sqlzoo.net/"}],
    "git":              [{"title": "Pro Git Book",                 "url": "https://git-scm.com/book/en/v2"}],
    "linux":            [{"title": "Linux Journey",               "url": "https://linuxjourney.com/"}],
    "aws":              [{"title": "AWS Training",                 "url": "https://aws.amazon.com/training/"}],
    "transformers":     [{"title": "HuggingFace NLP Course",       "url": "https://huggingface.co/learn/nlp-course/"}],
    "statistics":       [{"title": "StatQuest YouTube",           "url": "https://www.youtube.com/@statquest"}],
    "react":            [{"title": "React Official Docs",         "url": "https://react.dev/learn"}],
    "tensorflow":       [{"title": "TensorFlow Tutorials",        "url": "https://www.tensorflow.org/tutorials"}],
    "numpy":            [{"title": "NumPy Quickstart",            "url": "https://numpy.org/doc/stable/user/quickstart.html"}],
    "pandas":           [{"title": "Pandas Getting Started",      "url": "https://pandas.pydata.org/docs/getting_started/"}],
    "scikit-learn":     [{"title": "Scikit-learn Tutorials",      "url": "https://scikit-learn.org/stable/tutorial/"}],
    "linear algebra":   [{"title": "3Blue1Brown — Essence of LA", "url": "https://www.3blue1brown.com/topics/linear-algebra"}],
    "probability":      [{"title": "Khan Academy — Probability",  "url": "https://www.khanacademy.org/math/statistics-probability"}],
}


# ── Phase definitions ────────────────────────────────────────────────────────
_LEVEL_TO_PHASE = {
    1: "Phase 1 — Foundations",
    2: "Phase 2 — Core Skills",
    3: "Phase 3 — Advanced",
}

_PHASE_NAMES = [
    "Phase 1 — Foundations",
    "Phase 2 — Core Skills",
    "Phase 3 — Advanced",
]


def generate_roadmap(
    resume_skills: list[str],
    jd_skills:     list[str],
    include_prerequisites: bool = True,
) -> dict:
    """
    Generate a personalized, dependency-aware learning roadmap.

    Dependencies are AUTOMATICALLY INFERRED from skill levels and
    domain overlap — no manual dependency graph needed.

    Parameters
    ----------
    resume_skills          : skills extracted from the resume
    jd_skills              : skills required by the JD
    include_prerequisites  : if True, add transitive prerequisites to plan

    Returns
    -------
    {
        "gap_analysis":  { ... },
        "learning_plan": [
            {
                "step":       1,
                "skill":      "python",
                "type":       "prerequisite" | "required",
                "level":      1,
                "phase":      "foundation" | "core" | "advanced",
                "hours":      40,
                "days":       8,
                "resources":  [ {"title": ..., "url": ...} ],
                "depends_on": ["linux"]
            }, ...
        ],
        "total_hours":   120,
        "phases":        { "Phase 1 — Foundations": [...], ... },
    }
    """
    gap = compute_skill_gap(resume_skills, jd_skills)
    missing = gap["missing"]

    if not missing:
        return {
            "gap_analysis":  gap,
            "learning_plan": [],
            "total_hours":   0,
            "phases":        {},
            "message":       "🎉 Your resume already covers all required skills!",
        }

    known = set(resume_skills)
    missing_set = set(missing)

    # ── Expand with auto-inferred prerequisites ──────────────────────────
    if include_prerequisites:
        prereqs = _expand_prerequisites(missing, known)
    else:
        prereqs = set()

    all_to_learn = missing_set | prereqs

    # ── Build learning plan sorted by level then alphabetically ──────────
    items_with_level = []
    for skill in all_to_learn:
        level = SKILL_LEVELS.get(skill, 2)
        items_with_level.append((level, skill))
    items_with_level.sort(key=lambda x: (x[0], x[1]))

    learning_plan: list[dict] = []
    for i, (level, skill) in enumerate(items_with_level, 1):
        skill_type = "required" if skill in missing_set else "prerequisite"
        hours = _TIME_ESTIMATES.get(skill, _DEFAULT_HOURS)
        days = math.ceil(hours / 5)
        resources = _RESOURCES.get(
            skill,
            [{"title": f"Learn {skill.title()} on Coursera / YouTube",
              "url": f"https://www.coursera.org/search?query={skill.replace(' ', '+')}"}],
        )
        # Auto-infer dependencies (only include deps that are IN the plan)
        deps = infer_dependencies(skill, all_to_learn)

        phase_label = {1: "foundation", 2: "core", 3: "advanced"}.get(level, "core")

        learning_plan.append({
            "step":       i,
            "skill":      skill,
            "type":       skill_type,
            "level":      level,
            "phase":      phase_label,
            "hours":      hours,
            "days":       days,
            "resources":  resources,
            "depends_on": deps,
        })

    total_hours = sum(item["hours"] for item in learning_plan)

    # ── Phase grouping (level-based — guaranteed correct) ────────────────
    phases: dict[str, list[str]] = {p: [] for p in _PHASE_NAMES}
    for item in learning_plan:
        phase_key = _LEVEL_TO_PHASE.get(item["level"], "Phase 2 — Core Skills")
        phases[phase_key].append(item["skill"])

    # Remove empty phases
    phases = {k: v for k, v in phases.items() if v}

    return {
        "gap_analysis":  gap,
        "learning_plan": learning_plan,
        "total_hours":   total_hours,
        "phases":        phases,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. MODEL-BASED SKILL EXTRACTION (inference-time helper)
# ─────────────────────────────────────────────────────────────────────────────

def extract_skills_with_model(
    text: str,
    model,
    tokenizer,
    device: str = "cpu",
    threshold: float = 0.5,
) -> list[str]:
    """
    Use the trained skill-extraction head to predict skills from `text`.
    Falls back to rule-based extraction if model is None.
    """
    if model is None:
        return extract_skills(text)

    model.eval()
    with torch.no_grad():
        ids  = tokenizer.encode_batch([text], max_len=256).to(device)
        mask = tokenizer.make_padding_mask(ids).to(device)
        emb  = model.encode(ids, mask)                          # (1, D)
        logits = model.skill_head(emb)                          # (1, NUM_SKILLS)
        probs  = torch.sigmoid(logits).squeeze(0)               # (NUM_SKILLS,)

    predicted = [IDX2SKILL[i] for i, p in enumerate(probs) if p.item() >= threshold]

    # Merge with rule-based for higher recall
    rule_based = extract_skills(text)
    return sorted(set(predicted) | set(rule_based))


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    resume_text = """
    Data Scientist with 4 years experience. Skilled in Python, pandas, numpy,
    scikit-learn, statistics, SQL, git, linux, matplotlib. Built ML pipelines.
    """
    jd_text = """
    Looking for an MLOps Engineer. Must know: Python, PyTorch, Docker,
    Kubernetes, AWS, mlflow, machine learning, deep learning, cicd, git.
    """
    res_skills = extract_skills(resume_text)
    jd_skills  = extract_skills(jd_text)
    print("Resume skills:", res_skills)
    print("JD skills:    ", jd_skills)

    roadmap = generate_roadmap(res_skills, jd_skills)

    print("\n── Gap Analysis ────────────────────────────────")
    g = roadmap["gap_analysis"]
    print("Matched :", g["matched"])
    print("Missing :", g["missing"])
    print("Match % :", g["match_score"])

    print("\n── Learning Plan ───────────────────────────────")
    for item in roadmap["learning_plan"]:
        print(f"  Step {item['step']:02d}: {item['skill']:<30} "
              f"L{item['level']}  [{item['type']:12}] "
              f"~{item['hours']}h/{item['days']}d  "
              f"phase={item['phase']:<11} "
              f"deps={item['depends_on']}")

    print(f"\nTotal hours: {roadmap['total_hours']}")
    print("\nPhases:")
    for phase, skills in roadmap["phases"].items():
        print(f"  {phase}: {skills}")

    # ── Test infer_dependencies directly ──────────────────────────────────
    print("\n── Dependency Inference Examples ────────────────")
    pool = {"python", "statistics", "linear algebra", "machine learning",
            "deep learning", "pytorch", "numpy"}
    for s in ["deep learning", "machine learning", "python"]:
        deps = infer_dependencies(s, pool)
        print(f"  {s} → {deps}")
