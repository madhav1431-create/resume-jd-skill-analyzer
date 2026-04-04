"""
engine.py — AI Adaptive Onboarding Engine (upgraded)
------------------------------------------------------
Drop-in replacement for the original engine.py.
New features vs original:
  • 80+ skills (up from 16)
  • Alias / synonym matching (e.g. "unix" → "linux", "ml" → "machine learning")
  • Skill dependency graph with topological sort for ordered roadmap
  • Per-skill time estimates and resource links
  • Richer generate_overview with context-aware paragraphs
  • Readiness score breakdown
  • Works with pdfplumber (same as original)
"""

import re
import pdfplumber
from collections import deque


# ─────────────────────────────────────────────────────────────────────────────
# 1. SKILL VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────

# Canonical skill names
SKILLS = [
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go",
    "rust", "r", "scala", "kotlin", "swift", "php", "ruby", "matlab",
    # Web
    "html", "css", "react", "angular", "vue", "nodejs", "django", "flask",
    "fastapi", "spring boot", "rest api", "graphql",
    # Data / DB
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "cassandra", "sqlite", "nosql", "spark", "hadoop", "kafka", "airflow",
    "dbt", "snowflake", "bigquery",
    # ML / AI
    "machine learning", "deep learning", "neural networks", "nlp",
    "computer vision", "statistics", "probability", "linear algebra",
    "pytorch", "tensorflow", "keras", "scikit-learn", "xgboost",
    "huggingface", "transformers", "large language models",
    "feature engineering", "mlflow", "model deployment", "mlops",
    # DevOps / Cloud
    "docker", "kubernetes", "linux", "unix", "bash", "git", "github",
    "aws", "gcp", "azure", "cicd", "terraform",
    # SWE
    "data structures", "algorithms", "system design", "microservices",
    "object oriented programming", "unit testing", "agile",
    # General
    "excel", "tableau", "power bi", "communication", "problem solving",
]

# Aliases → canonical skill
ALIASES = {
    "py":          "python",
    "js":          "javascript",
    "ts":          "typescript",
    "node":        "nodejs",
    "node.js":     "nodejs",
    "express":     "nodejs",
    "postgres":    "postgresql",
    "pg":          "postgresql",
    "mongo":       "mongodb",
    "elastic":     "elasticsearch",
    "sk-learn":    "scikit-learn",
    "sklearn":     "scikit-learn",
    "dl":          "deep learning",
    "ml":          "machine learning",
    "cv":          "computer vision",
    "llm":         "large language models",
    "llms":        "large language models",
    "bert":        "transformers",
    "gpt":         "large language models",
    "tf":          "tensorflow",
    "k8s":         "kubernetes",
    "unix":        "linux",       # treat unix as linux
    "html/css":    "html",        # resume says "HTML/CSS" → capture both
    "ci/cd":       "cicd",
    "ci cd":       "cicd",
    "oop":         "object oriented programming",
    "dsa":         "data structures",
    "version control": "git",
    "github actions": "cicd",
    "amazon web services": "aws",
    "google cloud":      "gcp",
    "google cloud platform": "gcp",
    "microsoft azure":   "azure",
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. DEPENDENCY GRAPH  (skill → list of prerequisites)
# ─────────────────────────────────────────────────────────────────────────────

DEPENDENCIES = {
    "machine learning":   ["python", "statistics", "linear algebra"],
    "deep learning":      ["machine learning", "python"],
    "neural networks":    ["deep learning"],
    "nlp":                ["deep learning", "python"],
    "computer vision":    ["deep learning", "python"],
    "transformers":       ["deep learning", "nlp"],
    "large language models": ["transformers"],
    "pytorch":            ["python", "linear algebra"],
    "tensorflow":         ["python", "linear algebra"],
    "keras":              ["tensorflow"],
    "scikit-learn":       ["python"],
    "xgboost":            ["machine learning", "scikit-learn"],
    "feature engineering":["python", "statistics"],
    "mlflow":             ["machine learning", "python"],
    "model deployment":   ["machine learning", "docker"],
    "mlops":              ["model deployment", "docker", "kubernetes"],
    "docker":             ["linux"],
    "kubernetes":         ["docker"],
    "cicd":               ["git", "docker"],
    "terraform":          ["linux", "aws"],
    "django":             ["python", "sql"],
    "flask":              ["python"],
    "fastapi":            ["python"],
    "react":              ["javascript", "html", "css"],
    "angular":            ["typescript", "html", "css"],
    "vue":                ["javascript", "html", "css"],
    "typescript":         ["javascript"],
    "nodejs":             ["javascript"],
    "graphql":            ["rest api"],
    "postgresql":         ["sql"],
    "mysql":              ["sql"],
    "spark":              ["python", "sql"],
    "kafka":              ["linux"],
    "airflow":            ["python", "docker"],
    "bigquery":           ["sql", "gcp"],
    "snowflake":          ["sql"],
    "algorithms":         ["data structures", "python"],
    "system design":      ["algorithms", "sql"],
    "microservices":      ["docker", "rest api"],
    "bash":               ["linux"],
    "statistics":         ["probability"],
    "linear algebra":     [],
    "probability":        [],
    "python":             [],
    "java":               [],
    "javascript":         [],
    "html":               [],
    "css":                [],
    "linux":              [],
    "git":                [],
    "sql":                [],
    "aws":                ["linux"],
    "gcp":                ["linux"],
    "azure":              ["linux"],
    "data structures":    [],
    "rest api":           [],
    "unit testing":       ["python"],
    "agile":              [],
    "object oriented programming": ["python"],
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. TIME ESTIMATES & RESOURCES
# ─────────────────────────────────────────────────────────────────────────────

SKILL_META = {
    "python":             {"days": 14, "url": "https://docs.python.org/3/tutorial/",            "resource": "Python Official Tutorial"},
    "machine learning":   {"days": 30, "url": "https://www.coursera.org/specializations/machine-learning-introduction", "resource": "Andrew Ng — ML Specialization"},
    "deep learning":      {"days": 40, "url": "https://course.fast.ai/",                        "resource": "Fast.ai Practical Deep Learning"},
    "statistics":         {"days": 14, "url": "https://www.youtube.com/@statquest",             "resource": "StatQuest YouTube"},
    "probability":        {"days": 10, "url": "https://www.khanacademy.org/math/statistics-probability", "resource": "Khan Academy — Probability"},
    "linear algebra":     {"days": 10, "url": "https://www.youtube.com/playlist?list=PL49CF3715CB9EF31D", "resource": "3Blue1Brown — Essence of Linear Algebra"},
    "sql":                {"days": 7,  "url": "https://sqlzoo.net/",                            "resource": "SQLZoo"},
    "docker":             {"days": 7,  "url": "https://docs.docker.com/get-started/",           "resource": "Docker Get Started"},
    "kubernetes":         {"days": 14, "url": "https://kubernetes.io/docs/tutorials/",         "resource": "Kubernetes Tutorials"},
    "linux":              {"days": 7,  "url": "https://linuxjourney.com/",                      "resource": "Linux Journey"},
    "git":                {"days": 3,  "url": "https://git-scm.com/book/en/v2",                 "resource": "Pro Git Book"},
    "aws":                {"days": 21, "url": "https://aws.amazon.com/training/",               "resource": "AWS Training & Certification"},
    "gcp":                {"days": 21, "url": "https://cloud.google.com/training",              "resource": "Google Cloud Training"},
    "azure":              {"days": 21, "url": "https://learn.microsoft.com/en-us/azure/",       "resource": "Microsoft Learn — Azure"},
    "pytorch":            {"days": 21, "url": "https://pytorch.org/tutorials/",                 "resource": "PyTorch Official Tutorials"},
    "tensorflow":         {"days": 21, "url": "https://www.tensorflow.org/tutorials",          "resource": "TensorFlow Tutorials"},
    "scikit-learn":       {"days": 10, "url": "https://scikit-learn.org/stable/tutorial/",     "resource": "Scikit-Learn Tutorials"},
    "react":              {"days": 14, "url": "https://react.dev/learn",                        "resource": "React Official Docs"},
    "javascript":         {"days": 14, "url": "https://javascript.info/",                      "resource": "The Modern JavaScript Tutorial"},
    "typescript":         {"days": 7,  "url": "https://www.typescriptlang.org/docs/",           "resource": "TypeScript Official Docs"},
    "html":               {"days": 5,  "url": "https://developer.mozilla.org/en-US/docs/Learn/HTML", "resource": "MDN — HTML"},
    "css":                {"days": 5,  "url": "https://developer.mozilla.org/en-US/docs/Learn/CSS",  "resource": "MDN — CSS"},
    "data structures":    {"days": 14, "url": "https://neetcode.io/",                          "resource": "NeetCode — DSA"},
    "algorithms":         {"days": 14, "url": "https://neetcode.io/",                          "resource": "NeetCode — Algorithms"},
    "system design":      {"days": 21, "url": "https://github.com/donnemartin/system-design-primer", "resource": "System Design Primer"},
    "nlp":                {"days": 21, "url": "https://huggingface.co/learn/nlp-course/",       "resource": "HuggingFace NLP Course"},
    "transformers":       {"days": 21, "url": "https://huggingface.co/learn/nlp-course/",       "resource": "HuggingFace NLP Course"},
    "large language models": {"days": 14, "url": "https://www.deeplearning.ai/short-courses/", "resource": "DeepLearning.AI Short Courses"},
    "mlops":              {"days": 21, "url": "https://madewithml.com/",                        "resource": "Made With ML — MLOps"},
    "mlflow":             {"days": 5,  "url": "https://mlflow.org/docs/latest/index.html",      "resource": "MLflow Docs"},
    "model deployment":   {"days": 10, "url": "https://fastapi.tiangolo.com/",                  "resource": "FastAPI — Model Serving"},
}

DEFAULT_META = {"days": 7, "url": "https://www.coursera.org/search?query={skill}", "resource": "Search on Coursera"}


# ─────────────────────────────────────────────────────────────────────────────
# 4. CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(file_obj) -> str:
    """Extract and return lowercased text from a PDF file object."""
    text = ""
    with pdfplumber.open(file_obj) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.lower()


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation for matching."""
    return re.sub(r"[^a-z0-9 #+./]", " ", text.lower())


def detect_skills(text: str) -> list[str]:
    """
    Extract canonical skill names from text using direct match + alias lookup.
    Returns a de-duplicated, sorted list.
    """
    norm = _normalize(text)
    found = set()

    # 1. Direct canonical match (longest-first to avoid substring shadowing)
    for skill in sorted(SKILLS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(skill) + r"\b", norm):
            found.add(skill)

    # 2. Alias match
    for alias, canonical in ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", norm):
            found.add(canonical)

    return sorted(found)


def compute_skill_gap(resume_skills: list[str], jd_skills: list[str]) -> list[str]:
    """Return skills required by JD that are absent from resume."""
    resume_set = set(resume_skills)
    return [s for s in jd_skills if s not in resume_set]


def get_skill_meta(skill: str) -> dict:
    """Return time estimate + resource link for a skill."""
    meta = SKILL_META.get(skill, DEFAULT_META.copy())
    if "{skill}" in meta.get("url", ""):
        meta = meta.copy()
        meta["url"] = meta["url"].replace("{skill}", skill.replace(" ", "+"))
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# 5. TOPOLOGICAL ROADMAP
# ─────────────────────────────────────────────────────────────────────────────

def _get_prerequisites(skills: list[str], known: set[str]) -> set[str]:
    """BFS over DEPENDENCIES to collect all transitive prerequisites."""
    needed: set[str] = set()
    queue = deque(skills)
    visited = set(skills) | known

    while queue:
        s = queue.popleft()
        for prereq in DEPENDENCIES.get(s, []):
            if prereq not in visited:
                visited.add(prereq)
                needed.add(prereq)
                queue.append(prereq)
    return needed


def _topological_sort(skills_set: set[str]) -> list[str]:
    """Kahn's algorithm: skills with no in-plan deps come first."""
    from collections import defaultdict
    in_degree = {s: 0 for s in skills_set}
    adj = defaultdict(list)

    for skill in skills_set:
        for dep in DEPENDENCIES.get(skill, []):
            if dep in skills_set:
                adj[dep].append(skill)
                in_degree[skill] += 1

    queue = deque(s for s, d in in_degree.items() if d == 0)
    ordered = []
    while queue:
        node = queue.popleft()
        ordered.append(node)
        for nb in adj[node]:
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                queue.append(nb)

    remaining = sorted(skills_set - set(ordered))
    ordered.extend(remaining)
    return ordered


def generate_roadmap(missing_skills: list[str], resume_skills: list[str]) -> list[dict]:
    """
    Build an ordered, dependency-aware learning roadmap.

    Returns a list of step dicts:
        { step, skill, type ("required"|"prerequisite"), days, resource, url, depends_on }
    """
    known = set(resume_skills)
    prereqs = _get_prerequisites(missing_skills, known)
    all_to_learn = set(missing_skills) | prereqs
    ordered = _topological_sort(all_to_learn)

    roadmap = []
    for i, skill in enumerate(ordered, 1):
        meta = get_skill_meta(skill)
        skill_type = "required" if skill in set(missing_skills) else "prerequisite"
        deps_in_plan = [d for d in DEPENDENCIES.get(skill, []) if d in all_to_learn]
        roadmap.append({
            "step":       i,
            "skill":      skill,
            "type":       skill_type,
            "days":       meta["days"],
            "resource":   meta["resource"],
            "url":        meta["url"],
            "depends_on": deps_in_plan,
        })
    return roadmap


# ─────────────────────────────────────────────────────────────────────────────
# 6. OVERVIEW & LEVEL
# ─────────────────────────────────────────────────────────────────────────────

def generate_overview(role_hint: str, resume_skills: list[str],
                      jd_skills: list[str], gap: list[str]) -> tuple[str, str]:
    """
    Returns (summary_paragraph, level_string).
    """
    strong = sorted(set(resume_skills) & set(jd_skills))
    role   = role_hint.strip() or "this role"

    if not strong:
        strength_line = "Your resume currently shows limited direct skill overlap with the job requirements."
    elif len(strong) <= 3:
        strength_line = (
            f"You already have foundational skills — {', '.join(strong)} — "
            "that align with the job requirements."
        )
    else:
        strength_line = (
            f"You bring strong matched skills including {', '.join(strong[:5])}"
            + (f" and {len(strong)-5} more" if len(strong) > 5 else "")
            + ", which form a solid base for this role."
        )

    if not gap:
        gap_line = "No critical skill gaps were detected — you are well-positioned for this role."
    elif len(gap) <= 2:
        gap_line = (
            f"The main gap is {', '.join(gap)}. "
            "A focused upskilling effort here should get you job-ready quickly."
        )
    else:
        gap_line = (
            f"Key gaps include {', '.join(gap[:4])}"
            + (f" and {len(gap)-4} more" if len(gap) > 4 else "")
            + ". Bridging these will significantly boost your candidacy."
        )

    total_days = sum(get_skill_meta(s)["days"] for s in gap)
    time_line  = f"Estimated upskilling time: ~{total_days} days of focused study." if gap else ""

    summary = (
        f"Based on your resume and the provided job description for the {role} role, "
        f"here is a personalized assessment. {strength_line} {gap_line} {time_line}"
    ).strip()

    # Level heuristic
    n = len(resume_skills)
    if n >= 10:
        level = "Senior"
    elif n >= 5:
        level = "Mid-level"
    else:
        level = "Junior / Beginner"

    return summary, level


def readiness_breakdown(resume_skills: list[str], jd_skills: list[str]) -> dict:
    """
    Return a per-category readiness breakdown for display.
    Categories: core languages, frameworks, data & ml, devops, tools.
    """
    categories = {
        "Languages":    {"python", "java", "javascript", "typescript", "r", "scala", "go", "rust", "c++"},
        "Frameworks":   {"react", "angular", "vue", "django", "flask", "fastapi", "nodejs", "spring boot"},
        "Data & ML":    {"machine learning", "deep learning", "statistics", "pytorch", "tensorflow",
                         "scikit-learn", "sql", "spark", "nlp", "computer vision", "transformers"},
        "DevOps/Cloud": {"docker", "kubernetes", "aws", "gcp", "azure", "linux", "cicd", "terraform"},
        "Tools":        {"git", "airflow", "mlflow", "bigquery", "snowflake", "elasticsearch"},
    }
    resume_set = set(resume_skills)
    jd_set     = set(jd_skills)
    breakdown  = {}

    for cat, skills_in_cat in categories.items():
        jd_in_cat     = jd_set & skills_in_cat
        resume_in_cat = resume_set & skills_in_cat
        if not jd_in_cat:
            continue
        pct = int(len(resume_in_cat & jd_in_cat) / len(jd_in_cat) * 100)
        breakdown[cat] = {
            "score":   pct,
            "have":    sorted(resume_in_cat & jd_in_cat),
            "missing": sorted(jd_in_cat - resume_in_cat),
        }
    return breakdown
