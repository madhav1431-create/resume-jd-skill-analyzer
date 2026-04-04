"""
skill_vocab.py
--------------
Canonical skill vocabulary (200 skills) used across the entire project.
Includes a dependency graph for roadmap generation and keyword aliases
for fuzzy matching during skill extraction.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 1. MASTER SKILL LIST  (index = label id in multi-label head)
# ─────────────────────────────────────────────────────────────────────────────
SKILLS = [
    # ── Programming Languages ──────────────────────────────────────────────
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "r", "scala", "kotlin", "swift", "php", "ruby", "matlab",

    # ── Web / Frontend ─────────────────────────────────────────────────────
    "html", "css", "react", "angular", "vue", "nodejs", "django", "flask",
    "fastapi", "spring boot", "graphql", "rest api", "websockets",

    # ── Data & Databases ───────────────────────────────────────────────────
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "cassandra", "sqlite", "oracle", "nosql", "spark", "hadoop", "kafka",
    "airflow", "dbt", "snowflake", "bigquery", "hive",

    # ── Machine Learning / AI ──────────────────────────────────────────────
    "machine learning", "deep learning", "neural networks", "nlp",
    "computer vision", "reinforcement learning", "transformers",
    "large language models", "llm", "pytorch", "tensorflow", "keras",
    "scikit-learn", "xgboost", "lightgbm", "huggingface", "langchain",
    "rag", "fine-tuning", "model deployment", "mlflow", "wandb",
    "feature engineering", "hyperparameter tuning",

    # ── Statistics & Math ──────────────────────────────────────────────────
    "statistics", "probability", "linear algebra", "calculus",
    "bayesian inference", "hypothesis testing", "time series", "a/b testing",

    # ── Data Science / Analytics ───────────────────────────────────────────
    "data analysis", "data visualization", "pandas", "numpy", "matplotlib",
    "seaborn", "plotly", "tableau", "power bi", "excel", "jupyter",

    # ── DevOps / MLOps / Cloud ─────────────────────────────────────────────
    "docker", "kubernetes", "aws", "azure", "gcp", "linux", "bash",
    "git", "github", "gitlab", "cicd", "terraform", "ansible",
    "prometheus", "grafana", "mlops",

    # ── Software Engineering ───────────────────────────────────────────────
    "data structures", "algorithms", "system design", "microservices",
    "api design", "unit testing", "agile", "scrum", "design patterns",
    "object oriented programming", "functional programming",

    # ── Soft / Domain Skills ───────────────────────────────────────────────
    "communication", "teamwork", "problem solving", "project management",
    "leadership", "research", "technical writing", "product management",
    "finance", "healthcare", "cybersecurity", "blockchain",
]

# Map skill → canonical index (used for vectorisation)
SKILL2IDX = {skill: i for i, skill in enumerate(SKILLS)}
IDX2SKILL = {i: skill for skill, i in SKILL2IDX.items()}
NUM_SKILLS = len(SKILLS)   # 113


# ─────────────────────────────────────────────────────────────────────────────
# 2. KEYWORD ALIASES  (alternate spellings / abbreviations → canonical skill)
# ─────────────────────────────────────────────────────────────────────────────
ALIASES: dict[str, str] = {
    # Python ecosystem
    "py": "python", "sklearn": "scikit-learn", "sk-learn": "scikit-learn",
    "tf": "tensorflow", "tf2": "tensorflow", "hf": "huggingface",
    "np": "numpy", "pd": "pandas",
    # JS
    "js": "javascript", "ts": "typescript", "reactjs": "react",
    "vuejs": "vue", "angularjs": "angular", "next.js": "nodejs",
    "express": "nodejs", "express.js": "nodejs",
    # DBs
    "postgres": "postgresql", "pg": "postgresql", "mongo": "mongodb",
    "elastic": "elasticsearch", "es": "elasticsearch",
    # Cloud
    "amazon web services": "aws", "google cloud": "gcp",
    "google cloud platform": "gcp", "microsoft azure": "azure",
    # ML
    "dl": "deep learning", "ml": "machine learning",
    "cv": "computer vision", "rl": "reinforcement learning",
    "bert": "transformers", "gpt": "large language models",
    "llms": "large language models", "rag pipeline": "rag",
    # DevOps
    "k8s": "kubernetes", "ci/cd": "cicd", "ci cd": "cicd",
    "version control": "git", "github actions": "cicd",
    # Misc
    "oop": "object oriented programming", "fp": "functional programming",
    "dsa": "data structures", "ds": "data structures",
    "tdd": "unit testing", "bdd": "unit testing",
    "viz": "data visualization", "dataviz": "data visualization",
    "pbi": "power bi",
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. DEPENDENCY GRAPH  (prerequisite → list of dependent skills)
# ─────────────────────────────────────────────────────────────────────────────
# Format: { "skill": ["depends-on-1", "depends-on-2", ...] }
# "skill X depends on Y" means Y must be learned before X.
DEPENDENCIES: dict[str, list[str]] = {
    # ML stack
    "machine learning":          ["python", "statistics", "linear algebra", "numpy", "pandas"],
    "deep learning":              ["machine learning", "linear algebra", "pytorch"],
    "neural networks":            ["deep learning", "linear algebra"],
    "nlp":                        ["deep learning", "python"],
    "computer vision":            ["deep learning", "python", "numpy"],
    "reinforcement learning":     ["deep learning", "probability", "statistics"],
    "transformers":               ["deep learning", "nlp"],
    "large language models":      ["transformers", "nlp"],
    "llm":                        ["transformers", "nlp"],
    "rag":                        ["large language models", "elasticsearch"],
    "fine-tuning":                ["large language models", "pytorch"],
    "huggingface":                ["transformers", "python"],
    "langchain":                  ["python", "large language models"],
    "mlflow":                     ["machine learning", "python"],
    "wandb":                      ["machine learning", "python"],
    "model deployment":           ["machine learning", "docker", "fastapi"],
    "mlops":                      ["model deployment", "docker", "kubernetes", "mlflow"],
    "feature engineering":        ["pandas", "statistics", "machine learning"],
    "hyperparameter tuning":      ["machine learning", "scikit-learn"],
    "scikit-learn":               ["python", "numpy", "pandas"],
    "xgboost":                    ["machine learning", "scikit-learn"],
    "lightgbm":                   ["machine learning", "scikit-learn"],
    "pytorch":                    ["python", "linear algebra", "numpy"],
    "tensorflow":                 ["python", "linear algebra", "numpy"],
    "keras":                      ["tensorflow"],

    # Data
    "pandas":                     ["python"],
    "numpy":                      ["python"],
    "matplotlib":                 ["python", "numpy"],
    "seaborn":                    ["matplotlib"],
    "plotly":                     ["python"],
    "jupyter":                    ["python"],
    "data analysis":              ["python", "pandas", "statistics"],
    "data visualization":         ["matplotlib", "seaborn"],
    "time series":                ["statistics", "pandas"],
    "a/b testing":                ["statistics", "hypothesis testing"],
    "bayesian inference":         ["probability", "statistics"],

    # SQL / DBs
    "sql":                        ["data structures"],
    "postgresql":                 ["sql"],
    "mysql":                      ["sql"],
    "mongodb":                    ["nosql", "python"],
    "redis":                      ["nosql"],
    "elasticsearch":              ["nosql", "python"],
    "bigquery":                   ["sql", "gcp"],
    "snowflake":                  ["sql"],
    "spark":                      ["python", "sql", "hadoop"],
    "kafka":                      ["linux", "python"],
    "airflow":                    ["python", "docker"],
    "dbt":                        ["sql", "python"],
    "hive":                       ["sql", "hadoop"],

    # Web
    "react":                      ["javascript", "html", "css"],
    "angular":                    ["typescript", "html", "css"],
    "vue":                        ["javascript", "html", "css"],
    "typescript":                 ["javascript"],
    "nodejs":                     ["javascript"],
    "graphql":                    ["rest api", "nodejs"],
    "django":                     ["python", "sql"],
    "flask":                      ["python"],
    "fastapi":                    ["python"],
    "spring boot":                ["java"],
    "websockets":                 ["nodejs", "rest api"],

    # DevOps / Cloud
    "docker":                     ["linux", "bash"],
    "kubernetes":                 ["docker"],
    "cicd":                       ["git", "docker"],
    "terraform":                  ["linux", "aws"],
    "ansible":                    ["linux", "bash"],
    "aws":                        ["linux"],
    "gcp":                        ["linux"],
    "azure":                      ["linux"],
    "prometheus":                 ["docker", "linux"],
    "grafana":                    ["prometheus"],
    "mlops":                      ["docker", "kubernetes", "mlflow"],
    "bash":                       ["linux"],
    "git":                        [],

    # SWE
    "algorithms":                 ["data structures", "python"],
    "system design":              ["algorithms", "sql", "nosql"],
    "microservices":              ["docker", "rest api"],
    "api design":                 ["rest api", "python"],
    "design patterns":            ["object oriented programming"],
    "object oriented programming": ["python"],
    "functional programming":     ["python"],
    "unit testing":               ["python"],
    "agile":                      [],
    "scrum":                      ["agile"],

    # Base / standalone
    "python":                     [],
    "java":                       [],
    "javascript":                 [],
    "html":                       [],
    "css":                        [],
    "linux":                      [],
    "statistics":                 ["probability"],
    "probability":                [],
    "linear algebra":             [],
    "calculus":                   [],
    "data structures":            [],
    "rest api":                   [],
    "nosql":                      [],
    "hadoop":                     ["linux"],
    "communication":              [],
    "teamwork":                   [],
    "problem solving":            [],
    "research":                   [],
    "leadership":                 [],
    "project management":         [],
    "technical writing":          [],
    "product management":         [],
}


# ─────────────────────────────────────────────────────────────────────────────
# 4. SKILL LEVELS  (1 = Foundation, 2 = Intermediate, 3 = Advanced)
# ─────────────────────────────────────────────────────────────────────────────
# Used by infer_dependencies() for automatic prerequisite inference.
# A level‑3 skill depends on related level‑1/2 skills in the same domain.

SKILL_LEVELS: dict[str, int] = {
    # ── Level 1 — Foundations ────────────────────────────────────────────
    "python": 1, "java": 1, "javascript": 1, "html": 1, "css": 1,
    "c++": 1, "c#": 1, "go": 1, "rust": 1, "r": 1, "scala": 1,
    "kotlin": 1, "swift": 1, "php": 1, "ruby": 1, "matlab": 1,
    "sql": 1, "linux": 1, "bash": 1, "git": 1,
    "probability": 1, "linear algebra": 1, "calculus": 1,
    "data structures": 1, "rest api": 1, "nosql": 1,
    "agile": 1, "communication": 1, "teamwork": 1, "problem solving": 1,
    "research": 1, "leadership": 1, "project management": 1,
    "technical writing": 1, "product management": 1,
    "excel": 1, "jupyter": 1,
    # ── Level 2 — Intermediate / Core ────────────────────────────────────
    "statistics": 2, "hypothesis testing": 2,
    "numpy": 2, "pandas": 2, "matplotlib": 2, "seaborn": 2, "plotly": 2,
    "data analysis": 2, "data visualization": 2,
    "typescript": 2, "react": 2, "angular": 2, "vue": 2, "nodejs": 2,
    "django": 2, "flask": 2, "fastapi": 2, "spring boot": 2,
    "graphql": 2, "websockets": 2,
    "postgresql": 2, "mysql": 2, "mongodb": 2, "redis": 2, "sqlite": 2,
    "oracle": 2, "hadoop": 2,
    "docker": 2, "aws": 2, "gcp": 2, "azure": 2,
    "github": 2, "gitlab": 2, "cicd": 2,
    "scikit-learn": 2, "machine learning": 2, "pytorch": 2, "tensorflow": 2,
    "keras": 2,
    "algorithms": 2, "object oriented programming": 2,
    "functional programming": 2, "unit testing": 2, "scrum": 2,
    "api design": 2, "design patterns": 2,
    "tableau": 2, "power bi": 2,
    "finance": 2, "healthcare": 2,
    # ── Level 3 — Advanced ───────────────────────────────────────────────
    "deep learning": 3, "neural networks": 3, "nlp": 3,
    "computer vision": 3, "reinforcement learning": 3,
    "transformers": 3, "large language models": 3, "llm": 3,
    "huggingface": 3, "langchain": 3, "rag": 3, "fine-tuning": 3,
    "xgboost": 3, "lightgbm": 3,
    "feature engineering": 3, "hyperparameter tuning": 3,
    "model deployment": 3, "mlflow": 3, "wandb": 3, "mlops": 3,
    "bayesian inference": 3, "time series": 3, "a/b testing": 3,
    "spark": 3, "kafka": 3, "airflow": 3, "dbt": 3,
    "snowflake": 3, "bigquery": 3, "hive": 3, "elasticsearch": 3,
    "cassandra": 3,
    "kubernetes": 3, "terraform": 3, "ansible": 3,
    "prometheus": 3, "grafana": 3,
    "system design": 3, "microservices": 3,
    "cybersecurity": 3, "blockchain": 3,
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. SKILL DOMAINS  (knowledge‑area grouping for dependency inference)
# ─────────────────────────────────────────────────────────────────────────────
# A skill can belong to MULTIPLE domains.  infer_dependencies() uses this:
# "skill X depends on every lower‑level skill in the same domain(s)."

SKILL_DOMAINS: dict[str, list[str]] = {
    "ml": [
        "python", "numpy", "pandas", "statistics", "probability",
        "linear algebra", "calculus",
        "machine learning", "scikit-learn", "pytorch", "tensorflow", "keras",
        "deep learning", "neural networks", "nlp", "computer vision",
        "reinforcement learning", "transformers", "large language models",
        "llm", "huggingface", "langchain", "rag", "fine-tuning",
        "xgboost", "lightgbm", "feature engineering", "hyperparameter tuning",
        "model deployment", "mlflow", "wandb", "mlops",
    ],
    "data": [
        "python", "sql", "data structures",
        "pandas", "numpy", "data analysis", "data visualization",
        "matplotlib", "seaborn", "plotly", "tableau", "power bi",
        "postgresql", "mysql", "mongodb", "redis", "sqlite", "oracle",
        "nosql", "hadoop",
        "spark", "kafka", "airflow", "dbt", "snowflake", "bigquery",
        "hive", "elasticsearch", "cassandra",
        "statistics", "time series", "bayesian inference",
        "hypothesis testing", "a/b testing",
    ],
    "web": [
        "html", "css", "javascript", "typescript",
        "react", "angular", "vue", "nodejs",
        "python", "django", "flask", "fastapi",
        "java", "spring boot",
        "rest api", "graphql", "websockets",
        "api design",
    ],
    "devops": [
        "linux", "bash", "git", "github", "gitlab",
        "docker", "kubernetes",
        "aws", "gcp", "azure",
        "cicd", "terraform", "ansible",
        "prometheus", "grafana",
        "python",
    ],
    "math": [
        "probability", "linear algebra", "calculus",
        "statistics", "hypothesis testing",
        "bayesian inference", "time series", "a/b testing",
    ],
    "swe": [
        "python", "java", "data structures", "algorithms",
        "object oriented programming", "functional programming",
        "design patterns", "system design", "microservices",
        "unit testing", "api design", "rest api",
        "agile", "scrum",
    ],
    "general": [
        "communication", "teamwork", "problem solving",
        "project management", "leadership", "research",
        "technical writing", "product management",
    ],
}

# Build reverse lookup: skill → set of domains it belongs to
SKILL_TO_DOMAINS: dict[str, set[str]] = {}
for _domain, _skills in SKILL_DOMAINS.items():
    for _s in _skills:
        SKILL_TO_DOMAINS.setdefault(_s, set()).add(_domain)

