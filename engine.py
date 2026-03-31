
import pdfplumber

skills = [
    "python",
    "sql",
    "machine learning",
    "deep learning",
    "statistics",
    "data structures",
    "react",
    "javascript",
    "aws",
    "docker",
    "linux",
    "java",
    "git",
    "html",
    "css",
    "unix"
]

skill_graph = {
    "deep learning": ["machine learning"],
    "machine learning": ["python", "statistics"],
    "docker": ["linux"]
}


def extract_text_from_pdf(file_path):

    text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    return text.lower()


def detect_skills(text):

    detected = []

    for skill in skills:
        if skill in text:
            detected.append(skill)

    return detected


def compute_skill_gap(resume_skills, jd_skills):

    gap = []

    for skill in jd_skills:
        if skill not in resume_skills:
            gap.append(skill)

    return gap


# ⭐ NEW FUNCTION
def generate_overview(role_hint, resume_skills, jd_skills, gap):

    strong = list(set(resume_skills).intersection(jd_skills))
    missing = gap

    if role_hint.strip() == "":
        role_hint = "this role"

    if len(strong) == 0:
        strength_line = "Your resume currently shows limited alignment with the job requirements."
    else:
        strength_line = (
            f"You already possess key skills such as {', '.join(strong[:6])} "
            "which align well with the job requirements."
        )

    if len(missing) == 0:
        gap_line = "There are no major skill gaps detected for this role."
    else:
        gap_line = (
            f"However, there are important gaps including {', '.join(missing[:5])}. "
            "Bridging these gaps will significantly improve your chances of being job-ready."
        )

    summary = (
        f"Based on your resume and the provided job description, you appear to be "
        f"a suitable candidate for the {role_hint} role. "
        f"{strength_line} {gap_line} "
        "Overall, your profile shows solid potential with clear areas for improvement."
    )

    # simple level inference
    if len(resume_skills) > 8:
        level = "Senior"
    elif len(resume_skills) > 4:
        level = "Mid"
    else:
        level = "Beginner"

    return summary, level