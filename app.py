from engine import extract_text_from_pdf, detect_skills, compute_skill_gap, generate_overview
import streamlit as st

st.set_page_config(page_title="AI Adaptive Onboarding Engine", layout="wide")


# ---------------- TITLE ----------------
st.title("AI Adaptive Onboarding Engine")

st.markdown("### Transform onboarding into a personalized journey")

st.markdown(
"Upload your Resume and Job Description to generate a custom learning roadmap"
)

st.info(
"📌 Upload Resume (filename should include 'resume' or 'cv') and Job Description (include 'jd' or 'job')"
)

st.divider()

# ---------------- FILE UPLOAD ----------------
col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("📄 Upload Resume", type=["pdf"])

with col2:
    jd_file = st.file_uploader("📑 Upload Job Description", type=["pdf"])
    jd_text = st.text_area("📝 Or Paste Job Description")

st.markdown("### Optional: Target role hint")
st.caption("Example: Data Engineer, ML Engineer, SWE")

target_role = st.text_input("")

st.divider()

# ---------------- ANALYZE ----------------
if st.button("🔍 Analyze Profile"):

    if resume_file is None or (jd_file is None and not jd_text.strip()):
        st.warning("Upload Resume and JD")
        st.stop()

    if resume_file:
        name = resume_file.name.lower()
        if "resume" not in name and "cv" not in name:
            st.error("❌ Wrong file uploaded in Resume section. Filename must contain 'resume' or 'cv'.")
            st.stop()

    if jd_file:
        name = jd_file.name.lower()
        if "jd" not in name and "job" not in name:
            st.error("❌ Wrong file uploaded in Job Description section. Filename must contain 'jd' or 'job'.")
            st.stop()

    resume_text = extract_text_from_pdf(resume_file)

    if jd_file:
        jd_text_content = extract_text_from_pdf(jd_file)
    else:
        jd_text_content = jd_text.lower()

    resume_skills = detect_skills(resume_text)
    jd_skills = detect_skills(jd_text_content)

    missing_skills = compute_skill_gap(resume_skills, jd_skills)

    matched = len(jd_skills) - len(missing_skills)

    if len(jd_skills) > 0:
        match_percentage = int(matched / len(jd_skills) * 100)
    else:
        match_percentage = 0

    # ⭐ OVERVIEW SECTION
    st.success("✅ Analysis Complete!")

    st.markdown("## Overview")

    summary, level = generate_overview(
        target_role,
        resume_skills,
        jd_skills,
        missing_skills
    )

    st.info("Summary")
    st.write(summary)

    role_name = target_role if target_role.strip() != "" else "Inferred Role"
    st.success(f"Inferred role: {role_name} | Level: {level}")

    st.divider()

    # ---------------- DASHBOARD ----------------
    st.markdown("##  AI Insights Dashboard")

    c1, c2, c3 = st.columns(3)

    c1.metric("Match %", f"{match_percentage}%")
    c2.metric("Skills Found", len(resume_skills))
    c3.metric("Missing Skills", len(missing_skills))

    st.progress(match_percentage)

    st.divider()

    # ---------------- SKILLS ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ✅ Existing Skills")
        for skill in resume_skills:
            st.success(skill)

    with col2:
        st.markdown("### ❌ Missing Skills")
        for skill in missing_skills:
            st.error(skill)

    st.divider()

    # ---------------- ROADMAP ----------------
    st.markdown("##  Learning Roadmap")

    for skill in missing_skills:
        st.info(f"Learn *{skill}* → Suggested Time: 5 days")

    st.divider()

    # ---------------- EXPLAINABLE AI ----------------
    st.markdown("##  Explainable AI Decisions")

    for skill in missing_skills:
        with st.expander(skill):
            st.write("This skill exists in JD but not detected in Resume.")

    st.divider()

    # ---------------- RESUME REWRITE ----------------
    st.markdown("##  Resume Rewrite Suggestions")

    resume_lines = [
        line.strip()
        for line in resume_text.split("\n")
        if len(line.strip()) > 40
    ]

    selected_lines = resume_lines[:3] if len(resume_lines) >= 3 else resume_lines

    def improve_line(line):
        return "Enhanced: " + line + " with quantified measurable impact."

    for line in selected_lines:
        with st.expander(line):
            st.write(improve_line(line))

    st.divider()

    # ---------------- SUGGESTIONS ----------------
    st.markdown("##  Smart Suggestions")

    st.success("""
* Build projects covering missing skills  
* Add metrics like % improvement  
* Upload GitHub links  
* Focus on fundamentals before advanced topics  
""")

    st.divider()

    # ---------------- FINAL READINESS ----------------
    st.markdown("##  Job Readiness Estimate")

    if match_percentage > 75:
        st.success(" You are almost job ready!")
    elif match_percentage > 40:
        st.warning("⚡ Moderate preparation needed.")
    else:
        st.error(" Significant upskilling required.")