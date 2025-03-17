import streamlit as st
import nltk
import re
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import base64
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLP models
nltk.download('stopwords')


def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    pdf = PdfReader(file)
    text = "".join(page.extract_text() or "" for page in pdf.pages)
    return text.strip() if text else "No readable text found."


def rank_resumes(job_description, resumes):
    """Ranks resumes based on their similarity to the job description."""
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    job_desc_vector = vectors[0]
    resume_vectors = vectors[1:]
    return cosine_similarity([job_desc_vector], resume_vectors).flatten()


def extract_candidate_details(resume_text):
    """Extracts candidate name, email, and phone number from resume text."""
    lines = resume_text.split("\n")
    name = lines[0].replace("Name:",
                            "").strip() if lines else "Unknown Candidate"
    email = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                      resume_text)
    phone = re.search(r'\b\d{10}\b', resume_text)  # Matches 10-digit numbers
    return name, email.group(0) if email else "N/A", phone.group(
        0) if phone else "N/A"


def get_table_download_link(df):
    """Creates a download link for the ranking results."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="resume_rankings.csv">📥 Download Ranking as CSV</a>'


# Streamlit UI
st.set_page_config(page_title="Smart Resume Analyzer",
                   page_icon="📑",
                   layout="wide")
st.title("📊 AI-Powered Resume Screening & Ranking")

uploaded_files = st.file_uploader("📂 Upload Resumes (PDF Only)",
                                  type=["pdf"],
                                  accept_multiple_files=True)

if uploaded_files:
    st.success(f"✔️ {len(uploaded_files)} resumes uploaded successfully!")

    job_roles = [
        "💻 Software Engineer", "📊 Data Scientist", "📋 Project Manager",
        "🎨 UI/UX Designer", "☁️ DevOps Engineer"
    ]
    job_role = st.selectbox("🏢 Select Job Role", job_roles)

    experience_levels = [
        "🔰 Entry Level (0-2 years)", "⚡ Mid Level (2-5 years)",
        "🏆 Senior Level (5+ years)"
    ]
    experience = st.selectbox("📅 Select Experience Level", experience_levels)

    skills = [
        "🐍 Python", "☕ Java", "🤖 Machine Learning", "🧠 Deep Learning",
        "☁️ Cloud Computing", "🚀 Agile Methodology", "🛢️ SQL", "⚛️ ReactJS",
        "🐳 Docker"
    ]
    required_skills = st.multiselect("🛠️ Select Required Skills", skills)

    if job_role and experience and required_skills:
        job_description = f"We are looking for a {job_role[2:]} with {experience[2:]} of experience. Required skills include: {', '.join([s[2:] for s in required_skills])}."

        resumes = []
        candidate_details = []
        progress_bar = st.progress(0)

        for i, file in enumerate(uploaded_files):
            resume_text = extract_text_from_pdf(file)
            resumes.append(resume_text)
            candidate_details.append(extract_candidate_details(resume_text))
            progress_bar.progress((i + 1) / len(uploaded_files))

        scores = rank_resumes(job_description, resumes)
        ranked_resumes = list(
            zip([d[0] for d in candidate_details],
                [d[1] for d in candidate_details],
                [d[2] for d in candidate_details], scores, resumes))

        st.subheader("🏅 Ranked Candidates")
        ranking_df = pd.DataFrame(ranked_resumes,
                                  columns=[
                                      "Candidate Name", "Email",
                                      "Mobile Number", "Score", "Resume Text"
                                  ])
        st.dataframe(
            ranking_df.drop(columns=["Resume Text"]).sort_values(
                by="Score", ascending=False))
        st.markdown(get_table_download_link(
            ranking_df.drop(columns=["Resume Text"])),
                    unsafe_allow_html=True)

        # Bar Chart Visualization
        st.subheader("📊 Candidate Ranking Chart")
        fig, ax = plt.subplots()
        sorted_candidates = ranking_df.sort_values(by="Score", ascending=True)
        ax.barh(sorted_candidates["Candidate Name"],
                sorted_candidates["Score"],
                color='skyblue')
        ax.set_xlabel("Score")
        ax.set_ylabel("Candidate Name")
        ax.set_title("Resume Ranking Scores")
        st.pyplot(fig)

        # Show Top Resume
        st.subheader("🥇 Top Ranked Candidate")
        top_candidate = ranking_df.sort_values(by="Score",
                                               ascending=False).iloc[0]
        st.markdown(
            f"**{top_candidate['Candidate Name']}** - {top_candidate['Email']} - {top_candidate['Mobile Number']}"
        )
        st.text_area("📜 Top Resume Content",
                     top_candidate['Resume Text'],
                     height=300)
