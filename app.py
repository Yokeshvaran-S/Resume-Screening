import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip() if text else "No readable text found."

# Function to rank resumes based on similarity to job role, experience, and required skills
def rank_resumes(job_role, experience_level, required_skills, resumes):
    job_criteria = job_role + " " + experience_level + " " + " ".join(required_skills)
    documents = [job_criteria] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    job_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_vector], resume_vectors).flatten()
    
    return cosine_similarities

# ----- Streamlit UI -----
st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="wide")

# Custom CSS for better UI
st.markdown(
    """
    <style>
        .big-font { font-size: 24px !important; font-weight: bold; }
        .stButton > button { width: 100%; border-radius: 10px; }
        .stFileUploader { border: 2px solid #4CAF50; padding: 10px; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3281/3281323.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📂 Client Uploads", "📊 Interviewer Panel"])

# Client Upload Page
if page == "📂 Client Uploads":
    st.title("📂 Upload Resumes")
    st.markdown("<p class='big-font'>Upload resumes in PDF format</p>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("📂 Upload PDF resumes", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        st.success("✅ Resumes uploaded successfully!")
        st.markdown("**Uploaded Files:**")
        for file in uploaded_files:
            st.markdown(f"- {file.name}")

# Interviewer Panel Page
elif page == "📊 Interviewer Panel":
    st.title("📊 Resume Ranking System")
    st.markdown("<p class='big-font'>Filter Candidates</p>", unsafe_allow_html=True)
    
    job_role = st.selectbox("🎯 Select Job Role", ["Software Engineer", "Data Scientist", "Project Manager", "UX Designer", "Business Analyst"], index=0)
    experience_level = st.selectbox("📅 Select Experience Level", ["0-2 years", "2-5 years", "5+ years"], index=0)
    required_skills = st.multiselect("🔹 Select Required Skills", ["Python", "Machine Learning", "Project Management", "SQL", "UI/UX Design", "Cloud Computing", "Java", "Communication Skills"])
    
    if 'uploaded_files' in locals() and uploaded_files:
        if st.button("🚀 Rank Resumes"):
            st.info("Processing resumes... Please wait ⏳")

            resumes = []
            progress_bar = st.progress(0)

            for i, file in enumerate(uploaded_files):
                resumes.append(extract_text_from_pdf(file))
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            scores = rank_resumes(job_role, experience_level, required_skills, resumes)
            
            ranked_resumes = sorted(zip(uploaded_files, resumes, scores), key=lambda x: x[2], reverse=True)

            st.success("✅ Ranking completed!")
            st.subheader("📌 Ranked Resumes")
            
            # Styled Markdown Table
            table_md = "| Rank | Resume Name | Score |\n|------|--------------|-------|\n"
            
            for i, (file, resume_text, score) in enumerate(ranked_resumes, start=1):
                table_md += f"| {i} | {file.name} | {score:.2f} |\n"
            
            st.markdown(table_md, unsafe_allow_html=True)
            
            # Show top-ranked resume preview
            if ranked_resumes:
                st.subheader("🏆 Top Ranked Resume")
                top_resume_text = extract_text_from_pdf(ranked_resumes[0][0])
                st.text_area("🔹 Top Resume Content", top_resume_text, height=300)
