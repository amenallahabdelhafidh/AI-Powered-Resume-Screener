import streamlit as st
import joblib
import pdfplumber
import re
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('stopwords', quiet=True)

# ── Load model and embedder ────────────────────────────────────────────
model = joblib.load('resume_model.pkl')
embedder = joblib.load('embedder.pkl')

# ── Text cleaning ──────────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# ── Extract text from PDF ──────────────────────────────────────────────
def extract_text_from_pdf(pdf_file):
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
        return text.strip()
    except Exception as e:
        return ""

# ── Match score ────────────────────────────────────────────────────────
def get_match_score(resume_text, job_description):
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_description)
    resume_embedding = embedder.encode([resume_clean])
    job_embedding = embedder.encode([job_clean])
    similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
    return round(float(similarity) * 100, 2)

# ── Strong points ──────────────────────────────────────────────────────
def get_strong_points(resume_text, job_description):
    resume_words = set(clean_text(resume_text).split())
    job_words = set(clean_text(job_description).split())
    strong = resume_words.intersection(job_words)
    strong = [w for w in strong if len(w) > 3]
    return list(strong)[:10]

# ── Missing skills ─────────────────────────────────────────────────────
def get_missing_skills(resume_text, job_description):
    resume_words = set(clean_text(resume_text).split())
    job_words = set(clean_text(job_description).split())
    missing = job_words - resume_words - stop_words
    missing = [w for w in missing if len(w) > 3]
    return list(missing)[:10]
# ── UI Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)
#--------------------------------
def is_cv(text):
    cv_keywords = [
        'experience', 'education', 'skills', 'work',
        'university', 'degree', 'project', 'internship',
        'certification', 'languages', 'summary', 'objective',
        'employment', 'qualification', 'achievement'
    ]
    text_lower = text.lower()
    matches = [kw for kw in cv_keywords if kw in text_lower]
    return len(matches) >= 3

# ── Header ─────────────────────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align: center; color: #1E88E5;'>📄 AI-Powered Resume Screener</h1>
    <p style='text-align: center; color: gray;'>Upload your resume and paste a job description to see how well you match!</p>
""", unsafe_allow_html=True)

st.divider()

# ── Two column layout ──────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📋 Job Description")
    job_description = st.text_area(
        "Paste the job description here:",
        height=300,
        placeholder="e.g. We are looking for a Python developer with experience in machine learning..."
    )

with col_right:
    st.subheader("📤 Upload Your Resume")
    uploaded_file = st.file_uploader("Upload your resume as PDF", type=["pdf"])
    
    if uploaded_file:
        st.success(f"✅ {uploaded_file.name} uploaded successfully!")

st.divider()

# ── Analyze button ─────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze = st.button("🔍 Analyze My Resume", use_container_width=True)

if analyze:
    if not uploaded_file:
        st.error("⚠️ Please upload your resume first!")
    elif not job_description:
        st.error("⚠️ Please paste a job description first!")
    else:
        with st.spinner("🤖 Analyzing your resume..."):

            # Extract and clean
            raw_text = extract_text_from_pdf(uploaded_file)

            if not raw_text:
                st.error("⚠️ Could not read this PDF! Make sure it's a valid PDF file.")
                st.stop()

            if not is_cv(raw_text):
                st.error("⚠️ This doesn't look like a CV! Please upload a proper resume PDF.")
                st.stop()
            cleaned = clean_text(raw_text)


            # Predict
            embedding = embedder.encode([cleaned])
            prediction = model.predict(embedding)[0]

            # Top 3
            probas = model.predict_proba(embedding)[0]
            top3_idx = probas.argsort()[-3:][::-1]
            top3_categories = model.classes_[top3_idx]
            top3_scores = probas[top3_idx] * 100

            # Scores
            score = get_match_score(raw_text, job_description)
            strong = get_strong_points(raw_text, job_description)
            missing = get_missing_skills(raw_text, job_description)

        st.success("✅ Analysis Complete!")
        st.divider()

        # ── Metrics ───────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric("🎯 Predicted Category", prediction)
        m2.metric("📊 Match Score", f"{score}%",
                  delta="Good match!" if score > 60 else "Needs improvement")
        m3.metric("💪 Strong Points", f"{len(strong)} keywords")

        st.divider()

        # ── Top 3 categories ──────────────────────────────────────────
        st.subheader("🏆 Top 3 Predicted Categories")
        for cat, sc in zip(top3_categories, top3_scores):
            st.progress(int(sc), text=f"{cat} → {round(sc, 1)}%")

        st.divider()

        # ── Strong points & Missing skills side by side ───────────────
        left, right = st.columns(2)

        with left:
            st.subheader("💪 Strong Points")
            if strong:
                for point in strong:
                    st.success(f"✅ {point}")
            else:
                st.warning("No strong matches found!")

        with right:
            st.subheader("🔧 Missing Keywords")
            if missing:
                for skill in missing:
                    st.error(f"❌ {skill}")
            else:
                st.success("🎉 Your resume covers everything!")