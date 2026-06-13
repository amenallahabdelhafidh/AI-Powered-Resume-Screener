import streamlit as st
import joblib
import pdfplumber
import re
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
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
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text.strip()

# ── Match score ────────────────────────────────────────────────────────
def get_match_score(resume_text, job_description):
    # Clean both texts
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_description)
    
    # Use sentence transformer for semantic similarity
    resume_embedding = embedder.encode([resume_clean])
    job_embedding = embedder.encode([job_clean])
    
    # Calculate cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
    score = round(float(similarity) * 100, 2)
    
    return score

# ── Missing skills ─────────────────────────────────────────────────────
def get_missing_skills(resume_text, job_description):
    resume_words = set(resume_text.lower().split())
    job_words = set(job_description.lower().split())
    stop_words_set = set(stopwords.words('english'))
    missing = job_words - resume_words - stop_words_set
    missing = [w for w in missing if len(w) > 3]
    return list(missing)[:10]
def get_strong_points(resume_text, job_description):
    resume_words = set(clean_text(resume_text).split())
    job_words = set(clean_text(job_description).split())
    stop_words_set = set(stopwords.words('english'))
    
    # Words that appear in BOTH resume and job description
    strong = resume_words.intersection(job_words)
    strong = [w for w in strong if w not in stop_words_set]
    strong = [w for w in strong if len(w) > 3]
    
    return list(strong)[:10]

# ── App UI ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Resume Screener", page_icon="📄")

st.title("📄 AI-Powered Resume Screener")
st.markdown("Upload your resume and paste a job description to see how well you match!")

st.divider()

# Job description input
st.subheader("📋 Job Description")
job_description = st.text_area(
    "Paste the job description here:",
    height=200,
    placeholder="e.g. We are looking for a Python developer with experience in machine learning..."
)

st.divider()

# Resume upload
st.subheader("📤 Upload Your Resume")
uploaded_file = st.file_uploader("Upload your resume as PDF", type=["pdf"])

st.divider()

# Analyze button
if st.button("🔍 Analyze Resume", use_container_width=True):
    if not uploaded_file:
        st.error("Please upload your resume first!")
    elif not job_description:
        st.error("Please paste a job description first!")
    else:
        with st.spinner("Analyzing your resume..."):

            # Extract and clean text
            raw_text = extract_text_from_pdf(uploaded_file)
            cleaned = clean_text(raw_text)

            # Embed and predict
            embedding = embedder.encode([cleaned])
            prediction = model.predict(embedding)[0]

            # Top 3 predictions
            probas = model.predict_proba(embedding)[0]
            top3_idx = probas.argsort()[-3:][::-1]
            top3_categories = model.classes_[top3_idx]
            top3_scores = probas[top3_idx] * 100

            # Match score
            score = get_match_score(raw_text, job_description)

            # Missing skills
            missing = get_missing_skills(raw_text, job_description)

        # Results
        st.success("Analysis complete!")
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.metric("🎯 Predicted Category", prediction)

        with col2:
            st.metric("📊 Job Match Score", f"{score}%")

        st.divider()

        # Top 3 categories
        st.subheader("🏆 Top 3 Predicted Categories")
        for cat, sc in zip(top3_categories, top3_scores):
            st.progress(int(sc), text=f"{cat} → {round(sc, 1)}%")

        st.divider()

        # Strong points
    st.subheader("💪 Strong Points in Your Resume")
    strong_points = get_strong_points(raw_text, job_description)
    if strong_points:
        cols = st.columns(5)
    for i, point in enumerate(strong_points):
        cols[i % 5].success(point)
    

    st.divider()

        # Missing skills
    st.subheader("🔧 Keywords Missing from Your Resume")
    if missing:
            cols = st.columns(5)
            for i, skill in enumerate(missing):
                cols[i % 5].warning(skill)
    else:
            st.success("Great! Your resume covers all keywords from the job description!")