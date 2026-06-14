import streamlit as st
import joblib
import pdfplumber
import re
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('stopwords', quiet=True)

# ── Load model and embedder ────────────────────────────────────────────
cv_classifier = joblib.load('cv_classifier.pkl')
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

#--------------------------------
def is_cv(text):
    cleaned = clean_text(text)
    embedding = embedder.encode([cleaned])
    prediction = cv_classifier.predict(embedding)[0]
    confidence = cv_classifier.predict_proba(embedding).max() * 100
    return prediction == 1, confidence
# ── UI Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main-header {
        background: #26215C;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 {
        color: #CECBF6;
        margin: 0;
        font-size: 2.2rem;
    }
    .main-header p {
        color: #AFA9EC;
        margin-top: 0.5rem;
        font-size: 1rem;
    }
    div[data-testid="stMetric"] {
        background-color: #EEEDFE;
        border: 1px solid #CECBF6;
        border-radius: 10px;
        padding: 1rem;
    }
    .stButton button {
        background-color: #534AB7;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 2rem;
    }
    .stButton button:hover {
        background-color: #3C3489;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────
st.markdown("""
    <div class="main-header">
        <h1>📄 AI-Powered Resume Screener</h1>
        <p>Upload your resume and paste a job description to see how well you match — powered by NLP & Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <style>
        section[data-testid="stSidebar"] {
            background-color: #EEEDFE;
        }
        section[data-testid="stSidebar"] h3 {
            color: #26215C;
        }
        section[data-testid="stSidebar"] p, 
        section[data-testid="stSidebar"] li {
            color: #3C3489;
        }
        section[data-testid="stSidebar"] a {
            color: #534AB7;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("### 🧠 About This Tool")
    st.markdown("""
    This app uses **Sentence Transformers** to understand the meaning 
    of your resume, not just keywords.

    **How it works:**
    1. Upload your resume (PDF)
    2. Paste a job description
    3. Get a match score, strong points & missing keywords
    """)
    st.divider()
    st.markdown("### 📊 Model Info")
    st.markdown("""
    - **Accuracy:** 73.7%
    - **Categories:** 19 job types
    - **Embedding model:** all-MiniLM-L6-v2
    """)
    st.divider()
    st.markdown("Made by **Amen Allah Abdelhafidh**")
    st.markdown("[GitHub](https://github.com/amenallahabdelhafidh) · [LinkedIn](https://linkedin.com)")



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

            is_cv_result, cv_confidence = is_cv(raw_text)

            if not is_cv_result:
                st.error(f"⚠️ This doesn't look like a CV! (Confidence: {round(cv_confidence, 1)}%)")
                st.stop()
            cleaned = clean_text(raw_text)


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