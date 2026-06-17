import streamlit as st
import joblib
import pdfplumber
import re
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import google.generativeai as genai
nltk.download('stopwords', quiet=True)

# ── Load model and embedder ────────────────────────────────────────────
cv_classifier = joblib.load('cv_classifier.pkl')
model = joblib.load('resume_model.pkl')
embedder = joblib.load('embedder.pkl')
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

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

def get_ai_feedback(resume_text, job_description, missing_skills, score):
    prompt = f"""You are a professional career advisor. Analyze this resume against the job description.

Job Description: {job_description[:1000]}

Resume: {resume_text[:1500]}

Match Score: {score}%
Missing Keywords: {', '.join(missing_skills)}

Give 3 short, specific, actionable tips (2-3 sentences each) to improve this resume for this specific job. Be encouraging but honest. Format as a numbered list."""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not generate AI feedback: {str(e)}"

def extract_text_from_pdf(pdf_file):
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
        return text.strip()
    except Exception:
        return ""

def get_match_score(resume_text, job_description):
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_description)
    resume_embedding = embedder.encode([resume_clean])
    job_embedding = embedder.encode([job_clean])
    similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
    return round(float(similarity) * 100, 2)

def get_strong_points(resume_text, job_description):
    resume_words = set(clean_text(resume_text).split())
    job_words = set(clean_text(job_description).split())
    strong = resume_words.intersection(job_words)
    strong = [w for w in strong if len(w) > 3]
    return list(strong)[:10]

def get_missing_skills(resume_text, job_description):
    resume_words = set(clean_text(resume_text).split())
    job_words = set(clean_text(job_description).split())
    missing = job_words - resume_words - stop_words
    missing = [w for w in missing if len(w) > 3]
    return list(missing)[:10]

def is_cv(text):
    cleaned = clean_text(text)
    embedding = embedder.encode([cleaned])
    prediction = cv_classifier.predict(embedding)[0]
    confidence = cv_classifier.predict_proba(embedding).max() * 100
    return prediction == 1, confidence

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="centered"
)

# ── Global CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #FAFAFC; }

    .block-container { padding-top: 2rem; max-width: 780px; }

    /* Hero */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 2rem;
    }
    .hero-badge {
        display: inline-block;
        background: #EEEDFE;
        color: #534AB7;
        font-size: 13px;
        font-weight: 600;
        padding: 6px 16px;
        border-radius: 20px;
        margin-bottom: 1.2rem;
        letter-spacing: 0.3px;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        color: #1A1A2E;
        margin: 0 0 0.8rem;
        line-height: 1.15;
    }
    .hero h1 span { color: #534AB7; }
    .hero p {
        font-size: 1.05rem;
        color: #6B6B80;
        max-width: 520px;
        margin: 0 auto;
        line-height: 1.6;
    }

    /* Feature strip */
    .feature-strip {
        display: flex;
        justify-content: center;
        gap: 28px;
        margin: 2rem 0 2.5rem;
        flex-wrap: wrap;
    }
    .feature-pill {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13.5px;
        color: #4A4A60;
        font-weight: 500;
    }
    .feature-pill .dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #534AB7;
    }

    /* Upload card */
    .upload-card {
        background: white;
        border: 1px solid #E8E7F3;
        border-radius: 20px;
        padding: 2rem 2rem 1.5rem;
        box-shadow: 0 2px 24px rgba(38, 33, 92, 0.06);
        margin-bottom: 2rem;
    }
    .upload-card-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1A1A2E;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-label {
        font-size: 13px;
        font-weight: 600;
        color: #534AB7;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin: 1.2rem 0 0.5rem;
    }

    /* Buttons */
    .stButton button {
        background: #534AB7;
        color: white;
        font-weight: 700;
        font-size: 15px;
        border-radius: 12px;
        border: none;
        padding: 0.85rem 2rem;
        width: 100%;
        transition: all 0.15s ease;
        box-shadow: 0 4px 14px rgba(83, 74, 183, 0.25);
    }
    .stButton button:hover {
        background: #443C9C;
        box-shadow: 0 6px 18px rgba(83, 74, 183, 0.35);
        transform: translateY(-1px);
    }

    /* Results section */
    .results-header {
        text-align: center;
        margin: 2.5rem 0 1.5rem;
    }
    .results-header h2 {
        font-size: 1.6rem;
        font-weight: 800;
        color: #1A1A2E;
        margin-bottom: 0.3rem;
    }
    .results-header p { color: #6B6B80; font-size: 14px; }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E8E7F3;
        border-radius: 14px;
        padding: 1.1rem 1rem;
    }
    div[data-testid="stMetricLabel"] { font-size: 13px; color: #6B6B80; }

    .result-card {
        background: white;
        border: 1px solid #E8E7F3;
        border-radius: 16px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
    }
    .result-card-title {
        font-size: 15px;
        font-weight: 700;
        color: #1A1A2E;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .tag {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin: 3px 4px 3px 0;
    }
    .tag-good { background: #E7F6EE; color: #1B7A4D; }
    .tag-missing { background: #FDEDED; color: #B23A3A; }

    .stProgress > div > div { background-color: #534AB7; }

    section[data-testid="stSidebar"] { display: none; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Hero section ──────────────────────────────────────────────────────
st.markdown(
    '<div class="hero">'
    '<div class="hero-badge">AI-POWERED · NLP + MACHINE LEARNING</div>'
    '<h1>Know if your resume<br><span>actually matches</span> the job</h1>'
    '<p>Upload your resume and a job description. Get a semantic match score, '
    'your strongest keywords, what\'s missing, and personalized AI feedback — '
    'in seconds.</p>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="feature-strip">'
    '<div class="feature-pill"><span class="dot"></span> Semantic matching, not keyword counting</div>'
    '<div class="feature-pill"><span class="dot"></span> 46 job categories</div>'
    '<div class="feature-pill"><span class="dot"></span> AI career advisor</div>'
    '</div>',
    unsafe_allow_html=True
)

# ── Upload card ───────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown('<div class="upload-card-title">📋 Paste the job description</div>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Job description",
        height=160,
        placeholder="e.g. We are looking for a Python developer with experience in machine learning...",
        label_visibility="collapsed"
    )

    st.markdown('<div class="section-label">Your resume</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload resume",
        type=["pdf"],
        label_visibility="collapsed"
    )
    if uploaded_file:
        st.markdown(
            f'<div style="display:inline-flex; align-items:center; gap:6px; '
            f'background:#E7F6EE; color:#1B7A4D; font-size:13px; font-weight:600; '
            f'padding:6px 14px; border-radius:20px; margin-top:6px;">'
            f'✓ {uploaded_file.name}'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div style="height: 0.5rem"></div>', unsafe_allow_html=True)
    analyze = st.button("🔍 Analyze my resume", use_container_width=True)

# ── Analysis ──────────────────────────────────────────────────────────
if analyze:
    if not uploaded_file:
        st.error("⚠️ Please upload your resume first!")
    elif not job_description:
        st.error("⚠️ Please paste a job description first!")
    else:
        with st.spinner("🤖 Analyzing your resume..."):
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

            probas = model.predict_proba(embedding)[0]
            top3_idx = probas.argsort()[-3:][::-1]
            top3_categories = model.classes_[top3_idx]
            top3_scores = probas[top3_idx] * 100

            score = get_match_score(raw_text, job_description)
            strong = get_strong_points(raw_text, job_description)
            missing = get_missing_skills(raw_text, job_description)
            ai_feedback = get_ai_feedback(raw_text, job_description, missing, score)

        # ── Results header ───────────────────────────────────────────
        st.markdown(
            '<div class="results-header"><h2>Your results</h2>'
            '<p>Here\'s how your resume stacks up against this job</p></div>',
            unsafe_allow_html=True
        )

        m1, m2 = st.columns(2)
        m1.metric("Match score", f"{score}%")
        m2.metric("Strong keywords", len(strong))

        st.markdown(
            f'<div style="background:white; border:1px solid #E8E7F3; border-radius:14px; '
            f'padding:1rem 1.2rem; margin-bottom:1rem; text-align:center;">'
            f'<div style="font-size:13px; color:#6B6B80; margin-bottom:6px;">Predicted role</div>'
            f'<div style="font-size:18px; font-weight:700; color:#534AB7; line-height:1.3; '
            f'word-wrap:break-word;">{prediction}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── Top 3 categories ────────────────────────────────────────
        bars_html = ""
        for cat, sc in zip(top3_categories, top3_scores):
            pct = round(float(sc), 1)
            bars_html += (
                f'<div style="margin-bottom: 12px;">'
                f'<div style="display:flex; justify-content:space-between; font-size:13.5px; margin-bottom:4px;">'
                f'<span style="color:#1A1A2E; font-weight:600;">{cat}</span>'
                f'<span style="color:#534AB7; font-weight:700;">{pct}%</span>'
                f'</div>'
                f'<div style="background:#EFEFF6; border-radius:8px; height:10px; overflow:hidden;">'
                f'<div style="background:#534AB7; width:{pct}%; height:100%; border-radius:8px;"></div>'
                f'</div>'
                f'</div>'
            )
        st.markdown(
            f'<div class="result-card"><div class="result-card-title">🏆 Top 3 predicted categories</div>{bars_html}</div>',
            unsafe_allow_html=True
        )

        # ── Strong points & missing keywords ───────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            if strong:
                tags_html = "".join([f'<span class="tag tag-good">{s}</span>' for s in strong])
            else:
                tags_html = '<span style="color:#9999AA; font-size:13px;">No strong matches found</span>'
            st.markdown(
                f'<div class="result-card"><div class="result-card-title">💪 Strong points</div>{tags_html}</div>',
                unsafe_allow_html=True
            )

        with col_b:
            if missing:
                tags_html = "".join([f'<span class="tag tag-missing">{m}</span>' for m in missing])
            else:
                tags_html = '<span style="color:#9999AA; font-size:13px;">Your resume covers everything!</span>'
            st.markdown(
                f'<div class="result-card"><div class="result-card-title">🔧 Missing keywords</div>{tags_html}</div>',
                unsafe_allow_html=True
            )

        # ── AI feedback ─────────────────────────────────────────────
        st.markdown('<div class="result-card-title">🤖 AI career advisor feedback</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(ai_feedback)

# ── Footer ────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center; padding: 2rem 0 1rem; color: #9999AA; font-size: 13px;">'
    'Built by Amen Allah Abdelhafidh · '
    '<a href="https://github.com/amenallahabdelhafidh" style="color:#534AB7;">GitHub</a>'
    '</div>',
    unsafe_allow_html=True
)