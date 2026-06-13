# 📄 AI-Powered Resume Screener

An intelligent ATS (Applicant Tracking System) built with Machine Learning and NLP that analyzes resumes against job descriptions, predicts job categories, and provides actionable feedback.

🚀 **Live Demo:** [ai-resume-screener09.streamlit.app](https://ai-resume-screener09.streamlit.app/)

---

## ✨ Features

- **📊 Semantic Match Score** — Uses sentence transformers to compare resume meaning against job description (not just keywords)
- **🎯 Job Category Prediction** — Classifies resume into one of 19 job categories using ML
- **🏆 Top 3 Category Predictions** — Shows the top 3 most likely job categories with confidence scores
- **💪 Strong Points** — Highlights keywords in your resume that match the job description
- **🔧 Missing Keywords** — Shows important keywords from the job description missing in your resume
- **✅ CV Validation** — Automatically detects if the uploaded PDF is a real resume

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| ML Model | Logistic Regression (scikit-learn) |
| Text Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Similarity | Cosine Similarity |
| Web App | Streamlit |
| PDF Extraction | pdfplumber |
| Text Processing | NLTK |

---

## 🧠 How It Works

```
User uploads PDF resume + pastes job description
        ↓
PDF text extracted with pdfplumber
        ↓
CV validation check
        ↓
Text cleaned with NLTK (lowercase, stopwords removed)
        ↓
Sentence Transformer converts text to 384-dimensional embeddings
        ↓
Logistic Regression predicts job category
        ↓
Cosine similarity calculates semantic match score
        ↓
Results displayed: score + strong points + missing keywords
```

---

## 📊 Model Performance

| Model | Accuracy |
|---|---|
| TF-IDF + Logistic Regression | 64.99% |
| TF-IDF + Random Forest | 67.51% |
| TF-IDF + SVC | 65.19% |
| **HuggingFace + Logistic Regression** | **73.73% ✅** |
| HuggingFace + SVC | 73.73% |
| HuggingFace + Random Forest | 70.51% |

---

## 📁 Project Structure

```
resume-screener/
│
├── app.py                    # Streamlit web application
├── AI-CVScreener.ipynb       # Training notebook (Week 1 & 2)
├── resume_model.pkl          # Trained Logistic Regression model
├── embedder.pkl              # Sentence Transformer embedder
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Run Locally

**1. Clone the repository:**
```bash
git clone https://github.com/amenallahabdelhafidh/amenallahabdelhafidh-AI-Powered-Resume-Screener.git
cd amenallahabdelhafidh-AI-Powered-Resume-Screener
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the app:**
```bash
streamlit run app.py
```

---

## 📦 Dataset

Dataset used for training: [Resume Dataset — Kaggle](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)

- 2,484 resumes across 24 job categories
- Source: livecareer.com
- After filtering: 19 categories with 100+ resumes each

> Download the dataset from the link above and place it in the root folder as `Resume.csv`

---

## 👨‍💻 Author

**Amen Allah Abdelhafidh**
- 🎓 3rd Year Engineering Student — ESPRIT, Tunisia
- 💼 [LinkedIn](https://linkedin.com/in/amenallahabdelhafidh)
- 🐙 [GitHub](https://github.com/amenallahabdelhafidh)

---

## 🔮 Future Improvements

- [ ] Add Data Science category with additional dataset
- [ ] CV document classifier (detect non-CV uploads more accurately)
- [ ] Multi-language support (French, Arabic)
- [ ] Resume improvement suggestions using LLMs
- [ ] API endpoint for integration with HR systems

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
