import streamlit as st
import numpy as np
import pandas as pd
import re
import pickle
import os
import io
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

# ── PDF support (optional) ────────────────────────────────────────────────────
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2.5rem 2rem; border-radius: 16px;
        margin-bottom: 2rem; text-align: center; color: white;
    }
    .main-header h1 {
        font-family: 'Syne', sans-serif; font-size: 2.8rem;
        font-weight: 800; letter-spacing: -1px; margin: 0;
    }
    .main-header p { font-size: 1rem; color: #b0b8d8; margin-top: 0.5rem; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #2a2a4a; border-radius: 12px;
        padding: 1.2rem 1.5rem; text-align: center; color: white; margin-bottom: 1rem;
    }
    .metric-card .metric-value {
        font-family: 'Syne', sans-serif; font-size: 2rem;
        font-weight: 700; color: #7c83fd;
    }
    .metric-card .metric-label { font-size: 0.85rem; color: #8892b0; margin-top: 0.2rem; }

    .prediction-box {
        background: linear-gradient(135deg, #0d1b2a, #1b2838);
        border: 2px solid #7c83fd; border-radius: 16px;
        padding: 2rem; text-align: center; color: white; margin-top: 1.5rem;
    }
    .prediction-box .pred-label {
        font-size: 0.9rem; color: #8892b0;
        text-transform: uppercase; letter-spacing: 2px;
    }
    .prediction-box .pred-value {
        font-family: 'Syne', sans-serif; font-size: 2.2rem;
        font-weight: 800; color: #7c83fd; margin-top: 0.3rem;
    }
    .prediction-box .pred-id { font-size: 0.8rem; color: #5a637a; margin-top: 0.3rem; }

    .section-title {
        font-family: 'Syne', sans-serif; font-size: 1.4rem;
        font-weight: 700; color: #302b63;
        border-left: 4px solid #7c83fd;
        padding-left: 0.8rem; margin: 1.5rem 0 1rem 0;
    }

    .rank-card {
        border-radius: 14px; padding: 1.2rem 1.5rem;
        margin-bottom: 0.9rem; border: 1.5px solid #e2e5f7;
        background: #ffffff; box-shadow: 0 2px 12px rgba(48,43,99,0.07);
    }
    .rank-filename {
        font-family: 'Syne', sans-serif; font-size: 1rem;
        font-weight: 700; color: #302b63;
    }
    .rank-category {
        display: inline-block; background: #e8eaff; color: #302b63;
        border-radius: 20px; padding: 0.15rem 0.8rem;
        font-size: 0.78rem; font-weight: 600; margin-left: 0.5rem;
    }
    .rank-score-bar-bg {
        background: #eef0ff; border-radius: 8px;
        height: 10px; width: 100%; margin-top: 0.5rem;
    }
    .rank-score-bar-fill {
        background: linear-gradient(90deg, #7c83fd, #302b63);
        height: 10px; border-radius: 8px;
    }
    .rank-score-text {
        font-size: 0.82rem; color: #7c83fd;
        font-weight: 700; margin-top: 0.25rem;
    }

    .stTextArea textarea {
        border-radius: 10px; border: 1.5px solid #c9cfe8;
        font-family: 'DM Sans', sans-serif; font-size: 0.95rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #302b63, #7c83fd);
        color: white; border: none; border-radius: 10px;
        padding: 0.6rem 2rem; font-family: 'Syne', sans-serif;
        font-weight: 600; font-size: 1rem; width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }

    .sidebar-section {
        background: #f0f2ff; border-radius: 10px;
        padding: 1rem; margin-bottom: 1rem;
    }
    .info-chip {
        display: inline-block; background: #e8eaff; color: #302b63;
        border-radius: 20px; padding: 0.2rem 0.8rem;
        font-size: 0.8rem; font-weight: 600; margin: 0.2rem;
    }
    div[data-testid="stTabs"] button {
        font-family: 'Syne', sans-serif; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# NLTK Downloads
# ─────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)

download_nltk()

# ─────────────────────────────────────────────
# Category Mapping
# ─────────────────────────────────────────────
CATEGORY_MAPPING = {
    15: "Java Developer",    23: "Testing",
    8:  "DevOps Engineer",   20: "Python Developer",
    24: "Web Designing",     12: "HR",
    13: "Hadoop",             3: "Blockchain",
    10: "ETL Developer",     18: "Operations Manager",
    6:  "Data Science",      22: "Sales",
    16: "Mechanical Engineer", 1: "Arts",
    7:  "Database",          11: "Electrical Engineering",
    14: "Health and Fitness", 19: "PMO",
    4:  "Business Analyst",   9: "DotNet Developer",
    2:  "Automation Testing", 17: "Network Security Engineer",
    21: "SAP Developer",      5: "Civil Engineer",
    0:  "Advocate",
}

# ─────────────────────────────────────────────
# Text Preprocessing
# ─────────────────────────────────────────────
def resumeKeywords(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText).strip()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(cleanText)
    words = [w.lower() for w in tokens]
    stop_words = nltk.corpus.stopwords.words('english')
    words_new = [w for w in words if w not in stop_words]
    wn = WordNetLemmatizer()
    lemm_text = [wn.lemmatize(w) for w in words_new]
    return ' '.join(lemm_text)

# ─────────────────────────────────────────────
# Extract text from uploaded file
# ─────────────────────────────────────────────
def extract_text(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith('.txt'):
        return uploaded_file.read().decode('utf-8', errors='ignore')
    elif name.endswith('.pdf'):
        if PDF_SUPPORT:
            try:
                with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                    return '\n'.join(p.extract_text() or '' for p in pdf.pages)
            except Exception:
                return None
        return None
    return None

# ─────────────────────────────────────────────
# Load Dataset
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    if os.path.exists('ResumeDataSet.csv'):
        df = pd.read_csv('ResumeDataSet.csv')
        df['Processed_Resume'] = df['Resume'].apply(resumeKeywords)
        label = LabelEncoder()
        df['Encoded_Category'] = label.fit_transform(df['Category'])
        return df
    return None

# ─────────────────────────────────────────────
# Train & Cache Models
# ─────────────────────────────────────────────
@st.cache_resource
def train_models(_df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf.fit(_df['Processed_Resume'])
    X = tfidf.transform(_df['Processed_Resume'])
    y = _df['Encoded_Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    y_pred_mnb = mnb.predict(X_test)
    acc_mnb = accuracy_score(y_test, y_pred_mnb) * 100
    report_mnb = classification_report(y_test, y_pred_mnb, output_dict=True)

    knc = OneVsRestClassifier(KNeighborsClassifier())
    knc.fit(X_train, y_train)
    y_pred_knc = knc.predict(X_test)
    acc_knc = accuracy_score(y_test, y_pred_knc) * 100
    report_knc = classification_report(y_test, y_pred_knc, output_dict=True)

    return tfidf, mnb, knc, acc_mnb, acc_knc, report_mnb, report_knc

# ─────────────────────────────────────────────
# Load Pre-saved Models
# ─────────────────────────────────────────────
def load_saved_models():
    models = {}
    for name, fname in [('mnb', 'mnb.pkl'), ('knc', 'knc.pkl'), ('tfidf', 'tfidf.pkl')]:
        if os.path.exists(fname):
            models[name] = pickle.load(open(fname, 'rb'))
    return models

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <span style="font-size:2.5rem;">📄</span>
        <h2 style="font-family:'Syne',sans-serif; font-weight:800; margin:0.3rem 0; color:#302b63;">
            ResumeScreen
        </h2>
        <p style="font-size:0.8rem; color:#888;">AI-Powered Resume Classifier</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Data Analysis", "🤖 Train Models",
         "🔍 Predict Resume", "📁 Batch Predict", "🏆 Rank Resumes"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        '<div class="sidebar-section">'
        '<p style="font-size:0.8rem;font-weight:600;color:#302b63;margin:0 0 0.5rem 0;">📋 Supported Categories</p>'
        + "".join([f'<span class="info-chip">{v}</span>' for v in list(CATEGORY_MAPPING.values())[:8]])
        + '<span class="info-chip">+17 more</span></div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📄 Resume Screening System</h1>
    <p>Classify resumes into job categories using Machine Learning · Built with NLP + TF-IDF</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-value">25</div><div class="metric-label">Job Categories</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-value">3</div><div class="metric-label">ML Models</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">NLP</div><div class="metric-label">TF-IDF + Lemmatization</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
    steps = {
        "1️⃣ Upload Dataset": "Load `ResumeDataSet.csv` containing resumes and their job categories.",
        "2️⃣ Preprocessing": "Clean text — remove URLs, hashtags, stopwords, then lemmatize.",
        "3️⃣ Feature Extraction": "Transform cleaned text using TF-IDF Vectorizer.",
        "4️⃣ Model Training": "Train Naive Bayes (MNB) and K-Nearest Neighbors (KNN) classifiers.",
        "5️⃣ Prediction": "Paste any resume text to instantly predict its job category.",
        "6️⃣ Resume Ranking": "Upload multiple resumes + job description → get ranked results by match score.",
    }
    for title, desc in steps.items():
        with st.expander(title):
            st.write(desc)

    st.markdown('<div class="section-title">Quick Pipeline Overview</div>', unsafe_allow_html=True)
    pipeline_steps = ["Raw Resume", "Clean Text", "TF-IDF Vectors", "ML Model", "Job Category"]
    cols = st.columns(len(pipeline_steps))
    for i, (col, step) in enumerate(zip(cols, pipeline_steps)):
        with col:
            st.markdown(f"""
            <div style="background:#f0f2ff; border-radius:10px; padding:0.8rem; text-align:center;
                        border: 2px solid {'#7c83fd' if i == len(pipeline_steps)-1 else '#c9cfe8'};">
                <b style="font-size:0.85rem; color:#302b63;">{step}</b>
            </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE: DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Data Analysis":
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    df = load_data()
    if df is None:
        st.warning("⚠️ `ResumeDataSet.csv` not found. Please place it in the same folder as `app.py`.")
        st.info("📥 Expected columns: `Resume`, `Category`")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Resumes", len(df))
    c2.metric("Unique Categories", df['Category'].nunique())
    c3.metric("Null Values", int(df.isnull().sum().sum()))

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Raw Data", "📊 Bar Chart", "🥧 Pie Chart", "☁️ Word Cloud"])
    with tab1:
        st.dataframe(df[['Category', 'Resume']].head(20), width="stretch")
    with tab2:
        plot_data = pd.DataFrame({'Category': df['Category'].value_counts().index, 'Count': df['Category'].value_counts().values})
        fig = px.bar(plot_data, x='Category', y='Count', color='Category', color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(title='Count of Resumes per Job Category', xaxis_title='Job Type', yaxis_title='Count',
                          xaxis=dict(tickangle=-30), font=dict(family='DM Sans', size=12), plot_bgcolor='white', showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with tab3:
        plot_data = pd.DataFrame({'Category': df['Category'].value_counts().index, 'Count': df['Category'].value_counts().values})
        fig = px.pie(plot_data, values='Count', names='Category', title='Distribution of Job Categories',
                     color_discrete_sequence=px.colors.sequential.RdBu, hole=0.4, opacity=0.85)
        fig.update_layout(font=dict(family='DM Sans', size=13))
        st.plotly_chart(fig, width="stretch")
    with tab4:
        with st.spinner("Generating Word Cloud..."):
            text = ' '.join(df['Processed_Resume'])
            wordcloud = WordCloud(background_color='white', width=900, height=500, max_words=120, colormap='plasma').generate(text)
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════
# PAGE: TRAIN MODELS
# ═══════════════════════════════════════════════════════════════
elif page == "🤖 Train Models":
    st.markdown('<div class="section-title">Train ML Models</div>', unsafe_allow_html=True)
    df = load_data()
    if df is None:
        st.warning("⚠️ `ResumeDataSet.csv` not found.")
        st.stop()

    if st.button("🚀 Train Both Models (MNB + KNN)"):
        with st.spinner("Training models... This may take a moment."):
            tfidf, mnb, knc, acc_mnb, acc_knc, report_mnb, report_knc = train_models(df)
            pickle.dump(mnb, open('mnb.pkl', 'wb'))
            pickle.dump(knc, open('knc.pkl', 'wb'))
            pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
        st.success("✅ Models trained and saved!")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{acc_mnb:.1f}%</div><div class="metric-label">Multinomial Naive Bayes Accuracy</div></div>', unsafe_allow_html=True)
            st.markdown("**Classification Report — MNB**")
            st.dataframe(pd.DataFrame(report_mnb).transpose().style.format("{:.2f}"), width="stretch")
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{acc_knc:.1f}%</div><div class="metric-label">KNN (OneVsRest) Accuracy</div></div>', unsafe_allow_html=True)
            st.markdown("**Classification Report — KNN**")
            st.dataframe(pd.DataFrame(report_knc).transpose().style.format("{:.2f}"), width="stretch")

        fig = go.Figure(go.Bar(
            x=["Multinomial NB", "KNN (OvR)"], y=[acc_mnb, acc_knc],
            marker_color=["#7c83fd", "#302b63"],
            text=[f"{acc_mnb:.1f}%", f"{acc_knc:.1f}%"], textposition='outside'
        ))
        fig.update_layout(title="Model Accuracy Comparison", yaxis=dict(range=[0, 105], title="Accuracy (%)"),
                          plot_bgcolor='white', font=dict(family='DM Sans'))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("👆 Click the button above to train the models.")
        saved = load_saved_models()
        if saved:
            st.success(f"✅ Pre-saved models found: **{', '.join(saved.keys())}** — ready for prediction!")

# ═══════════════════════════════════════════════════════════════
# PAGE: PREDICT RESUME
# ═══════════════════════════════════════════════════════════════
elif page == "🔍 Predict Resume":
    st.markdown('<div class="section-title">Predict Job Category</div>', unsafe_allow_html=True)
    saved = load_saved_models()
    if 'tfidf' not in saved or ('mnb' not in saved and 'knc' not in saved):
        st.warning("⚠️ Trained models not found. Please go to **🤖 Train Models** tab first.")
        st.stop()

    model_choice = st.selectbox(
        "Choose a Model",
        [m for m in ["Multinomial Naive Bayes (MNB)", "K-Nearest Neighbors (KNN)"]
         if ('mnb' in saved and 'MNB' in m) or ('knc' in saved and 'KNN' in m)]
    )
    resume_text = st.text_area("📝 Paste Resume Text Here", height=280,
                               placeholder="Paste the full resume text here — skills, experience, education, etc.")

    if st.button("🔍 Predict Category"):
        if not resume_text.strip():
            st.error("Please paste some resume text first.")
        else:
            with st.spinner("Analyzing resume..."):
                cleaned = resumeKeywords(resume_text)
                features = saved['tfidf'].transform([cleaned])
                if "MNB" in model_choice:
                    pred_id = saved['mnb'].predict(features)[0]
                    model_name = "Multinomial Naive Bayes"
                    proba = saved['mnb'].predict_proba(features)[0]
                else:
                    pred_id = saved['knc'].predict(features)[0]
                    model_name = "KNN (OneVsRest)"
                    proba = None
                category = CATEGORY_MAPPING.get(int(pred_id), "Unknown")

            st.markdown(f"""
            <div class="prediction-box">
                <div class="pred-label">Predicted Job Category</div>
                <div class="pred-value">🎯 {category}</div>
                <div class="pred-id">Category ID: {pred_id} · Model: {model_name}</div>
            </div>""", unsafe_allow_html=True)

            if proba is not None:
                st.markdown('<div class="section-title">Top 5 Category Probabilities</div>', unsafe_allow_html=True)
                top5_idx = np.argsort(proba)[::-1][:5]
                top5_cats = [CATEGORY_MAPPING.get(i, f"Cat {i}") for i in top5_idx]
                top5_probs = [proba[i] * 100 for i in top5_idx]
                fig = go.Figure(go.Bar(
                    x=top5_probs, y=top5_cats, orientation='h',
                    marker_color=["#7c83fd", "#5a5fcf", "#302b63", "#1a1a5e", "#0d0d3a"],
                    text=[f"{p:.1f}%" for p in top5_probs], textposition='outside'
                ))
                fig.update_layout(xaxis_title="Probability (%)", plot_bgcolor='white',
                                  font=dict(family='DM Sans'), yaxis=dict(autorange="reversed"), margin=dict(l=20, r=60))
                st.plotly_chart(fig, width="stretch")

            with st.expander("🔎 View Preprocessed Text"):
                st.text(cleaned[:1000] + ("..." if len(cleaned) > 1000 else ""))

# ═══════════════════════════════════════════════════════════════
# PAGE: BATCH PREDICT
# ═══════════════════════════════════════════════════════════════
elif page == "📁 Batch Predict":
    st.markdown('<div class="section-title">Batch Resume Prediction</div>', unsafe_allow_html=True)
    st.write("Upload a CSV file with a `Resume` column to classify multiple resumes at once.")
    saved = load_saved_models()
    if 'tfidf' not in saved or ('mnb' not in saved and 'knc' not in saved):
        st.warning("⚠️ Models not found. Please go to **🤖 Train Models** tab first.")
        st.stop()

    model_choice = st.selectbox(
        "Choose Model for Batch Prediction",
        [m for m in ["Multinomial Naive Bayes (MNB)", "K-Nearest Neighbors (KNN)"]
         if ('mnb' in saved and 'MNB' in m) or ('knc' in saved and 'KNN' in m)]
    )
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        if 'Resume' not in batch_df.columns:
            st.error("❌ CSV must have a `Resume` column.")
        else:
            st.success(f"✅ Loaded {len(batch_df)} resumes.")
            st.dataframe(batch_df.head(5), width="stretch")
            if st.button("⚡ Run Batch Prediction"):
                with st.spinner("Processing all resumes..."):
                    batch_df['Processed'] = batch_df['Resume'].apply(resumeKeywords)
                    features = saved['tfidf'].transform(batch_df['Processed'])
                    preds = saved['mnb'].predict(features) if "MNB" in model_choice else saved['knc'].predict(features)
                    batch_df['Predicted_ID'] = preds
                    batch_df['Predicted_Category'] = batch_df['Predicted_ID'].apply(
                        lambda x: CATEGORY_MAPPING.get(int(x), "Unknown"))
                st.success("✅ Batch prediction complete!")
                st.dataframe(batch_df[['Resume', 'Predicted_Category']].head(20), width="stretch")

                pred_counts = batch_df['Predicted_Category'].value_counts().reset_index()
                pred_counts.columns = ['Category', 'Count']
                fig = px.bar(pred_counts, x='Category', y='Count', color='Category',
                             color_discrete_sequence=px.colors.qualitative.Pastel,
                             title="Predicted Category Distribution")
                fig.update_layout(xaxis=dict(tickangle=-30), plot_bgcolor='white',
                                  showlegend=False, font=dict(family='DM Sans'))
                st.plotly_chart(fig, width="stretch")

                csv_out = batch_df[['Resume', 'Predicted_Category']].to_csv(index=False).encode('utf-8')
                st.download_button(label="⬇️ Download Results as CSV", data=csv_out,
                                   file_name="predicted_resumes.csv", mime="text/csv")

# ═══════════════════════════════════════════════════════════════
# PAGE: 🏆 RANK RESUMES  ← NEW
# ═══════════════════════════════════════════════════════════════
elif page == "🏆 Rank Resumes":
    st.markdown('<div class="section-title">Resume Ranking</div>', unsafe_allow_html=True)
    st.write("Upload multiple resumes and enter a job description — candidates are ranked by how well they match.")

    if not PDF_SUPPORT:
        st.info("💡 **Tip:** Install `pdfplumber` to also support PDF uploads → `pip install pdfplumber`")

    saved = load_saved_models()
    has_models = 'tfidf' in saved and ('mnb' in saved or 'knc' in saved)

    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.markdown("""
        <div style="background:#f0f2ff; border-radius:14px; padding:1.2rem 1.4rem; border:1px solid #d0d4f5;">
            <p style="font-family:'Syne',sans-serif; font-weight:700; color:#302b63;
                      font-size:1rem; margin:0 0 0.8rem 0;">How Ranking Works</p>
            <p style="font-size:0.83rem; color:#555; line-height:1.8; margin:0;">
                1. Each resume is cleaned using NLP preprocessing.<br>
                2. Resumes + job description are converted to <b>TF-IDF vectors</b>.<br>
                3. <b>Cosine similarity</b> scores how close each resume is to the job description.<br>
                4. Resumes are sorted highest → lowest match.<br>
                5. If models are trained, the predicted job category is also shown.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_left:
        st.markdown("#### 📋 Job Description")
        jd_text = st.text_area(
            "Job description",
            height=160,
            placeholder="e.g. We are looking for a Python Developer with experience in Django, REST APIs, PostgreSQL, machine learning and cloud deployment on AWS...",
            label_visibility="collapsed"
        )

        st.markdown("#### 📂 Upload Resumes")
        accepted = ["txt", "pdf"] if PDF_SUPPORT else ["txt"]
        uploaded_resumes = st.file_uploader(
            f"Upload resume files ({', '.join(accepted)}) — you can select multiple",
            type=accepted,
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        use_model = False
        model_choice_rank = None
        if has_models:
            use_model = st.checkbox("Also predict job category for each resume", value=True)
            if use_model:
                model_choice_rank = st.selectbox(
                    "Model for category prediction",
                    [m for m in ["Multinomial Naive Bayes (MNB)", "K-Nearest Neighbors (KNN)"]
                     if ('mnb' in saved and 'MNB' in m) or ('knc' in saved and 'KNN' in m)]
                )

        rank_btn = st.button("🏆 Rank Resumes Now")

    # ── RUN RANKING ──────────────────────────────────────────
    if rank_btn:
        errors = []
        if not jd_text.strip():
            errors.append("Please enter a job description.")
        if not uploaded_resumes:
            errors.append("Please upload at least one resume file.")
        for e in errors:
            st.error(e)

        if not errors:
            with st.spinner("Reading and ranking resumes..."):

                # 1. Read resume texts
                resumes = []
                for f in uploaded_resumes:
                    text = extract_text(f)
                    if text is None and not PDF_SUPPORT and f.name.lower().endswith('.pdf'):
                        st.warning(f"⚠️ Skipped `{f.name}` — install `pdfplumber` for PDF support.")
                        continue
                    if text and text.strip():
                        resumes.append({'filename': f.name, 'raw_text': text})
                    else:
                        st.warning(f"⚠️ Could not extract text from `{f.name}` — skipping.")

                if not resumes:
                    st.error("No readable resumes found.")
                    st.stop()

                # 2. Clean
                for r in resumes:
                    r['cleaned'] = resumeKeywords(r['raw_text'])
                cleaned_jd = resumeKeywords(jd_text)

                # 3. TF-IDF vectorise together (JD first)
                all_texts = [cleaned_jd] + [r['cleaned'] for r in resumes]
                rank_tfidf = TfidfVectorizer(stop_words='english')
                tfidf_matrix = rank_tfidf.fit_transform(all_texts)

                jd_vec     = tfidf_matrix[0]
                resume_vecs = tfidf_matrix[1:]

                # 4. Cosine similarity
                scores = cosine_similarity(jd_vec, resume_vecs)[0]
                for i, r in enumerate(resumes):
                    r['score'] = float(scores[i])

                # 5. Category prediction
                if use_model and has_models:
                    feat = saved['tfidf'].transform([r['cleaned'] for r in resumes])
                    preds = saved['mnb'].predict(feat) if "MNB" in (model_choice_rank or "") else saved['knc'].predict(feat)
                    for i, r in enumerate(resumes):
                        r['category'] = CATEGORY_MAPPING.get(int(preds[i]), "Unknown")
                else:
                    for r in resumes:
                        r['category'] = None

                # 6. Sort
                resumes_sorted = sorted(resumes, key=lambda x: x['score'], reverse=True)

            # ── DISPLAY RESULTS ──────────────────────────────
            st.markdown("---")
            st.markdown('<div class="section-title">🏆 Ranked Results</div>', unsafe_allow_html=True)

            top_score = resumes_sorted[0]['score'] if resumes_sorted else 1.0
            medals = {0: "🥇", 1: "🥈", 2: "🥉"}

            for idx, r in enumerate(resumes_sorted):
                pct      = round(r['score'] * 100, 1)
                bar_pct  = round((r['score'] / max(top_score, 0.0001)) * 100, 1)
                medal    = medals.get(idx, f"#{idx + 1}")
                cat_badge = f'<span class="rank-category">{r["category"]}</span>' if r['category'] else ''

                st.markdown(f"""
                <div class="rank-card">
                    <div style="display:flex; align-items:center; gap:1rem;">
                        <span style="font-size:2rem; min-width:2.5rem;">{medal}</span>
                        <div style="flex:1;">
                            <span class="rank-filename">📄 {r['filename']}</span>{cat_badge}
                            <div class="rank-score-bar-bg">
                                <div class="rank-score-bar-fill" style="width:{bar_pct}%;"></div>
                            </div>
                            <span class="rank-score-text">Match Score: {pct}%</span>
                        </div>
                        <div style="text-align:right; min-width:4rem;">
                            <span style="font-family:'Syne',sans-serif; font-size:1.6rem;
                                         font-weight:800; color:#7c83fd;">{pct}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Chart ────────────────────────────────────────
            st.markdown('<div class="section-title">📊 Score Comparison Chart</div>', unsafe_allow_html=True)
            chart_df = pd.DataFrame({
                'Resume': [r['filename'] for r in resumes_sorted],
                'Match Score (%)': [round(r['score'] * 100, 2) for r in resumes_sorted],
                'Category': [r['category'] or 'N/A' for r in resumes_sorted]
            })
            fig = px.bar(
                chart_df, x='Resume', y='Match Score (%)',
                color='Match Score (%)',
                color_continuous_scale=["#c9cfe8", "#7c83fd", "#302b63"],
                text='Match Score (%)',
                hover_data=['Category']
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                xaxis_title='Resume File', yaxis_title='Match Score (%)',
                yaxis=dict(range=[0, 115]),
                plot_bgcolor='white', coloraxis_showscale=False,
                font=dict(family='DM Sans', size=12),
                xaxis=dict(tickangle=-20)
            )
            st.plotly_chart(fig, width="stretch")

            # ── Download ─────────────────────────────────────
            dl_df = pd.DataFrame({
                'Rank': list(range(1, len(resumes_sorted) + 1)),
                'Resume File': [r['filename'] for r in resumes_sorted],
                'Match Score (%)': [round(r['score'] * 100, 2) for r in resumes_sorted],
                'Predicted Category': [r['category'] or 'N/A' for r in resumes_sorted]
            })
            st.download_button(
                label="⬇️ Download Ranking as CSV",
                data=dl_df.to_csv(index=False).encode('utf-8'),
                file_name="resume_ranking.csv",
                mime="text/csv"
            )

            # ── Summary ──────────────────────────────────────
            st.info(
                f"✅ **{len(resumes_sorted)} resumes ranked.** "
                f"Top candidate: **{resumes_sorted[0]['filename']}** "
                f"with a match score of **{round(resumes_sorted[0]['score'] * 100, 1)}%**"
            )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="text-align:center; font-size:0.8rem; color:#aaa;">
    Resume Screening System · Built with Streamlit, scikit-learn & NLTK
</p>
""", unsafe_allow_html=True)