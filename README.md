# 📄 Resume Screening System — Streamlit App

An AI-powered resume classifier that predicts job categories from resume text using **NLP + TF-IDF + ML models** (Naive Bayes & KNN).

---

## 📁 Project Folder Structure

Make sure your folder looks like this before running:

```
your-project-folder/
│
├── app.py                  ← Streamlit app (the file you downloaded)
├── requirements.txt        ← Python dependencies
├── ResumeDataSet.csv       ← Your dataset (REQUIRED)
│
├── mnb.pkl                 ← Auto-generated after training
├── knc.pkl                 ← Auto-generated after training
└── tfidf.pkl               ← Auto-generated after training
```

> ⚠️ **Important:** `ResumeDataSet.csv` must be in the same folder as `app.py`.
> The `.pkl` files are created automatically when you click "Train" inside the app.

---

## 🚀 How to Run the App (Step-by-Step)

### Step 1 — Make sure Python is installed
```bash
python --version
# Should show Python 3.8 or above
```

### Step 2 — Create a virtual environment (recommended)
```bash
# Create it
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install all dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the Streamlit app
```bash
streamlit run app.py
```

The app will open automatically in your browser at:
```
http://localhost:8501
```

---

## 🖥️ App Pages & Features

| Page | What It Does |
|------|-------------|
| 🏠 **Home** | Overview of the system and pipeline |
| 📊 **Data Analysis** | Bar chart, Pie chart, Word Cloud of your dataset |
| 🤖 **Train Models** | Train MNB + KNN, see accuracy & classification reports |
| 🔍 **Predict Resume** | Paste a single resume → get predicted job category |
| 📁 **Batch Predict** | Upload a CSV of resumes → download predictions |

---

## 🔄 Typical Workflow Inside the App

```
1. Go to 📊 Data Analysis → verify your dataset loaded correctly
2. Go to 🤖 Train Models → click "Train Both Models"
3. Go to 🔍 Predict Resume → paste any resume text → click Predict
4. (Optional) Go to 📁 Batch Predict → upload CSV → download results
```

---

## 🐛 Common Issues & Fixes

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'streamlit'` | Run `pip install -r requirements.txt` |
| `ResumeDataSet.csv not found` | Make sure the CSV is in the **same folder** as `app.py` |
| `Models not found` error on Predict page | Go to **Train Models** tab and click the Train button first |
| `NLTK data not found` | The app auto-downloads it; if it fails, run `python -c "import nltk; nltk.download('all')"` |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |

---

## 📦 Dependencies

- `streamlit` — Web app framework
- `pandas`, `numpy` — Data handling
- `scikit-learn` — ML models (MNB, KNN, TF-IDF)
- `nltk` — NLP preprocessing (stopwords, lemmatization)
- `plotly` — Interactive charts
- `wordcloud`, `matplotlib` — Word cloud visualization
- `tensorflow` — (Optional) For RNN model from the notebook

---

## 💡 Tips

- The app **auto-saves** trained models (`mnb.pkl`, `knc.pkl`, `tfidf.pkl`) in your folder, so you only need to train once.
- On the **Predict** page, Naive Bayes also shows **Top 5 probability scores** per category.
- The **Batch Predict** page lets you download a CSV with predicted categories for all uploaded resumes.
