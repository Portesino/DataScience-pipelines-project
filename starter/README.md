
# 🏷️ StyleSense • Predictive Review Pipeline
> **Udacity Data Scientist Nanodegree** • Advanced Machine Learning Pipelines

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange.svg)](https://scikit-learn.org/)
[![spaCy](https://img.shields.io/badge/spaCy-NLP-green.svg)](https://spacy.io/)

### The Problem
StyleSense is a rapidly growing women's clothing e-commerce retailer. While thousands of new customers leave rich, descriptive text reviews, many omit the explicit "recommend" toggle. Without this binary signal, dissatisfied customers blend into the noise, and emerging product trends slip through the cracks.

### The Solution
A production-ready, end-to-end `scikit-learn` pipeline that analyzes unstructured review text and metadata to automatically predict missing customer recommendations. By fusing feature engineering with advanced NLP (spaCy lemmatization + TF-IDF), we bridge the data gap right at the point of feedback.

---

## ⚙️ Architecture & Data Workflow

The pipeline encapsulates numerical, categorical, and text preprocessing into a single, cohesive estimator object:


```

[Raw Data]
│
├──► Numerical ────► Median Imputation ──► StandardScaler
├──► Categorical ──► Freq Imputation ────► OneHotEncoder (Safe Mode)
│
└──► Text (Title + Body)
│
├──► Custom Character Counter (Stylistic Emphasis)
└──► spaCy Lemmatizer ──► TF-IDF Vectorizer

```

### Repository Blueprint
```text
datascience-pipelines-project/
├── starter/
│   ├── data/reviews.csv     # 18,442 anonymized customer records
│   └── starter.ipynb        # EDA, pipeline engineering & validation
├── requirements.txt         # Project dependencies
└── LICENSE.txt

```

---

## 🛠️ Custom Transformers

To comply with `scikit-learn`'s strict validation architecture, two custom, stateless transformers were built from scratch (extending `BaseEstimator` and `TransformerMixin`):

* **`CountCharacter`**: Extracts stylistic markers (exclamation marks, question marks, spacing density) as structural signals of customer emotion and emphasis.
* **`SpacyLemmatizer`**: Leverages an optimized `nlp.pipe` batch architecture to strip punctuation and whitespace, delivering clean lemmatized tokens ready for TF-IDF.

---

## 📊 Business Impact & Results

With a heavily imbalanced dataset (**81.6% recommended / 18.4% not recommended**), optimizing for raw accuracy would mask poor minority class performance. Our hyperparameter tuning (`RandomizedSearchCV` over 24 total fits) optimized directly for **Macro F1-Score** to yield the following production parameters:

* `classifier__n_estimators`: `300`
* `classifier__max_depth`: `40`
* `classifier__min_samples_split`: `5`
* `classifier__class_weight`: `'balanced'`
* `feature_engineering__tfidf_text__tfidf_vectorizer__max_features`: `3000`

### Performance Comparison (Baseline → Tuned)

| Metric | Baseline Model | Tuned Model | Delta |
| --- | --- | --- | --- |
| **Accuracy** | 0.8699 | 0.8802 | **+1.03 pp** |
| **Macro F1-Score** | 0.7233 | **0.8024** | **+7.91 pp** |
| **Recall (Class 0 — Dissatisfied)** | 0.4006 | **0.7125** | **+31.19 pp** |
| **Precision (Class 0)** | 0.7486 | 0.6472 | -10.14 pp |
| **F1-Score (Class 0)** | 0.5219 | **0.6783** | **+15.64 pp** |
| **F1-Score (Class 1 — Recommended)** | 0.9247 | 0.9264 | +0.0017 |

### Detailed Confusion Matrix Evolution

```text
BASELINE CONFUSION MATRIX                 TUNED CONFUSION MATRIX
Predicted:  0       1                     Predicted:  0       1
Actual: 0 [ 131 ] [ 196 ]                 Actual: 0 [ 233 ] [  94 ]
Actual: 1 [  44 ] [ 1474]                 Actual: 1 [ 127 ] [ 1391]

```

> 🎯 **Business Trade-off:** By injecting `class_weight='balanced'` and optimizing the text feature representation, we achieved a massive breakthrough: **recall for dissatisfied customers jumped from 40.06% to 71.25%**. The tuned model now flags **233 out of 327 unhappy customers** (cutting overlooked critical insights down from 196 to just 94). For StyleSense, this represents a **52% reduction in missed customer issues**. The minor operational cost of managing a few more false alarms (Class 0 Precision shift to 64.72%) is heavily outweighed by the ability to proactively safeguard brand reputation.

---

## 🚀 Quickstart & Setup

### 1. Environment Setup

```bash
# Clone the repository
git clone <repo-url> 
cd datascience-pipelines-project

# Spin up a virtual environment (Windows PowerShell example)
python -m venv venv
.\venv\Scripts\activate

# For macOS / Linux:
# source venv/bin/activate

# Install dependencies & the NLP pipeline
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm

```

### 2. Running the Project

* Open the workspace in your IDE or launch Jupyter Notebook.
* **Crucial:** Verify that your Jupyter kernel points to the Python interpreter inside the `venv` environment, not your global installation.
* Execute the cells sequentially within `starter/starter.ipynb`.
* *Execution Note:* The cross-validation tuning phase takes roughly **2 to 3 minutes** on a standard multi-core laptop CPU due to `n_jobs=-1` parallelization and optimized spaCy text batching (`batch_size=64`).

---

## 🎓 License & Context

Built as part of the **Udacity Data Scientist Nanodegree curriculum**. Code and documentation are available under the [MIT License](https://www.google.com/search?q=LICENSE.txt).
