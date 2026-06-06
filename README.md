# 💊 Drug Review Sentiment Analyzer — BioBERT

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://asifnawaz-drug-sentiment.streamlit.app)
[![HuggingFace](https://img.shields.io/badge/Model-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface)](https://huggingface.co/asif-nawaz-ml/biobert_drug_sentiment)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

> **BioBERT fine-tuned on 215,000+ real patient drug reviews** for 3-class sentiment classification — Positive, Neutral, Negative. Deployed as a live interactive web application.

---

## 🎯 Problem Statement

Pharmaceutical companies, hospitals, and health agencies need to understand patient experiences with medications at scale. Manual review analysis is impossible with millions of reviews. This project automates sentiment analysis using domain-specific BioBERT — trained on biomedical text.

---

## 🚀 Live Demo

**[💊 Try the App →](https://asifnawaz-drug-sentiment.streamlit.app)**

Enter any drug review and get:
- Predicted sentiment (Positive / Neutral / Negative)
- Confidence scores for each class
- Real-time BioBERT inference

---

## 📊 Results

| Model | Accuracy | F1 (weighted) | Notes |
|-------|----------|---------------|-------|
| TF-IDF + Logistic Regression | 80.37% | 0.816 | Baseline |
| **BioBERT (fine-tuned)** | **83.20%** | **0.848** | **Best model** |

### Per-Class Performance (BioBERT)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Negative | 0.86 | 0.81 | 0.83 |
| Neutral | 0.44 | 0.76 | 0.56 |
| Positive | 0.97 | 0.86 | 0.91 |

---

## 🗂️ Project Structure

```
drug-review-sentiment-biobert/
│
├── data/
│   ├── raw/                          # Original Kaggle dataset
│   └── processed/                    # Cleaned + labeled data
│
├── notebooks/
│   ├── 01_eda.ipynb                  # EDA + WordClouds
│   ├── 02_baseline.ipynb             # TF-IDF + LR baseline
│   └── 03_biobert.ipynb             # BioBERT fine-tuning
│
├── dashboard/
│   └── streamlit_app.py             # Live web application
│
├── reports/
│   └── figures/                     # Publication-ready plots
│       ├── 01_eda_overview.png
│       ├── 02_wordclouds.png
│       ├── 03_baseline_confusion.png
│       ├── 04_tfidf_features.png
│       └── model_comparison.png
│
├── src/
│   └── download_data.py             # Data pipeline
│
├── requirements.txt
└── README.md
```

---

## 🤖 Model Details

- **Base Model:** `dmis-lab/biobert-base-cased-v1.2`
- **Task:** 3-class sentiment classification
- **Training Data:** 59,368 balanced samples (20k per class)
- **Epochs:** 3
- **Learning Rate:** 2e-5
- **Max Length:** 256 tokens
- **Hardware:** Google Colab T4 GPU (~35 min training)
- **Model Hub:** [asif-nawaz-ml/biobert_drug_sentiment](https://huggingface.co/asif-nawaz-ml/biobert_drug_sentiment)

---

## 📈 Key Findings

- **Positive reviews dominate** — 66.3% of dataset (class imbalance handled)
- **Top negative words:** worse, not recommend, disappointed, horrible
- **Top positive words:** love, amazing, miracle, works, highly recommend
- **BioBERT outperforms** TF-IDF baseline by +2.83% accuracy and +3.2% F1
- **Neutral class hardest** — mixed sentiment confuses even BioBERT (F1: 0.56)

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| NLP Model | BioBERT (dmis-lab) |
| Framework | HuggingFace Transformers, PyTorch |
| Baseline | TF-IDF, Scikit-learn |
| Dashboard | Streamlit |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Deployment | Streamlit Cloud + HuggingFace Hub |

---

## 🚀 Run Locally

```bash
# Clone
git clone https://github.com/Asif5588-M/drug-review-sentiment-biobert.git
cd drug-review-sentiment-biobert

# Environment
conda create -n biobert-env python=3.11 -y
conda activate biobert-env
pip install -r requirements.txt

# Run app
streamlit run dashboard/streamlit_app.py
```

---

## 👨‍💻 Author

**Asif Nawaz**
- 🏥 Medical Technician | PMAS Arid Agriculture University
- 🎓 MPhil Economics (Health Economics)
- 📄 Published Researcher — HEC Y-Category Journal
- 🔗 [Upwork Profile](https://www.upwork.com/freelancers/~016fa773fe0b528410)
- 🌐 [Pakistan CHE Dashboard](https://asifnawaz-pakistan-health.streamlit.app)
- 🤗 [HuggingFace](https://huggingface.co/asif-nawaz-ml)

---

## 📄 Dataset

UCI Drug Review Dataset — Kaggle
- 215,063 patient reviews
- 3,436 unique drugs
- 884 medical conditions
- Rating scale: 1–10