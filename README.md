# 🔍 Customer Churn Prediction
### End-to-End Machine Learning Classification Project

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Completed-2ea44f?style=flat"/>
  <img src="https://img.shields.io/badge/Recall-1.000-brightgreen?style=flat"/>
</p>

> Membangun model klasifikasi machine learning untuk memprediksi pelanggan yang akan churn, dengan fokus pada metrik **Recall** agar tidak ada pelanggan berisiko yang terlewat — serta menghitung simulasi profitabilitas nyata dari penerapan model.

---

## 📌 Table of Contents
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Workflow](#-workflow)
- [EDA Insights](#-eda-insights)
- [Feature Engineering](#%EF%B8%8F-feature-engineering)
- [Model Comparison](#-model-comparison--evaluation)
- [Business Simulation](#-business-profit-simulation)
- [Strategic Recommendations](#-strategic-recommendations)
- [How to Run](#-how-to-run)

---

## 💼 Business Problem

Customer churn adalah salah satu tantangan terbesar dalam industri subscription-based. Kehilangan satu pelanggan bukan hanya kehilangan revenue saat ini, tetapi juga **Customer Lifetime Value (CLV)** jangka panjang.

**Pertanyaan Bisnis:**
- Pelanggan mana yang paling berisiko untuk churn?
- Faktor apa yang paling dominan mendorong churn?
- Seberapa besar potensi pendapatan yang bisa diselamatkan dengan model prediksi?

> **Pendekatan:** Karena biaya *False Negative* (pelanggan churn tidak terdeteksi = ~$500 hilang) jauh lebih mahal dari *False Positive* (program retensi tidak perlu = ~$50), maka **Recall dipilih sebagai metrik utama**.

---

## 🗃️ Dataset

| Info | Detail |
|------|--------|
| Sumber | [Kaggle — Customer Churn Dataset](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset) |
| File | `customer_churn_dataset-training-master.csv` |
| Target | `Churn`: 1 = berhenti berlangganan, 0 = tetap |
| Fitur Kategorikal | Gender, Subscription Type, Contract Length |
| Fitur Numerik | Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend, Last Interaction |
| Missing Values | Ada — ditangani dengan `dropna()` |
| Duplikat | Tidak ditemukan |

---

## 🔄 Workflow

```
Data Loading
    ↓
Exploratory Data Analysis (EDA)
    ↓ Chi-Square + Cramér's V  (kategorikal)
    ↓ Welch's T-Test + Violin Plot  (numerik)
Feature Engineering
    ↓ Split → Drop ID → Encode → Scale → SMOTE
Model Training  (4 Models)
    ↓ Logistic Regression | KNN | Decision Tree | SVM
Model Evaluation
    ↓ Recall · Confusion Matrix · Learning Curve
Best Model Selection → Business Profit Simulation
```

---

## 📊 EDA Insights

### Variabel Kategorikal — Chi-Square Test & Cramér's V

| Variabel | Churn Rate | Cramér's V | Kekuatan Hubungan |
|----------|:----------:|:----------:|:-----------------:|
| Contract Length (Monthly) | **100%** | 0.43 | 🔴 Kuat |
| Contract Length (Annual/Quarterly) | ~46% | 0.43 | 🔴 Kuat |
| Gender (Female) | 66% | 0.18 | 🟡 Sedang |
| Gender (Male) | 49% | 0.18 | 🟡 Sedang |
| Subscription Type | 56–58% | 0.02 | 🟢 Lemah |

**Key Finding:** Seluruh pelanggan dengan kontrak **Monthly** terbukti churn — ini adalah *strongest signal* dalam dataset.

---

### Variabel Numerik — Welch's T-Test

| Fitur | Temuan | Implikasi Bisnis |
|-------|--------|-----------------|
| **Tenure** | Churn → tenure lebih pendek | Pelanggan baru paling rentan — perkuat onboarding |
| **Support Calls** | Churn → lebih sering hubungi CS | Keluhan tidak terselesaikan → tingkatkan kualitas support |
| **Payment Delay** | Churn → lebih sering terlambat bayar | Tawarkan fleksibilitas pembayaran |
| **Usage Frequency** | Churn → jarang gunakan layanan | Dorong engagement lewat notifikasi & kampanye |
| **Last Interaction** | Churn → lebih lama tidak berinteraksi | Proaktif outreach untuk pelanggan pasif |
| **Total Spend** | Churn → cenderung low-spender | Sorot value layanan ke segmen ini |
| **Age** | Distribusi usia churn lebih luas | Semua segmen usia berisiko — perlu pendekatan personal |

---

## ⚙️ Feature Engineering

| # | Langkah | Detail | Alasan |
|---|---------|--------|--------|
| 1 | **Train-Test Split 80/20** | `stratify=y` | Proporsi kelas dipertahankan di kedua set |
| 2 | **Drop CustomerID** | Hapus dari X_train & X_test | Bukan fitur prediktif — cegah model menghafal ID |
| 3 | **Label Encoding** | Gender: Male=1, Female=0 | Variabel binary cukup representasi 0/1 |
| 4 | **One-Hot Encoding** | Subscription Type & Contract Length | Cegah asumsi ordinal yang salah |
| 5 | **StandardScaler + SMOTE** | Scaling numerik + balancing kelas | SMOTE **hanya pada train data** — zero data leakage |

---

## 🤖 Model Comparison & Evaluation

| Model | Recall Train | Recall Test | Gap | Status |
|-------|:-----------:|:-----------:|:---:|--------|
| Logistic Regression | 0.870 | 0.870 | 0.000 | ✅ Stabil |
| KNN (+ SMOTE) | 0.943 | 0.925 | 0.018 | ✅ Baik |
| **Decision Tree** | **1.000** | **1.000** | **0.000** | 🏆 **TERBAIK** |
| SVM (LinearSVC) | 0.867 | 0.867 | 0.000 | ✅ Stabil |

---

## 🏆 Best Model: Decision Tree

```
Confusion Matrix — Decision Tree (Test Set)

                  Predicted: 0    Predicted: 1
Actual: 0  (Non-Churn)  38,162            5
Actual: 1  (Churn)           5       49,995
```

| Metrik | Nilai |
|--------|:-----:|
| Recall | **1.000** |
| Precision | ~1.000 |
| F1-Score | ~1.000 |
| False Negatives | **5** dari 50.000 pelanggan churn |

> Dari **50.000 pelanggan yang benar-benar churn**, model hanya melewatkan **5 orang**.

---

## 💰 Business Profit Simulation

| Komponen | Detail | Nilai |
|----------|--------|------:|
| Biaya retensi / pelanggan | Program diskon, loyalty outreach | $50 |
| Pendapatan hilang / churn | Estimasi Customer Lifetime Value | $500 |
| Total biaya retensi | (FP + TP) × $50 | $2,500,000 |
| Pendapatan diselamatkan | TP × $500 = 49.995 × $500 | $24,997,500 |
| FN Loss | 5 × $500 | $2,500 |
| **Net Profit Bersih** | Saved − Cost − FN Loss | **$22,495,000** |
| **ROI Model** | Per $1 investasi retensi | **~$10** |

---

## 💡 Strategic Recommendations

| Prioritas | Segmen | Rekomendasi |
|:---------:|--------|-------------|
| 🔴 Tinggi | Kontrak Bulanan (churn 100%) | Insentif upgrade ke kontrak tahunan/kuartalan |
| 🔴 Tinggi | Pelanggan Baru (tenure pendek) | Program onboarding intensif di 3 bulan pertama |
| 🟡 Sedang | Support Calls tinggi | Tingkatkan kualitas CS & first-call resolution rate |
| 🟡 Sedang | Payment Delay tinggi | Fleksibilitas pembayaran & opsi cicilan |
| 🟢 Normal | Low Usage Frequency | Notifikasi & kampanye re-engagement |
| 🟢 Normal | Pelanggan Perempuan (churn 66%) | Personalisasi program retensi berbasis segmen |

---

## ✅ Strengths & ⚠️ Areas for Improvement

**Kelebihan:**
- Recall = 1.000 — hampir tidak ada pelanggan churn yang terlewat
- 4 model diuji dan dibandingkan dengan metrik yang tepat
- SMOTE hanya pada train data — tidak ada data leakage
- Validasi statistik dengan Chi-Square, Cramér's V, dan Welch's T-Test
- Simulasi profitabilitas konkret dengan asumsi biaya yang transparan

**Peluang Peningkatan:**
- Recall 1.000 bisa mengindikasikan overfitting — perlu validasi k-fold CV
- Tambahkan Random Forest / XGBoost untuk perbandingan lebih robust
- Feature importance Decision Tree belum divisualisasikan
- Threshold tuning pada model lain untuk optimasi precision-recall trade-off
- Sensitivity analysis pada simulasi profitabilitas

---

## 🔧 Tech Stack

```python
import pandas as pd, numpy as np                          # Data manipulation
import seaborn as sns, matplotlib.pyplot as plt           # Visualization
from sklearn.tree import DecisionTreeClassifier           # Best model
from sklearn.linear_model import LogisticRegression       # Baseline
from sklearn.neighbors import KNeighborsClassifier        # KNN
from sklearn.svm import LinearSVC                         # SVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE                  # Class balancing
from scipy.stats import chi2_contingency, ttest_ind       # Statistical tests
```


## 👩‍💻 Author

**Lhedya Monica Ismon**

[![LinkedIn](https://www.linkedin.com/in/lhedya/)]

---

<p align="center"><i>Made with curiosity and lots of ☕ | 2025</i></p>
