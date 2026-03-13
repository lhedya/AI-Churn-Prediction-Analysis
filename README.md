# 🚨 AI Churn Prediction & Analysis
**End-to-End Machine Learning Classification | Machine Learning Bootcamp - Dibimbing.id**

**Author:** Lhedya Monica Ismon

---

## 🎯 Objective
Membangun model klasifikasi machine learning untuk memprediksi 
pelanggan yang akan churn, dengan fokus pada metrik Recall agar 
tidak ada pelanggan berisiko yang terlewat, serta menghitung 
simulasi profitabilitas nyata dari penerapan model.

---

## 🗃️ Dataset
| Info | Detail |
|------|--------|
| Sumber | Kaggle — customer_churn_dataset-training-master.csv |
| Target | Churn: 1 = berhenti berlangganan, 0 = tetap |
| Fitur Kategorikal | Gender, Subscription Type, Contract Length |
| Fitur Numerik | Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend, Last Interaction |
| Missing Values | Ada — ditangani dengan dropna() |
| Duplikat | Tidak ditemukan |

---

## 📋 Analysis Steps
1. EDA kategorikal → Chi-Square Test & Cramér's V
2. EDA numerik → Welch's T-Test & Violin Plot
3. Feature Engineering (5 langkah)
4. Training 4 model: Logistic Regression, KNN, Decision Tree, SVM
5. Evaluasi model berbasis Recall
6. Simulasi profitabilitas & ROI

---

## 📊 EDA Insights

### Variabel Kategorikal (Chi-Square + Cramér's V)
| Variabel | Churn Rate | Cramér's V | Insight |
|----------|-----------|------------|---------|
| Contract Length (Monthly) | 100% | 0.43 (Kuat) | Semua pelanggan kontrak bulanan churn! |
| Contract Length (Annual/Quarterly) | ~46% | 0.43 (Kuat) | Kontrak jangka panjang jauh lebih loyal |
| Gender (Female) | 66% | 0.18 (Sedang) | Perempuan lebih berisiko churn vs laki-laki (49%) |
| Subscription Type | 56-58% | 0.02 (Lemah) | Tipe paket bukan faktor utama churn |

### Variabel Numerik (Welch's T-Test + Violin Plot)
| Fitur | Temuan | Implikasi Bisnis |
|-------|--------|-----------------|
| Tenure | Churn → tenure lebih pendek | Pelanggan baru paling rentan — butuh onboarding kuat |
| Support Calls | Churn → lebih banyak panggilan | Keluhan sering = masalah belum terselesaikan |
| Payment Delay | Churn → keterlambatan lebih tinggi | Tawarkan fleksibilitas pembayaran |
| Usage Frequency | Churn → frekuensi lebih rendah | Dorong engagement dengan notifikasi |
| Last Interaction | Churn → lebih lama tidak interaksi | Proaktif outreach untuk pelanggan pasif |
| Total Spend | Churn → pengeluaran lebih rendah | Sorot manfaat layanan ke pelanggan low-spend |
| Age | Distribusi usia churn lebih luas | Semua segmen usia bisa churn — personalisasi retention |

---

## ⚙️ Feature Engineering
| # | Langkah | Yang Dilakukan | Alasan |
|---|---------|---------------|--------|
| 1 | Train-Test Split 80/20 | Stratified split menggunakan stratify=y | Proporsi kelas Churn dipertahankan di kedua set |
| 2 | Drop CustomerID | Hapus kolom identifier dari X_train & X_test | Bukan fitur prediktif — bisa membuat model menghafal ID |
| 3 | Label Encoding | Gender: Male=1, Female=0 | Variabel binary cukup 0/1 |
| 4 | One-Hot Encoding | Subscription Type & Contract Length | Cegah asumsi ordinal yang salah |
| 5 | StandardScaler + SMOTE | Scaling numerik + seimbangkan kelas | SMOTE hanya pada train data — tidak ada data leakage |

---

## 🤖 Model Comparison & Evaluasi

| Model | Recall Train | Recall Test | Gap | Status |
|-------|-------------|-------------|-----|--------|
| Logistic Regression | 0.870 | 0.870 | 0.000 | Stabil |
| KNN (SMOTE) | 0.943 | 0.925 | 0.018 | Baik |
| **Decision Tree** | **1.000** | **1.000** | **0.000** | **TERBAIK** |
| SVM (LinearSVC) | 0.867 | 0.867 | 0.000 | Stabil |

> **Mengapa Recall sebagai metrik utama?**
> Biaya False Negative (pelanggan churn tidak terdeteksi = $500 hilang)
> jauh lebih mahal dari False Positive (program retensi tidak perlu = $50).
> Maka Recall adalah metrik paling relevan untuk kasus ini.

---

## 🏆 Model Terpilih: Decision Tree
- Recall = 1.000 pada KEDUA train dan test data — gap = 0.000
- Confusion Matrix: TN=38.162 | FP=5 | FN=5 | TP=49.995
- Dari 50.000 pelanggan churn, model berhasil mendeteksi 49.995 (hanya 5 terlewat)

---

## 💰 Simulasi Profitabilitas

| Komponen | Perhitungan | Nilai |
|----------|-------------|-------|
| Biaya retensi per pelanggan | Asumsi biaya program retensi per pelanggan (diskon, loyalty outreach) | $50 |
| Pendapatan hilang per churn | Estimasi Customer Lifetime Value yang hilang jika pelanggan churn | $500 |
| Total biaya retensi | (FP + TP) × $50 = 50.000 × $50 | $2,500,000 |
| Pendapatan diselamatkan | TP × $500 = 49.995 × $500 | $24,997,500 |
| **Net Profit Bersih** | Saved − Cost − FN Loss | **$22,495,000** |
| **ROI Model** | Per $1 investasi | **~$10 pendapatan diselamatkan** |

---

## 💡 Kesimpulan
Decision Tree adalah model terbaik dengan Recall = 1.000 pada train 
dan test, membuktikan kemampuannya mendeteksi hampir 100% pelanggan 
yang akan churn. Contract Length adalah faktor paling dominan 
(Cramér's V = 0.43) — pelanggan kontrak bulanan memiliki churn rate 
100%. Dengan menerapkan model ini, perusahaan berpotensi menghasilkan 
net profit bersih $22.495.000 dengan ROI ~$10 per $1 yang 
diinvestasikan untuk program retensi.

---

## ✅ Kelebihan Model
- Recall = 1.000 — tidak ada pelanggan churn yang terlewat
- 4 model diuji dan dibandingkan dengan metrik tepat (Recall)
- SMOTE hanya pada train data — tidak ada data leakage
- Validasi statistik dengan Chi-Square, Cramér's V, Welch T-Test
- Simulasi profitabilitas konkret dengan asumsi biaya transparan

## ⚠️ Peluang Peningkatan
- Recall 1.000 bisa mengindikasikan overfitting — perlu k-fold CV
- Tambahkan Random Forest / XGBoost untuk perbandingan lebih robust
- Feature importance Decision Tree belum divisualisasikan
- Threshold tuning pada model lain untuk optimalkan precision-recall
- Simulasi profitabilitas bisa diperkaya dengan sensitivity analysis

---

## 💼 Rekomendasi Strategis
| Segmen | Rekomendasi |
|--------|-------------|
| Kontrak Bulanan (churn 100%) | Tawarkan insentif upgrade ke kontrak tahunan/kuartalan |
| Pelanggan Baru (tenure pendek) | Program onboarding yang kuat di 3 bulan pertama |
| Support Calls tinggi | Tingkatkan kualitas customer service & resolusi masalah |
| Payment Delay tinggi | Tawarkan fleksibilitas pembayaran & cicilan |
| Low Usage Frequency | Kirim notifikasi & kampanye re-engagement |
| Pelanggan Perempuan (churn 66%) | Personalisasi program retensi untuk segmen ini |

---

## 🔧 Tools & Libraries
\`\`\`python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from scipy.stats import chi2_contingency, ttest_ind
import joblib
\`\`\`

---

## 📁 File Structure
- \`notebook/\` — Google Colab (.ipynb)

---

*Lhedya Monica Ismon | Machine Learning Bootcamp | Dibimbing.id*