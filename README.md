# Analisis SVM pada Breast Cancer Wisconsin

Proyek Big Data Kelompok 4. Skrip ini menganalisis klasifikasi kanker payudara (ganas vs jinak) dengan **Support Vector Machine (SVM)**, lalu membandingkannya dengan model lain.

## Isi folder

| File | Keterangan |
|---|---|
| `analisis_svm.py` | Skrip utama: tuning, evaluasi, dan uji statistik |
| `breast_cancer_wisconsin.csv` | Salinan dataset (skrip memuat data dari `sklearn.datasets.load_breast_cancer`) |
| `info_svm.json` | Hasil angka yang ditulis skrip (parameter terbaik, metrik, dll.) |
| `fig/` | Grafik hasil (`s01` sampai `s11`) |

## Alur analisis

1. **Tuning SVM**: GridSearchCV (10-fold stratified, skor F1) atas kernel `linear/rbf/poly/sigmoid`, `C`, dan `gamma`.
2. **Evaluasi data uji** (split 80/20, stratified, `random_state=42`): perbandingan dengan Logistic Regression, KNN, Decision Tree, Random Forest, Naive Bayes, dan Dummy baseline. Termasuk confusion matrix, ROC/PR, analisis threshold, learning curve, dan decision boundary (PCA).
3. **Pengujian**: Repeated 10-fold CV (10x) pada seluruh data, uji Wilcoxon signed-rank, dan koreksi Holm.

Parameter terbaik hasil tuning: kernel `rbf`, `C=10`, `gamma=scale` (F1 CV ≈ 0,965).

## Cara menjalankan

Butuh Python 3.12 (versi yang dipakai saat dikembangkan).

```powershell
# 1. Buat dan aktifkan virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Pasang dependensi
pip install numpy pandas matplotlib seaborn scipy scikit-learn

# 3. Siapkan folder keluaran grafik (jika belum ada)
mkdir fig

# 4. Jalankan
python analisis_svm.py
```

Keluaran: grafik baru di `fig/` dan `info_svm.json` diperbarui. Grid search memakai semua core CPU (`n_jobs=-1`), jadi bisa memakan waktu beberapa menit.

## Catatan Windows

Jika muncul error `An Application Control policy has blocked this file` saat mengimpor pandas, penyebabnya Smart App Control memblokir file `.pyd` di venv. Matikan Smart App Control di Windows Security, atau pakai Anaconda/WSL/Google Colab.
