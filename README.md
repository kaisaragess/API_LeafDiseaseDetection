# Backend REST API: Sistem Diagnosis & Komparasi Pipeline Citra Penyakit Daun Padi

**Proyek:** Kelompok 1 — Mata Kuliah Pengolahan Citra Digital  
**Dosen Pengampu:** Rangga Pebrianto, M.Kom.  
**Target Publikasi:** Jurnal Algoritma, Logika dan Komputasi (J-ALU) (SINTA 5)  
**Versi:** 1.0.0

---

## 1. Ringkasan Eksekutif

Backend REST API ini dibangun menggunakan **FastAPI**, **OpenCV**, dan **Scikit-Learn** untuk melayani klasifikasi penyakit daun padi (*Bacterial Leaf Blight*, *Brown Spot*, dan *Leaf Smut*). Sistem mengimplementasikan:
1. **Multi-Pipeline Scenarios:** `B0` (Raw Resize), `B1` (Gray World), `B2` (CLAHE Lab), dan `P1` (Proposed: Gray World + CLAHE Lab).
2. **5 Machine Learning Estimators:** `KNN`, `SVM`, `RandomForest`, `ExtraTrees`, dan `Ensemble_Voting`.
3. **Step-by-Step Base64 Image Returns:** Mengembalikan visualisasi setiap tahapan pengolahan citra (*Resize -> Gray World -> CLAHE -> Leaf Mask -> Lesion Mask -> Segmented Leaf*).
4. **Simulator Gangguan Pencahayaan:** Uji ketahanan on-the-fly dengan parameter *brightness*, *saturation*, dan *Gaussian blur*.
5. **Out-of-Distribution Guard:** Menolak otomatis foto non-daun jika rasio area daun < 8% (`422 Unprocessable Entity`).
6. **Dashboard Metrik Ilmiah:** Menyajikan hasil 25-fold Cross-Validation dan uji signifikansi Friedman & Wilcoxon-Holm.

---

## 2. Struktur Direktori

```text
mobile_programming/
├── artifacts/
│   ├── rice_leaf_backend_bundle.joblib  # 20 model, label encoder, scaler, konfigurasi
│   └── results/                         # Arsip CSV hasil eksperimen Colab
│       ├── ringkasan_95CI_macro_f1.csv
│       ├── uji_statistik_friedman_wilcoxon_lengkap.csv
│       └── hasil_robustness_test.csv
├── app/
│   ├── __init__.py
│   ├── config.py                        # Path artefak, ambang batas daun, konstanta
│   ├── schemas.py                       # Validasi Pydantic skema request & response
│   ├── image_pipeline.py                # Preprocessing GW, CLAHE, Segmentasi, GLCM
│   └── main.py                          # Routing FastAPI & penanganan error
├── tests/
│   ├── __init__.py
│   ├── test_health.py                   # Pengujian endpoint /health
│   ├── test_metrics.py                  # Pengujian endpoint /experiment-metrics
│   ├── test_guard.py                    # Pengujian penolakan non-daun (422)
│   └── test_prediction.py               # Pengujian /predict & /compare-pipelines
├── Dockerfile                           # Container image produksi
├── requirements.txt                     # Daftar dependensi Python
└── README.md                            # Panduan eksekusi & dokumentasi API
```

## 3. Integrasi Aplikasi Mobile (Flutter)

Bagi rekan pengembang aplikasi mobile (**Flutter**), panduan integrasi lengkap beserta contoh kode **Dart Service**, **Data Models**, dan cara menampilkan **6 visualisasi Base64** telah disediakan pada:
👉 **[FLUTTER_INTEGRATION_GUIDE.md](file:///e:/mobile_programming/FLUTTER_INTEGRATION_GUIDE.md)**

### Catatan Base URL untuk Mobile:
- **Android Emulator:** `http://10.0.2.2:8000/api/v1`
- **iOS Simulator:** `http://localhost:8000/api/v1`
- **Real Device (HP Fisik):** `http://<IP-Laptop-Anda>:8000/api/v1`

---

## 4. Instalasi & Menjalankan Lokal

### 3.1 Prasyarat
- Python 3.10 - 3.13
- Virtual Environment

### 3.2 Langkah Eksekusi

```bash
# 1. Masuk ke direktori proyek
cd e:/mobile_programming

# 2. Aktifkan virtual environment
.\venv\Scripts\Activate.ps1

# 3. Pasang dependensi
pip install -r requirements.txt

# 4. Jalankan server Uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Dokumentasi OpenAPI / Swagger UI interaktif dapat diakses pada:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## 5. Dokumentasi Endpoint API

| Method | Endpoint | Deskripsi | Status Code |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Health check & verifikasi kesiapan model RAM | `200 OK` |
| `POST` | `/api/v1/predict` | Diagnosis citra tunggal + visualisasi Base64 | `200 OK` / `422 Unprocessable` |
| `POST` | `/api/v1/compare-pipelines` | Komparasi stres 4 pipeline pada citra bersih vs terganggu | `200 OK` |
| `GET` | `/api/v1/experiment-metrics` | Dashboard ringkasan 25-fold CV & uji Friedman/Wilcoxon | `200 OK` |

---

## 6. Menjalankan Automated Testing

Pengujian otomatis dibangun dengan **pytest** dan **HTTPX**:

```bash
# Menjalankan seluruh pengujian unit dan integrasi
pytest tests/ -v
```

---

## 7. Deployment Menggunakan Docker

```bash
# Build image
docker build -t rice-leaf-backend:latest .

# Run container
docker run -d -p 7860:7860 --name rice-leaf-api rice-leaf-backend:latest
```
Akses API pada `http://localhost:7860/docs`.
Port `7860` kompatibel langsung dengan **Hugging Face Spaces (Docker SDK)**.
