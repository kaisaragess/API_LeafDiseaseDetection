# Software Requirements Specification (SRS)
## Backend REST API: Sistem Diagnosis & Komparasi Pipeline Citra Penyakit Daun Padi
**Proyek:** Kelompok 1 — Mata Kuliah Pengolahan Citra Digital  
**Dosen Pengampu:** Rangga Pebrianto, M.Kom.  
**Target Publikasi:** Jurnal Algoritma, Logika dan Komputasi (J-ALU) (SINTA 5)  
**Versi Dokumen:** 1.0.0  
**Status:** Siap Implementasi  

---

## 1. Pendahuluan & Ringkasan Eksekutif

### 1.1 Tujuan Sistem
Backend REST API ini dirancang untuk melayani inferensi diagnosis klasifikasi tiga penyakit daun padi (*Bacterial Leaf Blight*, *Brown Spot*, dan *Leaf Smut*) berbasis tangkapan kamera/foto. Sistem ini dibangun dengan fleksibilitas eksperimental tingkat tinggi untuk mendemonstrasikan kontribusi ilmiah pengolahan citra:
1. Mendukung pemilihan dinamis **4 skenario pipeline pengolahan citra** (`B0`, `B1`, `B2`, `P1`).
2. Mendukung pemilihan dinamis **5 algoritma machine learning** (`KNN`, `SVM`, `RandomForest`, `ExtraTrees`, `Ensemble_Voting`).
3. Mengembalikan visualisasi tahapan pemrosesan citra secara langsung (*step-by-step Base64*) untuk transparansi analisis.
4. Menyediakan simulator gangguan pencahayaan (*Robustness Test on-the-fly*).
5. Menyediakan metrik performa ilmiah 25-fold CV serta uji statistik (Friedman & Wilcoxon-Holm).

### 1.2 Batasan Sistem & Etika Data
- Sistem berstatus **purwarupa akademik (*academic prototype*)** yang dilatih pada *Rice Leaf Diseases Dataset* (120 citra pada latar belakang seragam).
- Sistem **bukan alat diagnosis lapangan otonom**. Setiap respons diagnosis wajib menyertakan peringatan (*disclaimer*) ilmiah.

---

## 2. Rekomendasi Tech Stack & Arsitektur

### 2.1 Pilihan Teknologi Rekomendasi

| Lapisan / Komponen | Teknologi Terpilih | Justifikasi Teknis |
| :--- | :--- | :--- |
| **Framework Web** | **FastAPI (Python 3.10+)** | Performa asinkronus tinggi, auto-generate dokumentasi Swagger/OpenAPI interaktif di `/docs`, dan validasi skema otomatis via Pydantic v2. |
| **Server ASGI** | **Uvicorn (Standard Worker)** | Server ASGI berbasis `uvloop` dengan latensi rendah dan konsumsi memori minimal. |
| **Image Processing Engine** | **OpenCV (Headless) + Scikit-Image** | `opencv-python-headless` sangat ringan di lingkungan Linux/Docker tanpa dependensi GUI. `skimage` digunakan untuk ekstraksi GLCM presisi. |
| **Model Inference Engine** | **Scikit-Learn + Joblib** | Pemuatan 20 model dalam satu bundel artefak (`rice_leaf_backend_bundle.joblib`) berukuran < 5 MB dengan waktu muat RAM < 500 ms. |
| **Data Serialization** | **Pydantic v2** | Serialisasi data JSON cepat dengan type-hinting ketat untuk mencegah runtime error. |
| **Containerization** | **Docker (Alpine / Slim Debian)** | Memastikan lingkungan Linux yang sepenuhnya dapat direproduksi (*reproducible*), portabel ke cloud mana pun. |
| **Deployment Target** | **Hugging Face Spaces (Docker SDK) / Render** | Gratis, mendukung container Docker dengan RAM 16 GB, dan langsung menyediakan HTTPS publik untuk diakses aplikasi mobile/web. |

### 2.2 Arsitektur Alur Sistem

```
[Kamera HP / Web Client]
          │
          ▼ (Multipart Form-Data: Foto Daun + Opsi Pipeline/Model)
[FastAPI Gateway]
   ├── 1. Validasi File & Ukuran (< 10 MB, JPG/PNG)
   ├── 2. Simulasi Gangguan (Opsional: Brightness, Saturation, Blur)
   ├── 3. Engine Pengolahan Citra (Resize 384x128 -> GW -> CLAHE -> Segmentasi)
   │        ├── Validasi Keberadaan Daun (Leaf Area Ratio >= 8%)
   │        └── Ekstraksi Vektor Fitur (142 Fitur: HSV + Lab + GLCM + Geometri)
   ├── 4. Routing Model Artefak (Pilih 1 dari 20 Estimator di RAM)
   └── 5. Agregasi Respons (Diagnosis, Confidence, Visualisasi Base64, Waktu Eksekusi)
          │
          ▼ (JSON Response + Data URI Image Base64)
[Tampilan Hasil Diagnosis & Visualisasi Tahap Citra]
```

---

## 3. Spesifikasi Kebutuhan Fungsional (Functional Requirements)

### FR-01: Health & Runtime Metadata Check
- Sistem harus menyediakan endpoint `GET /api/v1/health` untuk memverifikasi kesiapan server, ketersediaan model `.joblib`, dan versi modul Python.

### FR-02: Single-Image Diagnosis & Visual Inspection
- Sistem harus menerima berkas foto (`.jpg`, `.jpeg`, `.png`) bersama pilihan `pipeline_mode` (`B0`, `B1`, `B2`, `P1`) dan `classifier` (`KNN`, `SVM`, `RandomForest`, `ExtraTrees`, `Ensemble_Voting`).
- Sistem harus mengembalikan:
  - Label kelas penyakit hasil prediksi.
  - Skor keyakinan (*confidence score*) dan distribusi probabilitas seluruh kelas.
  - Ringkasan fitur geometri (rasio daun dan rasio bercak).
  - Gambar Base64 untuk 6 tahapan pemrosesan citra.
  - Durasi waktu *preprocessing* dan *prediction* dalam milidetik.

### FR-03: Out-of-Distribution / Non-Leaf Guard
- Sistem harus menolak foto yang tidak memiliki komponen daun padi dengan kode status `422 Unprocessable Entity` jika rasio luas daun (`leaf_area_ratio`) terhitung di bawah `0.08` (8%).

### FR-04: Multi-Pipeline Comparative Stress Test
- Sistem harus menyediakan endpoint `POST /api/v1/compare-pipelines` untuk memproses 1 citra yang sama secara paralel ke dalam 4 pipeline (`B0`, `B1`, `B2`, `P1`) pada kondisi citra bersih vs citra terganggu pencahayaan, guna membuktikan hipotesis bahwa `P1` memiliki penurunan (*drop*) terkecil.

### FR-05: Experiment Metrics & Statistical Dashboard
- Sistem harus menyediakan endpoint `GET /api/v1/experiment-metrics` yang membaca dan menyajikan ringkasan hasil 25-fold CV, 95% Confidence Interval, serta signifikansi uji Friedman dan Wilcoxon-Holm dari berkas CSV hasil eksperimen.

---

## 4. Spesifikasi Kebutuhan Non-Fungsional (Non-Functional Requirements)

1. **Kecepatan Inferensi (Performance):**  
   - Waktu pemrosesan citra dan ekstraksi fitur per permintaan tidak boleh melebihi **80 milidetik**.
   - Waktu inferensi model machine learning per permintaan tidak boleh melebihi **15 milidetik**.
2. **Ketersediaan Memori (Resource Footprint):**  
   - Penggunaan RAM server saat seluruh 20 model dimuat tidak boleh melebihi **350 MB**, memungkinkan dijalankan pada paket VPS/Cloud gratisan terkecil (512 MB RAM).
3. **Reproducibility & Thread Safety:**  
   - Operasi prediksi bersifat *stateless* dan *read-only* terhadap model artefak, menjamin aman dieksekusi secara konkuren.
4. **Keamanan (Security):**  
   - Middleware CORS dikonfigurasi untuk menerima permintaan dari domain aplikasi pengguna.
   - Pengecekan *Magic Byte* pada berkas unggahan untuk memverifikasi integritas file citra asli.

---

## 5. Kontrak Antarmuka API (API Endpoint Specifications)

### 5.1 Endpoint `POST /api/v1/predict`
Melakukan diagnosis penyakit daun padi berdasarkan konfigurasi pipeline dan model pilihan.

#### Request Headers
```http
Content-Type: multipart/form-data
```

#### Request Body (Form-Data)
| Field | Tipe Data | Wajib | Nilai Baku | Validasi / Deskripsi |
| :--- | :--- | :-: | :--- | :--- |
| `image` | `Binary File` | Ya | - | File gambar (.jpg, .jpeg, .png), maks. 10 MB. |
| `pipeline_mode` | `String` | Tidak | `"P1"` | Enum: `["B0", "B1", "B2", "P1"]` |
| `classifier` | `String` | Tidak | `"SVM"` | Enum: `["KNN", "SVM", "RandomForest", "ExtraTrees", "Ensemble_Voting"]` |
| `brightness_shift`| `Float` | Tidak | `0.0` | Rentang: `-0.20` s.d. `+0.20` ($\pm 20\%$) |
| `saturation_shift`| `Float` | Tidak | `0.0` | Rentang: `-0.15` s.d. `+0.15` ($\pm 15\%$) |
| `apply_blur` | `Boolean` | Tidak | `false` | Penerapan filter Gaussian Blur ringan |

#### Response: `200 OK` (Berhasil)
```json
{
  "status": "success",
  "meta": {
    "pipeline_used": "P1",
    "classifier_used": "SVM",
    "model_cv_macro_f1": 0.8965,
    "total_features_extracted": 142,
    "preprocessing_time_ms": 42.15,
    "prediction_time_ms": 2.10
  },
  "perturbation_applied": {
    "brightness_shift": 0.0,
    "saturation_shift": 0.0,
    "apply_blur": false
  },
  "prediction": {
    "predicted_class": "Brown spot",
    "confidence_score": 0.9245,
    "probabilities": {
      "Bacterial leaf blight": 0.0341,
      "Brown spot": 0.9245,
      "Leaf smut": 0.0414
    }
  },
  "feature_summary": {
    "leaf_area_ratio": 0.4521,
    "lesion_area_ratio": 0.1382,
    "lesion_density_per_1k_px": 8.41
  },
  "visualizations_base64": {
    "1_resized_input": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "2_gray_world": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "3_clahe_lab": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "4_leaf_mask_morphology": "data:image/png;base64,iVBORw0KGgo...",
    "5_lesion_mask": "data:image/png;base64,iVBORw0KGgo...",
    "6_final_segmented": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
  },
  "disclaimer": "Sistem ini merupakan purwarupa akademik pada data terbatas (120 citra) dan bukan alat diagnosis lapangan siap pakai."
}
```

#### Response: `422 Unprocessable Entity` (Objek Bukan Daun)
```json
{
  "status": "error",
  "error_code": "INVALID_LEAF_OBJECT",
  "message": "Area daun padi tidak terdeteksi secara memadai (rasio daun: 3.4%). Pastikan fokus kamera mengarah pada daun padi.",
  "leaf_area_ratio": 0.034
}
```

---

### 5.2 Endpoint `POST /api/v1/compare-pipelines`
Menguji performa kestabilan prediksi 4 pipeline sekaligus pada kondisi bersih vs terganggu pencahayaan.

#### Request Body (Form-Data)
| Field | Tipe Data | Wajib | Nilai Baku | Deskripsi |
| :--- | :--- | :-: | :--- | :--- |
| `image` | `Binary File` | Ya | - | Berkas citra daun padi |
| `classifier` | `String` | Tidak | `"SVM"` | Model penguji |
| `brightness_shift`| `Float` | Tidak | `-0.20` | Uji penurunan cahaya -20% |

#### Response: `200 OK`
```json
{
  "status": "success",
  "classifier_tested": "SVM",
  "perturbation_condition": "Brightness -20%",
  "comparisons": {
    "B0": {
      "clean_class": "Brown spot",
      "clean_confidence": 0.761,
      "perturbed_class": "Leaf smut",
      "perturbed_confidence": 0.528,
      "is_consistent": false,
      "confidence_drop": -0.233
    },
    "B1": {
      "clean_class": "Brown spot",
      "clean_confidence": 0.812,
      "perturbed_class": "Brown spot",
      "perturbed_confidence": 0.749,
      "is_consistent": true,
      "confidence_drop": -0.063
    },
    "B2": {
      "clean_class": "Brown spot",
      "clean_confidence": 0.834,
      "perturbed_class": "Brown spot",
      "perturbed_confidence": 0.725,
      "is_consistent": true,
      "confidence_drop": -0.109
    },
    "P1": {
      "clean_class": "Brown spot",
      "clean_confidence": 0.924,
      "perturbed_class": "Brown spot",
      "perturbed_confidence": 0.898,
      "is_consistent": true,
      "confidence_drop": -0.026
    }
  },
  "scientific_insight": "Pipeline P1 menunjukkan kestabilan tertinggi dengan penurunan keyakinan terkecil (-2.6%) dibandingkan Baseline B0 (-23.3%)."
}
```

---

## 6. Struktur Direktori Proyek Backend

```text
rice-leaf-api/
├── artifacts/
│   ├── rice_leaf_backend_bundle.joblib  # 20 model, encoder, scaler, konfigurasi
│   └── results/                         # Arsip CSV eksperimen dari Colab
├── app/
│   ├── __init__.py
│   ├── config.py                        # Konfigurasi path dan threshold
│   ├── schemas.py                       # Skema request & response Pydantic
│   ├── image_pipeline.py                # OpenCV GW, CLAHE, Segmentasi, GLCM, Base64
│   └── main.py                          # Routing FastAPI & penanganan error
├── Dockerfile                           # Definisi container produksi
├── requirements.txt                     # Dependensi Python
└── README.md                            # Petunjuk eksekusi lokal & deployment
```

---

## 7. Strategi Deployment & Pengujian

1. **Pengujian Lokal:**
   ```bash
   pip install -r requirements.txt
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   Akses dokumentasi interaktif pada browser: `http://localhost:8000/docs`.

2. **Containerization (Docker):**
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   EXPOSE 7860
   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
   ```
   *(Port `7860` kompatibel langsung dengan Hugging Face Spaces)*.