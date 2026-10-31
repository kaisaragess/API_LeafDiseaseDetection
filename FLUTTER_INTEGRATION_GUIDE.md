# Panduan Integrasi Mobile App (Flutter)
## Backend REST API Klasifikasi Penyakit Daun Padi

Panduan ini dibuat khusus untuk mempermudah rekan developer Flutter dalam mengonsumsi seluruh endpoint Backend API ini secara cepat dan tepat.

---

## 1. Konfigurasi Base URL

Saat pengujian lokal dari aplikasi Flutter, perhatikan alamat IP yang digunakan:

| Target Running Flutter | Base URL yang Digunakan | Penjelasan |
| :--- | :--- | :--- |
| **Android Emulator** | `http://10.0.2.2:8000/api/v1` | `10.0.2.2` adalah alias host mesin laptop/PC dari dalam emulator Android. |
| **iOS Simulator** | `http://localhost:8000/api/v1` | iOS Simulator berbagi jaringan langsung dengan macOS/PC. |
| **HP Fisik (Real Device Android/iOS)** | `http://192.168.x.x:8000/api/v1` | Gunakan alamat IPv4 laptop kamu (pastikan HP dan laptop terhubung di Wi-Fi yang sama). |
| **Server Cloud / Hugging Face** | `https://<space-name>.hf.space/api/v1` | URL publik HTTPS jika backend sudah di-deploy. |

---

## 2. Dependensi Flutter (`pubspec.yaml`)

Tambahkan dependensi berikut pada proyek Flutter Anda:

```yaml
dependencies:
  flutter:
    sdk: flutter
  http: ^1.2.0          # Untuk HTTP Request multipart
  image_picker: ^1.1.0  # Untuk mengambil foto dari Kamera / Galeri
```

Jalankan `flutter pub get`.

---

## 3. Model Data Dart (Siap Pakai)

Buat berkas `lib/models/rice_disease_model.dart`:

```dart
import 'dart:convert';
import 'dart:typed_data';

class PredictResponse {
  final String status;
  final PredictionMeta meta;
  final PredictionDetail prediction;
  final FeatureSummary featureSummary;
  final Map<String, String> visualizationsBase64;
  final String disclaimer;

  PredictResponse({
    required this.status,
    required this.meta,
    required this.prediction,
    required this.featureSummary,
    required this.visualizationsBase64,
    required this.disclaimer,
  });

  factory PredictResponse.fromJson(Map<String, dynamic> json) {
    return PredictResponse(
      status: json['status'] ?? '',
      meta: PredictionMeta.fromJson(json['meta'] ?? {}),
      prediction: PredictionDetail.fromJson(json['prediction'] ?? {}),
      featureSummary: FeatureSummary.fromJson(json['feature_summary'] ?? {}),
      visualizationsBase64: Map<String, String>.from(json['visualizations_base64'] ?? {}),
      disclaimer: json['disclaimer'] ?? '',
    );
  }
}

class PredictionMeta {
  final String pipelineUsed;
  final String classifierUsed;
  final double preprocessingTimeMs;
  final double predictionTimeMs;
  final double modelCvMacroF1;

  PredictionMeta({
    required this.pipelineUsed,
    required this.classifierUsed,
    required this.preprocessingTimeMs,
    required this.predictionTimeMs,
    required this.modelCvMacroF1,
  });

  factory PredictionMeta.fromJson(Map<String, dynamic> json) {
    return PredictionMeta(
      pipelineUsed: json['pipeline_used'] ?? '',
      classifierUsed: json['classifier_used'] ?? '',
      preprocessingTimeMs: (json['preprocessing_time_ms'] as num?)?.toDouble() ?? 0.0,
      predictionTimeMs: (json['prediction_time_ms'] as num?)?.toDouble() ?? 0.0,
      modelCvMacroF1: (json['model_cv_macro_f1'] as num?)?.toDouble() ?? 0.0,
    );
  }
}

class PredictionDetail {
  final String predictedClass;
  final double confidenceScore;
  final Map<String, double> probabilities;

  PredictionDetail({
    required this.predictedClass,
    required this.confidenceScore,
    required this.probabilities,
  });

  factory PredictionDetail.fromJson(Map<String, dynamic> json) {
    final rawProbs = json['probabilities'] as Map<String, dynamic>? ?? {};
    final probs = rawProbs.map((key, value) => MapEntry(key, (value as num).toDouble()));

    return PredictionDetail(
      predictedClass: json['predicted_class'] ?? 'Tidak Diketahui',
      confidenceScore: (json['confidence_score'] as num?)?.toDouble() ?? 0.0,
      probabilities: probs,
    );
  }
}

class FeatureSummary {
  final double leafAreaRatio;
  final double lesionAreaRatio;
  final double lesionDensityPer1kPx;
  final int estimatedLesionBlobs;

  FeatureSummary({
    required this.leafAreaRatio,
    required this.lesionAreaRatio,
    required this.lesionDensityPer1kPx,
    required this.estimatedLesionBlobs,
  });

  factory FeatureSummary.fromJson(Map<String, dynamic> json) {
    return FeatureSummary(
      leafAreaRatio: (json['leaf_area_ratio'] as num?)?.toDouble() ?? 0.0,
      lesionAreaRatio: (json['lesion_area_ratio'] as num?)?.toDouble() ?? 0.0,
      lesionDensityPer1kPx: (json['lesion_density_per_1k_px'] as num?)?.toDouble() ?? 0.0,
      estimatedLesionBlobs: json['estimated_lesion_blobs'] ?? 0,
    );
  }
}

// Exception khusus jika objek yang difoto bukan daun padi (Error 422)
class NonLeafException implements Exception {
  final String message;
  final double leafAreaRatio;

  NonLeafException(this.message, this.leafAreaRatio);

  @override
  String toString() => message;
}
```

---

## 4. API Service Flutter (`lib/services/api_service.dart`)

```dart
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import '../models/rice_disease_model.dart';

class RiceLeafApiService {
  // Ubah Base URL sesuai target perangkat (lihat bagian 1)
  static const String baseUrl = 'http://10.0.2.2:8000/api/v1';

  /// 1. Health Check
  Future<bool> checkHealth() async {
    try {
      final res = await http.get(Uri.parse('$baseUrl/health')).timeout(const Duration(seconds: 5));
      return res.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  /// 2. Diagnosis Daun Padi (POST /predict)
  Future<PredictResponse> predictLeafDisease({
    required File imageFile,
    String pipelineMode = 'P1',       // Pilihan: 'B0', 'B1', 'B2', 'P1'
    String classifier = 'SVM',        // Pilihan: 'KNN', 'SVM', 'RandomForest', 'ExtraTrees', 'Ensemble_Voting'
    double brightnessShift = 0.0,     // Rentang: -0.20 s.d. +0.20
    double saturationShift = 0.0,     // Rentang: -0.15 s.d. +0.15
    bool applyBlur = false,
  }) async {
    final uri = Uri.parse('$baseUrl/predict');
    final request = http.MultipartRequest('POST', uri);

    // Lampirkan file foto
    request.files.add(await http.MultipartFile.fromPath('image', imageFile.path));

    // Parameter konfigurasi
    request.fields['pipeline_mode'] = pipelineMode;
    request.fields['classifier'] = classifier;
    request.fields['brightness_shift'] = brightnessShift.toString();
    request.fields['saturation_shift'] = saturationShift.toString();
    request.fields['apply_blur'] = applyBlur.toString();

    final streamedResponse = await request.send().timeout(const Duration(seconds: 20));
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      final jsonMap = jsonDecode(response.body);
      return PredictResponse.fromJson(jsonMap);
    } else if (response.statusCode == 422) {
      // Out-of-Distribution Guard: Bukan Daun Padi
      final errorJson = jsonDecode(response.body);
      throw NonLeafException(
        errorJson['message'] ?? 'Objek bukan daun padi.',
        (errorJson['leaf_area_ratio'] as num?)?.toDouble() ?? 0.0,
      );
    } else {
      throw Exception('Gagal melakukan diagnosis: ${response.statusCode} - ${response.body}');
    }
  }

  /// 3. Komparasi 4 Pipeline Sekaligus (POST /compare-pipelines)
  Future<Map<String, dynamic>> comparePipelines({
    required File imageFile,
    String classifier = 'SVM',
    double brightnessShift = -0.20,
  }) async {
    final uri = Uri.parse('$baseUrl/compare-pipelines');
    final request = http.MultipartRequest('POST', uri);

    request.files.add(await http.MultipartFile.fromPath('image', imageFile.path));
    request.fields['classifier'] = classifier;
    request.fields['brightness_shift'] = brightnessShift.toString();

    final streamedResponse = await request.send().timeout(const Duration(seconds: 25));
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Gagal komparasi pipeline: ${response.statusCode}');
    }
  }

  /// 4. Dashboard Metrik Ilmiah (GET /experiment-metrics)
  Future<Map<String, dynamic>> getExperimentMetrics() async {
    final response = await http.get(Uri.parse('$baseUrl/experiment-metrics'));
    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Gagal memuat metrik');
    }
  }
}
```

---

## 5. Cara Menampilkan Gambar Base64 di Widget Flutter

Backend mengembalikan 6 visualisasi proses pengolahan citra dalam format Data URI Base64 (`data:image/jpeg;base64,...`).

Gunakan helper widget berikut untuk menampilkannya langsung di layar aplikasi:

```dart
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/material.dart';

Widget buildBase64Image(String dataUriString, {double height = 200, double width = 100}) {
  try {
    // Bersihkan prefix data URI jika ada (misal "data:image/jpeg;base64,")
    final base64Clean = dataUriString.contains(',')
        ? dataUriString.split(',')[1]
        : dataUriString;

    final Uint8List imageBytes = base64Decode(base64Clean);

    return ClipRRect(
      borderRadius: BorderRadius.circular(8),
      child: Image.memory(
        imageBytes,
        height: height,
        width: width,
        fit: BoxFit.cover,
      ),
    );
  } catch (e) {
    return Container(
      height: height,
      width: width,
      color: Colors.grey[300],
      child: const Icon(Icons.broken_image, color: Colors.grey),
    );
  }
}
```

### Menampilkan Carousel 6 Tahapan Pemrosesan:
```dart
// Di dalam halaman hasil diagnosis (StatefulWidget):
Widget buildVisualizationStages(Map<String, String> stages) {
  final stageTitles = {
    '1_resized_input': '1. Resized Input (384x128)',
    '2_gray_world': '2. Gray World (Color Constancy)',
    '3_clahe_lab': '3. CLAHE Contrast (Lab Space)',
    '4_leaf_mask_morphology': '4. Masking Daun Padi',
    '5_lesion_mask': '5. Masking Bercak Penyakit',
    '6_final_segmented': '6. Segmentasi Akhir',
  };

  return SizedBox(
    height: 240,
    child: ListView(
      scrollDirection: Axis.horizontal,
      children: stages.entries.map((entry) {
        return Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8.0),
          child: Column(
            children: [
              buildBase64Image(entry.value, height: 180, width: 90),
              const SizedBox(height: 6),
              Text(
                stageTitles[entry.key] ?? entry.key,
                style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold),
              ),
            ],
          ),
        );
      }).toList(),
    ),
  );
}
```

---

## 6. Contoh Menangani Error Non-Daun (Status 422) di UI

Jika pengguna memfoto benda sembarang (bukan daun padi), tampilkan dialog peringatan:

```dart
try {
  final result = await apiService.predictLeafDisease(imageFile: pickedFile);
  // Navigasi ke Halaman Hasil Diagnosis
} on NonLeafException catch (e) {
  // Tangani otomatis error 422
  showDialog(
    context: context,
    builder: (ctx) => AlertDialog(
      title: const Text('⚠️ Objek Bukan Daun Padi'),
      content: Text(e.message),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(ctx),
          child: const Text('Foto Ulang'),
        ),
      ],
    ),
  );
} catch (e) {
  // Tangani error umum/koneksi
  ScaffoldMessenger.of(context).showSnackBar(
    SnackBar(content: Text('Terjadi kesalahan: $e')),
  );
}
```

---

## 7. Catatan Penting untuk Presentasi / Uji Demo

1. **Pemilihan Pipeline & Model Fleksibel:**  
   UI Flutter bisa menyediakan Dropdown untuk `Pipeline` (`B0`, `B1`, `B2`, `P1`) dan `Model` (`KNN`, `SVM`, `RandomForest`, `ExtraTrees`, `Ensemble_Voting`).
2. **Slider Gangguan Pencahayaan:**  
   Bisa menambahkan Slider `Brightness (-20% s.d. +20%)` di UI untuk menunjukkan ketahanan model `P1` dibanding baseline saat demo di depan dosen.
3. **Telemetry Waktu Eksekusi:**  
   Waktu komputasi `meta.preprocessingTimeMs` dan `meta.predictionTimeMs` bisa ditampilkan di bawah hasil diagnosis (rata-rata total waktu hanya $\approx 30\text{ ms}$).
