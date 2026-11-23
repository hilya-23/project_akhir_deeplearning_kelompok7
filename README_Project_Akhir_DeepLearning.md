# Project Akhir Deep Learning: Deteksi dan Penghitungan Sel Darah

> **Eksperimen komputasional untuk membandingkan YOLOv8s dan Faster R-CNN ResNet-50 FPN pada dataset TXL-PBC.**

## 1. Gambaran Proyek

Proyek ini mengevaluasi dua pendekatan deteksi objek dalam mendeteksi sekaligus menghitung sel darah pada citra mikroskopis. Model yang dibandingkan adalah YOLOv8s (one-stage detector) dan Faster R-CNN ResNet-50 FPN (two-stage detector). Evaluasi tidak hanya melihat performa deteksi bounding box, tetapi juga kesalahan penghitungan per kelas serta aspek efisiensi komputasi.

Objek yang menjadi fokus adalah tiga kelas sel darah:

| ID kelas | Nama kelas | Keterangan |
|---:|---|---|
| 0 | WBC | White Blood Cell / sel darah putih |
| 1 | RBC | Red Blood Cell / sel darah merah |
| 2 | Platelet | Trombosit |

Penelitian ini merupakan evaluasi berbasis eksperimen komputasional dan **bukan sistem diagnosis atau pengganti pemeriksaan tenaga medis**.

## 2. Tujuan

- Membandingkan performa deteksi objek YOLOv8s dan Faster R-CNN ResNet-50 FPN.
- Mengevaluasi ketepatan penghitungan WBC, RBC, dan Platelet berdasarkan ground truth.
- Mengamati pengaruh konfigurasi eksperimen, termasuk resolusi input dan confidence threshold, terhadap hasil.
- Mengukur aspek efisiensi model, seperti waktu inferensi dan jumlah parameter, sesuai eksperimen di notebook.
- Menganalisis pola kesalahan penghitungan, termasuk over-counting dan under-counting.

## 3. Dataset

Dataset yang digunakan adalah **TXL-PBC**, dengan anotasi bounding box dalam format YOLO dan pembagian data train, validation, serta test yang telah disediakan.

### Struktur data yang diharapkan

Notebook mencari file `data.yaml` pada folder dataset `TXL-PBC`, serta menggunakan struktur citra dan label seperti berikut:

```text
TXL-PBC/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Setiap citra memiliki file label `.txt` dengan nama yang sesuai. Format label YOLO berisi lima nilai pada setiap baris:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Koordinat bounding box pada label YOLO bersifat ternormalisasi terhadap ukuran citra. Notebook mengubah anotasi menjadi format `xyxy` dalam satuan piksel untuk pipeline evaluasi bersama dan penggunaan model Faster R-CNN.

### Pembagian dataset

- `train`: digunakan untuk pelatihan model.
- `val`: digunakan untuk evaluasi selama pelatihan, pemilihan konfigurasi, dan pemantauan performa.
- `test`: digunakan untuk evaluasi akhir setelah konfigurasi ditentukan.

**Pembagian dataset resmi dipertahankan dan tidak diacak ulang.** Data test tidak digunakan sebagai dasar pemilihan konfigurasi model.

## 4. Pipeline Penelitian

Pipeline eksperimen dalam notebook terdiri dari beberapa tahap berikut.

1. **Konfigurasi lingkungan dan direktori** — menghubungkan Google Drive, menentukan lokasi dataset dan output, menetapkan seed, serta menyiapkan direktori kerja.
2. **Ekstraksi dan audit dataset** — mengekstrak dataset, memeriksa inventaris citra dan label, serta memeriksa file label yang hilang atau kosong.
3. **Exploratory Data Analysis (EDA)** — menganalisis distribusi kelas, jumlah objek, dan karakteristik dataset serta menyajikan visualisasi.
4. **Konversi anotasi** — mengubah label YOLO yang ternormalisasi menjadi bounding box `xyxy` dalam koordinat piksel.
5. **Verifikasi anotasi** — memeriksa konsistensi jumlah label dan bounding box, batas koordinat, serta visualisasi sampel citra untuk memastikan box sesuai objek.
6. **Pemeriksaan duplikasi atau kemiripan citra** — menggunakan perceptual hash (pHash) dan jarak Hamming untuk memeriksa citra yang identik atau sangat mirip.
7. **Persiapan data dan augmentasi** — menggunakan augmentasi yang dipilih dengan mempertimbangkan karakteristik citra sel darah.
8. **Pelatihan YOLOv8s** — melatih model, menjalankan konfigurasi eksperimen, dan menyimpan log serta bobot terbaik.
9. **Evaluasi model** — mengevaluasi prediksi menggunakan evaluator bersama agar hasil YOLOv8 dan Faster R-CNN dapat dibandingkan pada metrik yang sama.
10. **Eksperimen konfigurasi dan resolusi** — menguji konfigurasi yang ditentukan pada notebook, termasuk eksperimen resolusi input 640 dan 832.
11. **Eksperimen multi-seed** — menjalankan konfigurasi terpilih YOLOv8 dengan seed yang ditetapkan pada notebook (42, 43, dan 44).
12. **Pelatihan dan evaluasi Faster R-CNN** — melatih Faster R-CNN ResNet-50 FPN serta melakukan evaluasi pada split yang sesuai.
13. **Analisis kesalahan** — mengidentifikasi pola kesalahan penghitungan, termasuk over-counting, under-counting, dan kasus deviasi besar.
14. **Analisis performa dan efisiensi** — merangkum hasil deteksi, penghitungan, latensi, jumlah parameter, serta hubungan antara performa deteksi dan penghitungan.

## 5. Model yang Digunakan

### 5.1 YOLOv8s

YOLOv8s merupakan model deteksi objek dengan pendekatan **one-stage**, yaitu memprediksi objek dalam satu alur deteksi. Notebook menggunakan bobot awal `yolov8s.pt` dan konfigurasi pelatihan yang dicatat dalam notebook.

Konfigurasi dasar yang tercantum pada notebook:

| Parameter | Nilai |
|---|---:|
| Model awal | `yolov8s.pt` |
| Input image size | 640 |
| Epoch | 100 |
| Patience | 20 |
| Batch size | 16 |
| Optimizer | AdamW |
| Initial learning rate (`lr0`) | 0.001 |
| Final learning rate factor (`lrf`) | 0.01 |
| Weight decay | 0.0005 |
| Workers | 2 |
| AMP | Aktif |
| Deterministic | Aktif |
| Seed awal | 42 |

Konfigurasi ini merupakan konfigurasi dasar; konfigurasi pada run tertentu dapat berbeda jika notebook menerapkan override untuk eksperimen.

### 5.2 Faster R-CNN ResNet-50 FPN

Faster R-CNN merupakan model deteksi objek dengan pendekatan **two-stage**. Notebook menggunakan backbone ResNet-50 dengan Feature Pyramid Network (FPN) dan pretrained weights, kemudian mengganti head prediksi agar sesuai dengan kelas dataset.

Konfigurasi yang tercantum dalam pipeline:

| Parameter | Nilai |
|---|---:|
| Arsitektur | Faster R-CNN ResNet-50 FPN |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Weight decay | 0.0005 |
| Batch size | 4 |
| Scheduler | StepLR |
| Step size | 3 |
| Gamma | 0.1 |
| Jumlah keluaran | 4 (3 kelas objek + background) |

Label model Faster R-CNN disesuaikan dengan evaluator, termasuk penyesuaian label ke indeks kelas yang digunakan pada dataset.

## 6. Augmentasi Data

Augmentasi dipilih dengan mempertimbangkan karakteristik visual sel darah, terutama ukuran objek serta informasi warna dari proses pewarnaan.

| Augmentasi | Konfigurasi dasar | Pertimbangan |
|---|---|---|
| Horizontal flip | 0.5 | Orientasi horizontal bukan ciri pembeda utama sel |
| Vertical flip | 0.5 | Orientasi vertikal bukan ciri pembeda utama sel |
| Rotasi bebas | 0° | Rotasi non-90° dihindari untuk mengurangi perubahan box dan label noise |
| Hue | 0.01 | Perubahan hue dibuat kecil karena warna memiliki informasi visual |
| Saturation | 0.3 | Variasi ringan |
| Value/brightness | 0.3 | Variasi intensitas ringan |
| Translate | 0.1 | Variasi posisi objek |
| Scale | 0.3 | Variasi ukuran dengan perhatian pada objek kecil |
| Mosaic | 1.0 | Digunakan selama pelatihan dan ditutup pada 10 epoch terakhir |
| MixUp | 0.0 | Tidak digunakan |

## 7. Pemeriksaan Data dan Anotasi

Notebook melakukan sejumlah pemeriksaan sebelum pelatihan dan evaluasi:

- Inventaris jumlah citra dan file label per split.
- Pemeriksaan citra yang tidak memiliki file label dan label kosong.
- Pemeriksaan nama file yang sama antar-split.
- Parsing label YOLO dan konversi ke koordinat bounding box piksel.
- Pemeriksaan kesesuaian jumlah objek setelah konversi.
- Pemeriksaan box agar berada dalam batas citra.
- Verifikasi visual pada sampel citra, termasuk pemeriksaan 20 citra acak.
- Penyimpanan ground truth dalam format JSON untuk evaluator.
- Pemeriksaan kemiripan citra menggunakan pHash.

### Catatan pemeriksaan kemiripan

pHash digunakan untuk mendeteksi citra yang identik atau memiliki kemiripan visual tinggi berdasarkan jarak Hamming. Pemeriksaan ini bukan pengganti metadata asal slide atau apusan. Jika informasi asal apusan tidak tersedia, kemungkinan kemiripan antar-lapang pandang dari apusan yang sama tidak dapat dipastikan hanya dengan pHash.

## 8. Metrik Evaluasi

### 8.1 Metrik deteksi

- **mAP@0.5**: mean Average Precision pada IoU 0.5.
- **mAP@[0.5:0.95]**: Average Precision pada rentang IoU 0.5 sampai 0.95 dengan interval 0.05.
- **AP per kelas**: performa deteksi masing-masing kelas WBC, RBC, dan Platelet.

### 8.2 Metrik penghitungan

- **MAE (Mean Absolute Error)**: rata-rata nilai absolut selisih jumlah objek prediksi dengan jumlah ground truth.
- **MAPE (Mean Absolute Percentage Error)**: persentase rata-rata kesalahan relatif terhadap jumlah ground truth, sesuai implementasi evaluator.
- **Bias**: menunjukkan kecenderungan model melakukan prediksi jumlah lebih tinggi atau lebih rendah daripada ground truth.
- **Korelasi (`r`)**: digunakan dalam analisis hubungan jumlah prediksi dan jumlah aktual sesuai evaluator notebook.
- **Macro MAE**: ringkasan MAE antar kelas sebagaimana dihitung pada notebook.

### 8.3 Efisiensi

- **Latensi inferensi**: waktu yang dibutuhkan model untuk memproses citra.
- **Jumlah parameter**: jumlah parameter model sebagai salah satu indikator ukuran kompleksitas model.

Metrik deteksi dan penghitungan dianalisis secara terpisah. Model dengan mAP lebih tinggi belum tentu memiliki galat penghitungan yang lebih rendah.

## 9. Eksperimen dan Analisis

Notebook memuat rangkaian eksperimen yang meliputi:

- Run pelatihan YOLOv8s dan pencatatan konfigurasi serta status run.
- Evaluasi menggunakan evaluator bersama untuk membandingkan model dalam format metrik yang seragam.
- Eksperimen variasi konfigurasi yang dicatat pada notebook.
- Ablasi resolusi input 640 dan 832.
- Eksperimen multi-seed untuk konfigurasi YOLOv8 yang dipilih.
- Pelatihan dan evaluasi Faster R-CNN ResNet-50 FPN.
- Analisis korelasi antara mAP dan metrik penghitungan.
- Analisis kasus over-counting, under-counting, missed detection, dan false alarms.
- Analisis kasus dengan deviasi penghitungan terbesar dan visualisasi hasil.

Hasil numerik eksperimen tidak dicantumkan sebagai angka tetap pada README ini karena dapat berubah mengikuti run notebook. Gunakan tabel dan file evaluasi yang dihasilkan notebook sebagai hasil eksperimen aktual.

## 10. Cara Menjalankan Notebook

Notebook disiapkan untuk Google Colab.

### Persiapan

1. Siapkan akun Google yang memiliki akses ke Google Drive.
2. Unggah notebook ke Google Colab atau buka notebook melalui Colab.
3. Siapkan dataset TXL-PBC dalam bentuk ZIP pada Google Drive.
4. Pastikan struktur folder ZIP memuat folder dataset `TXL-PBC` dan file `data.yaml` yang dicari notebook.

### Konfigurasi notebook

Pada bagian **0.2 Konfigurasi — EDIT BAGIAN INI SAJA**, sesuaikan variabel berikut:

```python
ZIP_PATH = '/content/drive/MyDrive/Project_Akhir_DeepLearning/TXL-PBC_Dataset-master.zip'
PROJECT_DIR = '/content/drive/MyDrive/Hasil_Akhir'
OWNER = 'Khofifah dan Hilyatul Abidah'
SEED = 42
```

- `ZIP_PATH`: path file ZIP dataset di Google Drive.
- `PROJECT_DIR`: direktori permanen untuk menyimpan hasil eksperimen.
- `OWNER`: informasi pemilik/anggota yang dicantumkan dalam notebook.
- `SEED`: seed untuk proses acak yang dikonfigurasi.

### Menjalankan

1. Jalankan sel notebook dari bagian awal secara berurutan.
2. Izinkan notebook mengakses Google Drive.
3. Pastikan dataset berhasil diekstrak dan terdeteksi.
4. Jalankan audit data, EDA, verifikasi anotasi, dan pemeriksaan pHash.
5. Jalankan tahap pelatihan dan eksperimen sesuai urutan notebook.
6. Pastikan konfigurasi final dipilih berdasarkan data validation.
7. Jalankan evaluasi test sesuai protokol notebook setelah konfigurasi ditetapkan.
8. Periksa file log, tabel metrik, visualisasi, dan bobot yang tersimpan di `PROJECT_DIR`.

**Disarankan menggunakan runtime GPU Google Colab** untuk proses pelatihan agar waktu komputasi lebih efisien. Notebook dapat memerlukan waktu lebih lama jika dijalankan menggunakan CPU.

## 11. Direktori Output

Notebook menyiapkan folder output berikut pada `PROJECT_DIR`:

```text
Hasil_Akhir/
├── splits/
├── gt/
├── predictions/
├── weights/
├── figures/
└── logs/
```

| Folder | Fungsi |
|---|---|
| `splits/` | Daftar file dataset pada split train, validation, dan test serta informasi split |
| `gt/` | Ground truth yang telah dikonversi ke format evaluator |
| `predictions/` | Hasil prediksi model yang disimpan untuk evaluasi dan analisis |
| `weights/` | Bobot model yang disimpan dari run eksperimen |
| `figures/` | Grafik, visualisasi EDA, verifikasi, dan analisis |
| `logs/` | Log run, tabel hasil evaluasi, dan data pendukung eksperimen |

File yang disiapkan notebook antara lain:

- `logs/experiment_log.csv`: catatan run eksperimen, termasuk run yang gagal.
- `logs/results_eval.csv`: hasil evaluasi dalam format bersama.
- `splits/train.txt`, `splits/val.txt`, dan `splits/test.txt`: daftar file masing-masing split.
- `splits/split_all.csv`: metadata split dan citra.
- `gt/gt_train.json`, `gt/gt_val.json`, dan `gt/gt_test.json`: ground truth yang disimpan dalam format JSON.
- File checkpoint model di `weights/`.
- File prediksi dan visualisasi yang dihasilkan selama proses eksperimen.

Nama file tambahan dapat mengikuti nama run atau tahapan evaluasi yang dijalankan.

## 12. Reproducibility dan Pengelolaan Run

- Seed awal ditetapkan pada notebook dan fungsi seed digunakan untuk proses random Python, NumPy, dan PyTorch.
- Split dataset resmi tidak diubah.
- Run eksperimen dicatat pada log.
- Notebook memiliki mekanisme untuk melewati run yang bobotnya sudah tersimpan, kecuali run dipaksa ulang melalui pengaturan terkait.
- Konfigurasi eksperimen dan override sebaiknya dicatat serta tidak diubah secara sembarangan agar perbandingan tetap dapat ditelusuri.
- Evaluasi test mengikuti pembatasan penggunaan test yang ditetapkan notebook.

## 13. Teknologi dan Library

- Python
- Google Colab
- Google Drive
- PyTorch
- Torchvision
- Ultralytics YOLO
- TorchMetrics / evaluator metrik terkait
- pycocotools
- NumPy
- Pandas
- Matplotlib
- Pillow
- ImageHash (pHash)

Library dan versi yang digunakan mengikuti lingkungan runtime serta instalasi yang dilakukan notebook.

## 14. Batasan

- Dataset yang digunakan memiliki karakteristik dan distribusi tertentu; performa pada dataset atau kondisi pencitraan lain belum tentu sama.
- Pemeriksaan pHash tidak dapat memastikan bahwa citra berasal dari apusan yang berbeda jika metadata asal apusan tidak tersedia.
- Evaluasi penghitungan bergantung pada anotasi ground truth, konfigurasi confidence threshold, dan protokol evaluasi yang digunakan.
- Eksperimen komputasional ini tidak dimaksudkan untuk penggunaan diagnosis klinis.

## 15. Penulis / Tim

Notebook mencantumkan pemilik proyek:

- Khofifah
- Hilyatul Abidah

Silakan menyesuaikan bagian ini apabila README digunakan pada repository publik atau repository tim.

---

**Catatan:** README ini menjelaskan alur dan konfigurasi proyek. Untuk hasil numerik final, gunakan hasil evaluasi aktual yang dihasilkan dari notebook.
