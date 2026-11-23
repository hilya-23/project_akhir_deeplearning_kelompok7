# Project Akhir Deep Learning: Deteksi dan Penghitungan Sel Darah

> **Eksperimen komputasional untuk membandingkan YOLOv8s dan Faster R-CNN ResNet-50 FPN pada dataset TXL-PBC.**

Proyek ini merupakan evaluasi eksperimen *deep learning* untuk mendeteksi dan menghitung tiga kelas sel darah pada citra mikroskopis, yaitu White Blood Cell (WBC), Red Blood Cell (RBC), dan Platelet. Perbandingan dilakukan menggunakan YOLOv8s (*one-stage detector*) dan Faster R-CNN ResNet-50 FPN (*two-stage detector*). Evaluasi mencakup performa deteksi, galat penghitungan, analisis kesalahan, serta efisiensi komputasi.

> Penelitian ini merupakan eksperimen komputasional, **bukan sistem diagnosis atau pengganti pemeriksaan tenaga medis**.

## Daftar Isi
- [Tujuan](#1-tujuan)
- [Dataset](#2-dataset)
- [Pipeline Penelitian](#3-pipeline-penelitian)
- [Pemeriksaan Dataset dan Anotasi](#4-pemeriksaan-dataset-dan-anotasi)
- [Model dan Konfigurasi](#5-model-dan-konfigurasi)
- [Augmentasi Data](#6-augmentasi-data)
- [Evaluator dan Metrik](#7-evaluator-dan-metrik)
- [Eksperimen](#8-rangkaian-eksperimen)
- [Penentuan Confidence Threshold](#9-penentuan-confidence-threshold)
- [Analisis](#10-analisis)
- [Efisiensi Komputasi](#11-efisiensi-komputasi)
- [Cara Menjalankan](#12-cara-menjalankan-notebook)
- [Struktur Output](#13-struktur-output)
- [Reproducibility](#14-reproducibility-dan-pengelolaan-run)
- [Keterbatasan](#15-keterbatasan)
- [Anggota Kelompok](#16-anggota-kelompok)

## 1. Tujuan
- Membandingkan performa deteksi YOLOv8s dan Faster R-CNN ResNet-50 FPN.
- Mengevaluasi ketepatan penghitungan WBC, RBC, dan Platelet berdasarkan anotasi *ground truth*.
- Mengamati pengaruh konfigurasi, termasuk resolusi input dan *confidence threshold*.
- Mengukur efisiensi model, antara lain latensi dan jumlah parameter.
- Menganalisis kesalahan deteksi dan penghitungan, termasuk *over-counting* dan *under-counting*.

## 2. Dataset
Dataset yang digunakan adalah **TXL-PBC**, dengan anotasi bounding box format YOLO dan split train, validation, serta test yang telah disediakan. Split resmi dipertahankan dan tidak diacak ulang.

| ID | Kelas | Keterangan |
|---:|---|---|
| 0 | WBC | White Blood Cell / sel darah putih |
| 1 | RBC | Red Blood Cell / sel darah merah |
| 2 | Platelet | Trombosit |

### Struktur dataset yang diharapkan
Notebook mencari `data.yaml` di dalam folder dataset TXL-PBC.

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

Setiap citra memiliki file label `.txt` dengan nama yang sesuai. Format label YOLO:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Koordinat bounding box ternormalisasi terhadap ukuran citra. Untuk evaluator bersama dan Faster R-CNN, anotasi dikonversi menjadi `xyxy` dalam koordinat piksel.

- `train`: pelatihan model.
- `val`: validasi, pemilihan konfigurasi dan threshold.
- `test`: evaluasi akhir setelah konfigurasi dikunci.

Data test tidak digunakan untuk memilih konfigurasi model.

## 3. Pipeline Penelitian
1. **Konfigurasi lingkungan dan direktori:** menghubungkan Google Drive, menetapkan lokasi dataset/output, seed, dan direktori kerja.
2. **Ekstraksi dan audit dataset:** mengekstrak dataset serta memeriksa inventaris citra dan label.
3. **EDA:** menganalisis distribusi kelas, jumlah objek, ukuran objek/citra dan karakteristik dataset.
4. **Konversi anotasi:** mengubah label YOLO ternormalisasi menjadi bounding box `xyxy` piksel.
5. **Verifikasi anotasi:** memeriksa jumlah box, batas koordinat dan visualisasi sampel.
6. **Pemeriksaan kemiripan citra:** pHash dan jarak Hamming untuk memeriksa citra identik atau sangat mirip.
7. **Persiapan dan augmentasi data:** augmentasi dipilih berdasarkan karakteristik citra sel darah.
8. **Pelatihan YOLOv8s:** run, log, checkpoint, dan konfigurasi eksperimen.
9. **Evaluasi bersama:** hasil kedua model diproses oleh evaluator yang sama.
10. **Ablasi resolusi:** pengujian input 640 dan 832.
11. **Multi-seed:** evaluasi konfigurasi terpilih YOLOv8s menggunakan seed 42, 43, dan 44.
12. **Pelatihan dan evaluasi Faster R-CNN:** menggunakan split dan evaluator yang sesuai.
13. **Analisis kesalahan:** over-counting, under-counting, missed detection, false positive, serta kasus deviasi besar.
14. **Analisis performa dan efisiensi:** merangkum deteksi, penghitungan, latensi, jumlah parameter, dan hasil analisis.

## 4. Pemeriksaan Dataset dan Anotasi
Sebelum pelatihan, notebook melakukan:
- Inventaris jumlah citra dan label pada setiap split.
- Pemeriksaan file label hilang atau kosong.
- Pemeriksaan nama file yang sama antar-split.
- Parsing label YOLO dan konversi ke koordinat piksel.
- Pemeriksaan jumlah objek sebelum dan setelah konversi.
- Pemeriksaan bounding box terhadap batas citra.
- Verifikasi visual pada 20 citra acak.
- Penyimpanan *ground truth* dalam format JSON.
- Pemeriksaan kemiripan citra menggunakan pHash.

### Pemeriksaan kemiripan (pHash)
Pemeriksaan menggunakan pHash 64-bit dan jarak Hamming:
- Jarak 0: identik berdasarkan hash.
- Jarak ≤ 5: indikator citra yang sangat mirip.

pHash hanya mendeteksi kemiripan visual. Karena informasi asal apusan/slide tidak tersedia, pemeriksaan ini tidak dapat memastikan bahwa citra dari lapang pandang berbeda berasal dari apusan yang berbeda. Pemeriksaan pHash bukan pengganti metadata asal slide.

## 5. Model dan Konfigurasi
### 5.1 YOLOv8s
YOLOv8s merupakan *one-stage detector*. Notebook menggunakan bobot awal COCO `yolov8s.pt` dan checkpoint terbaik dipilih berdasarkan validation.

| Parameter | Konfigurasi dasar |
|---|---:|
| Model awal | `yolov8s.pt` |
| Input size | 640 |
| Epoch maksimum | 100 |
| Patience | 20 |
| Batch size | 16 |
| Optimizer | AdamW |
| Initial LR (`lr0`) | 0,001 |
| Final LR factor (`lrf`) | 0,01 |
| Weight decay | 0,0005 |
| Workers | 2 |
| AMP | Aktif |
| Deterministic | Aktif |
| Seed | 42 (run multi-seed: 42, 43, 44) |

Konfigurasi run tertentu dapat mengikuti override eksperimen yang dicatat di notebook/log.

### 5.2 Faster R-CNN ResNet-50 FPN
Faster R-CNN merupakan *two-stage detector* menggunakan backbone ResNet-50 FPN dengan pretrained weights COCO. Head disesuaikan untuk tiga kelas objek dan background.

| Parameter | Konfigurasi |
|---|---:|
| Arsitektur | Faster R-CNN ResNet-50 FPN |
| Optimizer | AdamW |
| Learning rate | 0,0001 |
| Weight decay | 0,0005 |
| Batch size | 4 |
| Scheduler | StepLR |
| Step size | 3 |
| Gamma | 0,1 |
| Output | 4 (3 kelas + background) |
| Seed eksperimen | 42, 43, 44 |
| NMS IoU inferensi | 0,6 |

Label torchvision disesuaikan kembali menjadi label 0-indexed untuk evaluator bersama.

## 6. Augmentasi Data
Augmentasi YOLOv8s dipilih dengan mempertimbangkan karakteristik visual sel darah, terutama warna pewarnaan dan ukuran objek.

| Augmentasi | Nilai | Pertimbangan |
|---|---:|---|
| Horizontal flip | 0,5 | Orientasi horizontal bukan ciri pembeda utama |
| Vertical flip | 0,5 | Orientasi vertikal bukan ciri pembeda utama |
| Rotasi bebas | 0° | Rotasi non-90° dapat mengubah box dan menambah label noise |
| Hue | 0,01 | Warna penting; hue dibatasi |
| Saturation | 0,3 | Variasi ringan |
| Value/brightness | 0,3 | Variasi intensitas ringan |
| Translate | 0,1 | Variasi posisi |
| Scale | 0,3 | Variasi ukuran dengan perhatian pada objek kecil |
| Mosaic | 1,0 | Dimatikan pada 10 epoch terakhir |
| MixUp | 0 | Tidak digunakan |

Faster R-CNN dilatih tanpa augmentasi pada konfigurasi eksperimen notebook.

## 7. Evaluator dan Metrik
### Evaluator bersama
Semua model dievaluasi menggunakan `torchmetrics.MeanAveragePrecision` dengan dukungan `pycocotools`, menggunakan koordinat piksel asli.

Format prediksi bersama:

```text
{nama_citra: {boxes: Nx4 xyxy piksel, scores: N, labels: N}}
```

Ketentuan:
- Bounding box: `xyxy`, koordinat piksel asli.
- Label: 0-indexed (0=WBC, 1=RBC, 2=Platelet).
- Prediksi mentah disimpan dengan confidence ≥ 0,001 untuk evaluasi mAP.
- Threshold penghitungan diterapkan setelah prediksi mentah tersimpan, sehingga threshold dapat diuji tanpa inferensi ulang.
- Evaluasi menggunakan fungsi yang sama dan hasil ditulis ke `results_eval.csv`.

### Metrik deteksi
- **mAP@0.5:** AP pada IoU 0,5.
- **mAP@[0.5:0.95]:** AP pada rentang IoU 0,5 sampai 0,95 dengan interval 0,05.
- **AP per kelas:** performa deteksi WBC, RBC, dan Platelet.

### Metrik penghitungan
- **MAE:** rata-rata selisih absolut antara jumlah prediksi dan ground truth; dihitung pada semua citra.
- **MAPE:** persentase kesalahan relatif; dalam evaluator dihitung pada citra dengan ground truth > 0 untuk kelas tersebut.
- **Bias:** kecenderungan prediksi jumlah lebih tinggi atau lebih rendah dari ground truth.
- **Korelasi (`r`):** hubungan antara jumlah prediksi dan aktual sesuai analisis notebook.
- **Macro MAE:** ringkasan MAE antar kelas sesuai evaluator.

Metrik deteksi dan penghitungan dianalisis secara terpisah. mAP tinggi tidak otomatis berarti galat penghitungan rendah.

## 8. Rangkaian Eksperimen
### 8.1 Pelatihan YOLOv8s
Run pelatihan dicatat pada `experiment_log.csv`, termasuk run gagal. Checkpoint terbaik dipilih menggunakan validation. Run yang checkpoint-nya telah tersimpan dapat dilewati ketika notebook dijalankan kembali.

### 8.2 Ablasi resolusi 640 vs 832
Ablasi mengubah satu faktor, yaitu resolusi input, dari 640 ke 832, dengan faktor lainnya dipertahankan (seed 42, batch, augmentasi, dan konfigurasi pelatihan sesuai notebook). Sebagian besar citra berukuran ≤ 640 piksel, sehingga resolusi 832 merupakan upsampling.

Notebook mempertahankan resolusi 640 sebagai konfigurasi final: hasil 832 pada seed 42 sedikit lebih tinggi, tetapi berada dalam rentang variasi antar-seed 640. Hasil 832 hanya satu seed, sehingga hipotesis manfaat khusus pada Platelet masih inkonklusif. Keputusan juga mempertimbangkan efisiensi dan tidak melakukan upsampling sebagian besar citra.

### 8.3 Multi-seed YOLOv8s
Konfigurasi final dijalankan menggunakan tiga seed (42, 43, 44). Evaluasi validation digunakan untuk memilih konfigurasi; test dilakukan setelah konfigurasi dikunci.

### 8.4 Faster R-CNN
Faster R-CNN ResNet-50 FPN dilatih dan dievaluasi menggunakan tiga seed (42, 43, 44), NMS IoU inferensi 0,6, dan evaluator yang sama.

### 8.5 Kesetaraan dan perbedaan setting
| Aspek | YOLOv8s | Faster R-CNN | Catatan |
|---|---|---|---|
| Seed | 42, 43, 44 | 42, 43, 44 | Disamakan |
| NMS IoU inferensi | 0,6 | 0,6 | Disamakan |
| Threshold penghitungan | Dipilih per kelas di validation | Dipilih per kelas di validation | Prosedur sama, nilai model/kelas dapat berbeda |
| Bobot awal | COCO | COCO | Sama sumber pretrained |
| Optimizer | AdamW | AdamW | Sama |
| Augmentasi | Flip, HSV, scale, mosaic | Tidak ada | Berbeda |
| Resolusi | 640 | Min size bawaan torchvision 800 | Berbeda |
| Epoch | Hingga 100, patience 20 | 13, StepLR | Berbeda |
| Seleksi checkpoint | Best berdasarkan validation | Epoch terbaik berdasarkan validation | Validation |

Perbedaan konfigurasi tersebut perlu dipertimbangkan ketika menafsirkan hasil, karena perbedaan performa tidak dapat sepenuhnya dikaitkan dengan arsitektur model saja.

## 9. Penentuan Confidence Threshold
Confidence threshold untuk penghitungan sel dipilih **per model dan per kelas** menggunakan validation.

Prosedur:
1. Prediksi mentah model digunakan kembali tanpa inferensi ulang.
2. Threshold diuji pada grid 0,05–0,95 dengan interval 0,05.
3. MAE penghitungan dirata-ratakan berdasarkan tiga seed pada validation.
4. Threshold yang meminimalkan MAE validation dipilih untuk setiap kelas.
5. Threshold tersebut diterapkan apa adanya pada test, tanpa tuning ulang di test.

Threshold mAP dan threshold penghitungan memiliki fungsi berbeda: prediksi mentah confidence ≥ 0,001 disimpan untuk evaluasi deteksi, sementara threshold terpilih dari validation digunakan untuk penghitungan.

## 10. Analisis
Notebook memuat analisis berikut:

### 10.1 mAP vs galat penghitungan
Perbandingan mAP dan MAE penghitungan dilakukan pada hasil test dengan tiga seed dan threshold yang dipilih dari validation. Korelasi mAP–MAE antar-run yang jumlahnya sedikit dan membentuk klaster model tidak digunakan sebagai bukti utama.

### 10.2 Uji berpasangan dan bootstrap
Galat penghitungan per citra dibandingkan menggunakan uji Wilcoxon berpasangan dan bootstrap 95% CI pada data test. Analisis menggunakan rata-rata galat absolut antar tiga seed.

### 10.3 Penguraian TP, FP, dan FN
Galat penghitungan diuraikan menggunakan IoU 0,5 menjadi TP, FP, dan FN. Jenis FP yang dianalisis mencakup deteksi ganda, sel terpotong di tepi citra, salah kelas, dan deteksi latar/objek lain.

### 10.4 MAE berdasarkan sumber dataset
MAE dianalisis per sumber dataset untuk melihat variasi performa pada kelompok citra dari sumber berbeda.

### 10.5 Kasus over-count RBC
Kasus over-count RBC dianalisis melalui visualisasi ground truth dan prediksi yang ditumpuk untuk meninjau penyebab prediksi berlebih.

### 10.6 Ukuran sampel
Ukuran sampel test diperiksa sebagai konteks interpretasi metrik, terutama untuk kelas dengan jumlah objek terbatas.

Hasil numerik final mengikuti output notebook yang benar-benar dijalankan; README ini tidak mengisi angka hasil yang belum tersedia.

## 11. Efisiensi Komputasi
Pengukuran dilakukan pada batch 1, setelah warm-up, hingga 100 citra validation. GPU yang digunakan harus dicantumkan saat melaporkan hasil.

Parameter yang dilaporkan:
- Jumlah parameter (juta).
- Waktu membaca berkas.
- Preprocessing.
- Inferensi.
- NMS/postprocessing jika dapat dipisahkan.
- Waktu model.
- Waktu end-to-end.

Pada YOLOv8s, preprocessing, inferensi, dan NMS diambil dari `r.speed` Ultralytics. Pada Faster R-CNN, NMS berada di dalam `roi_heads` sehingga tidak dapat dipisahkan langsung dari inferensi. `model_ms` digunakan sebagai basis perbandingan waktu model.

## 12. Cara Menjalankan Notebook
Notebook disiapkan untuk **Google Colab**.

### Persiapan
1. Siapkan akun Google dan Google Drive.
2. Buka atau unggah `Project_Akhir_DeepLearning_Revisi.ipynb` ke Colab.
3. Siapkan dataset TXL-PBC dalam bentuk ZIP di Google Drive.
4. Pastikan ZIP dataset memuat folder `TXL-PBC` dan `data.yaml` yang dicari notebook.
5. Disarankan menggunakan runtime GPU.

### Konfigurasi
Pada bagian **0.2 Konfigurasi — EDIT BAGIAN INI SAJA**, sesuaikan variabel:

```python
ZIP_PATH = '/content/drive/MyDrive/Project_Akhir_DeepLearning/TXL-PBC_Dataset-master.zip'
PROJECT_DIR = '/content/drive/MyDrive/Hasil_Akhir'
OWNER = 'Khofifah dan Hilyatul Abidah'
SEED = 42
```

- `ZIP_PATH`: lokasi ZIP dataset di Google Drive.
- `PROJECT_DIR`: direktori penyimpanan permanen output eksperimen.
- `OWNER`: identitas pemilik/anggota proyek.
- `SEED`: seed konfigurasi awal.

### Urutan menjalankan
1. Jalankan sel notebook secara berurutan.
2. Izinkan akses Google Drive.
3. Pastikan dataset berhasil diekstrak dan `data.yaml` ditemukan.
4. Jalankan audit, EDA, konversi/verifikasi anotasi, dan pHash.
5. Jalankan pelatihan dan eksperimen sesuai urutan notebook.
6. Pilih konfigurasi final berdasarkan validation.
7. Jalankan evaluasi test sesuai protokol setelah konfigurasi dikunci.
8. Jalankan analisis threshold, statistik, error, dan efisiensi.
9. Periksa log, tabel metrik, visualisasi, dan weights pada `PROJECT_DIR`.

## 13. Struktur Output
Notebook menyiapkan direktori output berikut:

```text
Hasil_Akhir/
├── splits/
├── gt/
├── predictions/
├── weights/
├── figures/
└── logs/
```

| Folder | Keterangan |
|---|---|
| `splits/` | Daftar file train, validation, test dan informasi split |
| `gt/` | Ground truth dalam format evaluator |
| `predictions/` | Prediksi model untuk evaluasi dan analisis |
| `weights/` | Checkpoint/bobot model |
| `figures/` | Grafik, EDA, verifikasi, dan analisis |
| `logs/` | Log run, tabel evaluasi, dan catatan eksperimen |

File output utama yang disiapkan notebook:
- `logs/experiment_log.csv`: log run, termasuk run yang gagal.
- `logs/results_eval.csv`: hasil evaluasi model.
- `splits/train.txt`, `splits/val.txt`, `splits/test.txt`: daftar file per split.
- `splits/split_all.csv`: informasi split dan citra.
- `gt/gt_train.json`, `gt/gt_val.json`, `gt/gt_test.json`: ground truth.
- `logs/ambang_hitung_val.json`: threshold penghitungan terpilih.
- `logs/sweep_ambang_val.csv`: hasil sweep threshold validation.
- File ringkasan multi-seed, prediksi, weights, dan visualisasi lainnya sesuai sel notebook yang dijalankan.

Nama file tambahan dapat mengikuti run atau tahap eksperimen. File yang tersedia bergantung pada sel notebook yang dijalankan.

## 14. Reproducibility dan Pengelolaan Run
- Seed dikonfigurasi untuk Python, NumPy, dan PyTorch.
- Split dataset resmi dipertahankan.
- Run dan statusnya dicatat dalam log.
- Run yang bobotnya telah tersimpan dapat dilewati ketika notebook dijalankan kembali.
- Override konfigurasi perlu dicatat agar eksperimen dapat ditelusuri.
- Evaluasi test mengikuti protokol notebook dan dicatat.

## 15. Keterbatasan
- Dataset memiliki karakteristik dan distribusi tertentu; performa pada dataset atau kondisi pencitraan lain belum tentu sama.
- pHash tidak dapat memastikan asal apusan/slide ketika metadata tidak tersedia.
- Faster R-CNN dilatih tanpa augmentasi, sedangkan YOLOv8s menggunakan augmentasi.
- Resolusi input dan konfigurasi epoch/scheduler kedua model berbeda.
- Jadwal LR Faster R-CNN meluruh cepat (StepLR, step 3, gamma 0,1); kurva yang mendatar setelah sekitar epoch 9 dapat dipengaruhi jadwal LR, bukan bukti konvergensi. Catatan ini berdasarkan konfigurasi lama yang direkonstruksi dan belum diverifikasi terhadap bobot.
- Evaluasi ulang test `frcnn_r50fpn_s42` dengan NMS IoU 0,6 merupakan pengecualian yang dicatat di `test_reeval_log.json`, karena run tersebut sebelumnya pernah dievaluasi menggunakan NMS IoU 0,5.
- Waktu GPU Faster R-CNN lama tidak tercatat.
- Test relatif kecil (126 citra, 49 Platelet); metrik per kelas/ukuran dapat bervariasi.
- Pemeriksaan pHash tidak dapat mendeteksi semua kemungkinan kebocoran, khususnya citra dari apusan yang sama dengan lapang pandang berbeda.
- Eksperimen ini tidak ditujukan untuk diagnosis klinis.

## 16. Anggota Kelompok
**Kelompok 7**
- Hilyatul Abidah (240441100132)
- Khofifah (240441100004)

---

README ini menjelaskan pipeline dan konfigurasi proyek. Untuk hasil numerik final, gunakan output eksperimen aktual dari notebook.
