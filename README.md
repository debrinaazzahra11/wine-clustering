# Laporan Proyek Machine Learning
### Nama       : Debrina Azzahra
### Nim        : 211351155
### Kelas      : TIF Malam B
### Algorima   : K-Means

## Domain Proyek
  Anallisis jenis anggur yang ditanam di wilayah yang sama di Italia, namun berasal dari tiga jenis anggur  berbeda. Jenis anggur memiliki peran penting dalam menentukan karakteristik kimia dari anggur yang dihasilkan. Analisis kimiawi melibatkan identifikasi dan pengukuran 13 data utama yang dianggap signifikan dalam menentukan kualitas dan karakteristik anggur. data-data ini mencakup berbagai elemen seperti alkohol, asam malat, abu, alkalinitas abu, magnesium, jumlah fenol, flavanoid, fenol nonflavanoid, proanthocyanin, intensitas warna, warna, OD280/OD315 anggur encer, dan prolin.

  Format Referensi: [Wine Cluster](https://www.kaggle.com/datasets/harrywang/wine-dataset-for-clustering/data) 

  # BUSINESS UNDERSTANDING
  Dilakukan proses klasterisasi untuk mengidentifikasi pola-pola kemiripan atau perbedaan antara sampel-sampel anggur. Klasterisasi adalah teknik pengelompokan data yang sering digunakan dalam analisis data untuk mengelompokkan objek-objek yang serupa berdasarkan atribut-atribut tertentu.
  Penerapan klasterisasi pada data ini dapat membantu mengidentifikasi kelompok-kelompok anggur yang memiliki profil kimiawi serupa atau berbeda. Dengan mengelompokkan anggur berdasarkan kemiripan karakteristik kimianya, kita dapat memahami lebih baik bagaimana tiga kultivar tersebut berbeda satu sama lain dalam hal komposisi kimia anggurnya.

### Problem Statements
 Menganalisis cluster anggur berdasarkan kandungan kimianya. Mencakup berbagai elemen seperti alkohol, asam malat, abu, alkalinitas abu, magnesium, jumlah fenol, flavanoid, fenol nonflavanoid, proanthocyanin, intensitas warna, warna, OD280/OD315 anggur encer, dan prolin yang kemudian di bedakan berdasarkan kluster kluster yang ada.

### Goals
  Dataset yang di ambil dari kaggle bertujuan untuk menngelompokkan data anggur, berdasarkan kluster yang ada.
  
### Rubrik/Kriteria Tambahan (Opsional) :
**Solution statements**<br>
- Platform berbasis web maupun aplikasi yang memberikan informasi mengenai kluster-kluster dalam data anggur.
- Model yang dihasilkan dari datasets menggunakan algoritma k-Means.

## Data Understanding

Tahap ini, membuat ringkasaan (summary) dan mengidentifikasi potensi masalah m data. Tahap ini juga harus dilakukan secara cermat dan tidak terburu-buru, seperti pada visualisasi data, yang terkadang insight-nya sulit didapat dika dihubungkan dengan summary data nya. Jika ada masalah pada tahap ini yang belum terjawab, maka akan menggangu pada p modeling. Dataset yang saya gunakan berasal dari Kaggle yang didapat dari perkebunan anggun yang berada di Italia.

Dataset: [Wine Cluster](https://www.kaggle.com/datasets/harrywang/wine-dataset-for-clustering/data)

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:

- Menu          (Menentukan ukuran menu)            = objek (Ukuran : Besar, sedang, kecil)
- Calories      (Menentukan jumlah kalori)          = int (mg)
- Fat_Calories  (Menentukan jumlah lemak kalori )   = int (kcal)
- Total_fat     (Menentukan jumlah lemak total)     = float (mg)
- Cholesterol   (Menentukan jumalah kolestrol)      = int (mg)
- Sodium        (Menentukan jumlah sodium)          = int (9)
- Carbohydrate  (Menentukan jumlah karbohidrat)     = int (g)
- Sugars        (Menentukan jumlah gula)            = int (g)
- Protein       (Menentukan jumlah protein)         = float (g)
