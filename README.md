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

- Alcohol                         ( Menentukan jumlah kandungan alkohol) (%)                  
- Malic acid                      ( Menetukan jumlah kandungan asam malat ) (g/L)
- Ash                             ( Menetukan jumlah kangdungan abu ) (g/L)
- Alcalinity of ash               ( Menetukan jumlah kandungan alkalinitas abu ) (meq/L) 
- Magnesium                       ( Menentukan jumlah kandungan magnesium ) (g/L)
- Total phenols                   ( Menentukan jumlah kandungan senyawa fenolik ) (g/L)
- Flavanoids                      ( Menentukan jumlah kandungan flavanoids ) (g/L)
- Nonflavanoid phenols            ( Menentukan Jumlah kandungan Nonflavanoid phenols ) (g/L)
- Proanthocyanins                 ( Menentukan jumlah kandungan Proanthocyanins ) (g/L)
- Color intensity                 ( Menentukan jumlah kandungan Intensitas Warna ) 
- Hue                             ( Menentukan jumlah kandungan jenis warna ) 
- OD280/OD315 of diluted wines    (Menentukan jumlah kandungan anggur yang diencerkan )
- Proline                         ( Menentukan jumlah kandungan proline ) (g/L)

## Data Preparation
**Data Collection**<br>
Untuk data collection ini, saya mendapatkan dataset yang nantinya digunakan dari website kaggle dengan nama dataset wine clustering.

## Library yang akan digunakan
**Data Discovery And Profiling**<br>
Teknik EDA.<br>

Import semua library yang dibutuhkan

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sklearn.cluster as cluster
    from sklearn.model_selection import train_test_split
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from yellowbrick.cluster import KElbowVisualizer
    from sklearn.decomposition import PCA

 ## Deskripsi Dataset
Karena kita memakai googgle collab bukan csv maka kita Import file 

    from google.colab import files

Upload token kaggle agar nanti bisa mendownload sebuah dataset dari kaggle melalui google colab

    file.upload()

Setelah mengupload filenya, selanjutnya membuat folder untuk menyimpan file kaggle.json yang sudah diupload tadi

    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
    !ls ~/.kaggle

Lalu download datasetsnya
     !kaggle datasets download -d harrywang/wine-dataset-for-clustering

Extract file yang tadi telah didownload

    !mkdir wine-dataset-for-clustering
    !unzip wine-dataset-for-clustering.zip -d wine-dataset-for-clustering
    !ls wine-dataset-for-clustering

## Data Descovery
Membaca sebuah file CSV yang berisi data anggur

    df = pd.read_csv("/content/wine-dataset-for-clustering/wine-clustering.csv")
    df.head()
    
Menampilkan beberapa baris pertama dari suatu dataset yang disimpan dalam bentuk dataframe.

    df.head()

Menampilkan informasi ringkas tentang tabel data dalam pandas.

    df.info()
    
Metode ini memberikan gambaran cepat tentang statistik dasar dari setiap kolom numerik dalam DataFrame.

    df.describe()

## EDA

1. Mengurutkan kandungan nilai alkohol terendah sampapi tertinggi

    df.sort_values("Alcohol").head(10)

   ![Alt text](image-1.png)

3. Mengurutkan kandungan nilai proline terendah sampai tertinggi

    df.sort_values("Proline").head()

    ![Alt text](image-1.png)

4. Menampilkan berapa banyak tipe kandungan Alcohol

    df["Alcohol"].value_counts()

    ![Alt text](image-1.png)

5. Mengurutkan banyak kandungan nilai alkohol berdasarkan kandungan nilai magnesium

    df_sorted = df.sort_values(by=['Alcohol', 'Magnesium'], ascending=[True,False])
    df_sorted.head()

    ![Alt text](image-1.png)

6. Menampilkan jumlah kandungan nilai alkohol pada estimasi jumlah kandungan nilai magnesium

    pd.crosstab(df["Alcohol"],df["Magnesium"], margins=True)

    ![Alt text](image-1.png)


## Preprocessing

Lakukan seleksi fitur dengan memisahkan mana saja atribut yang akan dijadikan sebagai fitur dan atribut mana yang dijadikan label

    features = ['Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium', 'Total_Phenols', 'Flavanoids', 'Nonflavanoid_Phenols', 'Proanthocyanins', 'Color_Intensity']
    x = df[features]
    y = df['Alcohol']
    x.shape, y.shape
    
Hasilnya ((178, 9), (178,))

Untuk membagi dataset menjadi data pelatihan (train) dan data pengujian (test).

    x_train, X_test, y_train, y_test = train_test_split(x, y, random_state = 70)
    y_test.shape

Hasilnya (45,)

Untuk mencari nilai yang kosong atau null

    df.isnull().sum()

Membuat model Linear Regressionnya

    model = LinearRegression()

Import modul LinearRegression dari scikit-learn

    from sklearn.linear_model import LinearRegression

Inisialisasi model regresi linear

    lr = LinearRegression()

Melatih model dengan data pelatihan

    lr.fit(x_train, y_train)

Melakukan prediksi

    predik = lr.predict(X_test)

Mengukur akurasi model regresi linear dengan membandingkan hasil prediksi model pada data pengujian 

    y = lr.score(X_test, y_test)
    print('Akurasi Model Regresi Linier : ', y)

Hasilnya 0.9991840271557658. Untuk memprediksi jumlah kalori berdasarkan nilai-nilai yang diberikan. Hasil prediksi akan disimpan dalam variabel prediksi 

