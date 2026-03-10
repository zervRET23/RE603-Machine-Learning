# RE603-Machine-Learning
 Week 3 membahas **Exploratory Data Analysis (EDA)** dan **Feature Engineering** sebagai tahap awal dalam proses Machine Learning.

---

## Dataset

Dataset yang digunakan dalam tugas ini berasal dari **Telco Customer Churn Dataset**, yang berisi informasi pelanggan perusahaan telekomunikasi dan apakah pelanggan tersebut berhenti menggunakan layanan (churn) atau tidak.

Dataset yang digunakan:

* `Telco-Customer-Churn.csv`
* `company.csv`

### Deskripsi Dataset

Dataset ini berisi beberapa informasi pelanggan seperti:

* **CustomerID** – ID unik pelanggan
* **Gender** – Jenis kelamin pelanggan
* **SeniorCitizen** – Status pelanggan lansia
* **Partner** – Apakah pelanggan memiliki pasangan
* **Dependents** – Apakah pelanggan memiliki tanggungan
* **Tenure** – Lama berlangganan (bulan)
* **PhoneService** – Apakah menggunakan layanan telepon
* **InternetService** – Jenis layanan internet
* **OnlineSecurity, OnlineBackup, DeviceProtection** – Layanan tambahan
* **Contract** – Jenis kontrak pelanggan
* **PaymentMethod** – Metode pembayaran
* **MonthlyCharges** – Biaya bulanan
* **TotalCharges** – Total biaya pelanggan
* **Churn** – Target variable (Yes / No)

Target dari analisis ini adalah memahami faktor-faktor yang mempengaruhi **customer churn**.

---

## Materi

### 1. Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) merupakan proses eksplorasi data untuk memahami karakteristik dataset sebelum dilakukan proses modeling.

Proses yang dilakukan pada tahap ini meliputi:

* Melihat struktur dataset
* Mengetahui tipe data
* Mengecek missing values
* Melihat statistik deskriptif
* Visualisasi distribusi data
* Analisis korelasi antar fitur

Tujuan dari EDA adalah untuk menemukan **pola, hubungan, dan insight penting dari data**.

---

#### Implementasi Exploratory Data Analysis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Melihat 5 data pertama
df.head()

# Informasi dataset
df.info()

# Statistik deskriptif
df.describe()

# Mengecek missing values
df.isnull().sum()

# Visualisasi distribusi target
sns.countplot(x='Churn', data=df)
plt.title("Customer Churn Distribution")
plt.show()

# Korelasi antar fitur
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
```

---

### EDA Insights

Berdasarkan hasil eksplorasi data, beberapa insight yang diperoleh antara lain:

* Pelanggan dengan **kontrak bulanan (month-to-month)** memiliki kemungkinan churn yang lebih tinggi dibandingkan kontrak tahunan.
* Pelanggan dengan **biaya bulanan tinggi (MonthlyCharges)** cenderung memiliki tingkat churn yang lebih tinggi.
* Pelanggan dengan **tenure yang rendah** lebih sering melakukan churn dibanding pelanggan yang sudah lama berlangganan.
* Beberapa layanan tambahan seperti **OnlineSecurity dan TechSupport** berkorelasi dengan tingkat churn yang lebih rendah.
* Distribusi churn pada dataset menunjukkan bahwa sebagian besar pelanggan **tidak churn**, sehingga dataset memiliki sedikit **class imbalance**.

Insight ini dapat membantu dalam proses **feature selection dan model building** pada tahap machine learning berikutnya.

---

### 2. Feature Engineering

Feature Engineering merupakan proses mengubah atau mempersiapkan fitur agar dapat digunakan oleh model Machine Learning secara optimal.

Proses yang dilakukan meliputi:

* Encoding data kategorikal
* Transformasi data
* Normalisasi / scaling
* Persiapan dataset sebelum training model

Tujuan dari feature engineering adalah **membuat representasi data lebih mudah dipahami oleh algoritma machine learning**.

---

#### Implementasi Feature Engineering

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Kolom yang perlu encoding
columns_to_encode = [
    'StreamingMovies',
    'StreamingTV',
    'TechSupport',
    'DeviceProtection',
    'OnlineBackup',
    'OnlineSecurity',
    'MultipleLines'
]

# Label Encoding
l_enc = LabelEncoder()

for col in columns_to_encode:
    df[col] = l_enc.fit_transform(df[col])

# Encoding target variable
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Feature Scaling
scaler = StandardScaler()

numerical_cols = df.select_dtypes(include=['int64','float64']).columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Dataset setelah preprocessing
df.head()
```

---

## Tools

* Python 3
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook / VSCode

---

**Machine Learning RE603 — Week 3**
