# RE603-Machine-Learning
Week 5 membahas **K-Nearest Neighbors (KNN)** mulai dari eksplorasi data (EDA), preprocessing, training model, hingga evaluasi performa model.
Dataset yang digunakan adalah Iris Dataset.

---

## Dataset
File: `iris.csv`

Fitur dalam dataset:
- sepal_length
- sepal_width
- petal_length
- petal_width
- species (target)

---

## Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

%matplotlib inline
```

## Load Dataset
```python
df = pd.read_csv('DATASETS/iris.csv')
print(df.head())
```

## Exploratory Data Analysis 
```python
print(df.info())
print(df.describe())

sns.pairplot(df, hue='species')

df['species'].value_counts().plot(kind='bar')
```

## Data Preprocessing
```python
# Encoding target
df['species'] = df['species'].astype('category').cat.codes
```

## Data Exploration
```python
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

## Model Training
```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```

## Prediction
```python
predictions = knn.predict(X_test)
```

## Evaluation
```python
print("KNN Accuracy:", metrics.accuracy_score(y_test, predictions))
print(metrics.confusion_matrix(y_test, predictions))
print(metrics.classification_report(y_test, predictions))
```

## Visualization
```python
plt.figure(figsize=(6,4))
sns.heatmap(metrics.confusion_matrix(y_test, predictions), annot=True, fmt='d')
plt.title("KNN Confusion Matrix")
```

## Conclusion
- KNN digunakan untuk klasifikasi berdasarkan kedekatan jarak antar data
- Pemilihan nilai K mempengaruhi performa model
- Dataset Iris memiliki pola yang jelas sehingga KNN dapat menghasilkan akurasi tinggi
- Evaluasi model dilakukan menggunakan:

    - Accuracy
    - Confusion Matrix
    - Precision, Recall, F1-score

- Model KNN cocok untuk dataset dengan ukuran kecil hingga menengah dan pola yang tidak terlalu kompleks
