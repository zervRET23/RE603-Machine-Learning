# RE603-Machine-Learning
Week 6 membahas **Naive Bayes** dan **Decision Tree** mulai dari eksplorasi data (EDA), preprocessing, training model, hingga evaluasi performa model.
Dataset yang digunakan adalah Titanic Dataset.

---

## Dataset
File: `titanic.csv`

Fitur dalam dataset:
- Pclass
- Sex
- Age
- SibSp
- Parch
- Fare
- Embarked
- Survived (target)

---

## Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

%matplotlib inline
```

## Load Dataset
```python
df = pd.read_csv('DATASETS/titanic.csv')
print(df.head())
```

## Exploratory Data Analysis 
```python
print(df.info())
print(df.describe())

sns.countplot(x='Survived', data=df)
sns.countplot(x='Sex', hue='Survived', data=df)

df.hist(figsize=(10,8))
```

## Data Preprocessing
```python
# Handle missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encoding categorical data
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
```

## Data Exploration
```python
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

## Model Training

Naive Bayes
```python
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
```

Decision Tree
```python
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
```

## Prediction

Naive Bayes
```python
nb_predictions = nb_model.predict(X_test)
```

Decision Tree
```python
dt_predictions = dt_model.predict(X_test)
```

## Evaluation

Naive Bayes
```python
print("Naive Bayes Accuracy:", metrics.accuracy_score(y_test, nb_predictions))
print(metrics.confusion_matrix(y_test, nb_predictions))
print(metrics.classification_report(y_test, nb_predictions))
```

Decision Tree
```python
print("Decision Tree Accuracy:", metrics.accuracy_score(y_test, dt_predictions))
print(metrics.confusion_matrix(y_test, dt_predictions))
print(metrics.classification_report(y_test, dt_predictions))
```

## Visualization
```python
plt.figure(figsize=(6,4))
sns.heatmap(metrics.confusion_matrix(y_test, nb_predictions), annot=True, fmt='d')
plt.title("Naive Bayes Confusion Matrix")

plt.figure(figsize=(6,4))
sns.heatmap(metrics.confusion_matrix(y_test, dt_predictions), annot=True, fmt='d')
plt.title("Decision Tree Confusion Matrix")
```

## Conclusion
- Naive Bayes dapat digunakan untuk klasifikasi sederhana dengan performa yang cepat
- Decision Tree mampu menangkap pola yang lebih kompleks pada data
- Pada dataset Titanic, fitur seperti Sex, Pclass, dan Fare memiliki pengaruh besar terhadap keselamatan penumpang
- Evaluasi model dilakukan menggunakan:

    - Accuracy
    - Confusion Matrix
    - Precision, Recall, F1-score

- Pemilihan model tergantung pada kebutuhan interpretasi dan kompleksitas data

