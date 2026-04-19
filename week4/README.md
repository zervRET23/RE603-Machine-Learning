# RE603-Machine-Learning
 Week 4 membahas **Linear Regression** mulai dari eksplorasi data (EDA), preprocessing, training model, hingga evaluasi performa model.
Dataset yang digunakan adalah USA Housing Dataset.

---

## Dataset
File: `USA_Housing.csv`

Fitur dalam dataset:
- Avg. Area Income
- Avg. Area House Age
- Avg. Area Number of Rooms
- Avg. Area Number of Bedrooms
- Area Population
- Price (target)

---

## Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro

import statsmodels.api as sm

%matplotlib inline
```

## Load Dataset
```python
df = pd.read_csv('DATASETS/USA_Housing.csv')
print(df.head())
```

## Exploratory Data Analysis 
```python
print(df.info())
print(df.describe())

sns.pairplot(df)

df['Price'].plot.hist(bins=25, figsize=(8,4))
```

## Data Exploration
```python
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

## Model Training
```python
model = LinearRegression()
model.fit(X_train, y_train)
```

## Prediction
```python
predictions = model.predict(X_test)
```

## Evalution
```python
print("MAE:", metrics.mean_absolute_error(y_test, predictions))
print("MSE:", metrics.mean_squared_error(y_test, predictions))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, predictions)))
```

## Residual Analysis
```python
residuals = y_test - predictions
sns.histplot(residuals, bins=50)
```

## Assumption Testing
Normality test
```python
shapiro_test = shapiro(residuals)
print("Shapiro-Wilk Test:", shapiro_test)
```

Heteroskedasticity test
```python
X_test_const = sm.add_constant(X_test)
bp_test = het_breuschpagan(residuals, X_test_const)

print("Breusch-Pagan Test:")
print("Lagrange Multiplier:", bp_test[0])
print("p-value:", bp_test[1])
print("f-value:", bp_test[2])
print("f p-value:", bp_test[3])
```

Autocorrelation
```python
dw = durbin_watson(residuals)
print("Durbin-Watson:", dw)
```

## Conclusion
- Linear Regression dapat digunakan untuk memprediksi harga rumah
- Evaluasi model dilakukan menggunakan MAE, MSE, dan RMSE
- Asumsi klasik diuji menggunakan:

    - Shapiro-Wilk (normalitas)
    - Breusch-Pagan (heteroskedastisitas)
    - Durbin-Watson (autokorelasi)


