# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1

2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.

3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.

4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler, RobustScaler
df = pd.read_csv("bmi.csv")
# Display first few rows
print(df.head())
```
![Screenshot 2025-04-23 154335](https://github.com/user-attachments/assets/7ac1574b-d25c-4e83-98e2-5a0bb02e5269)
```
# Drop missing values
df = df.dropna()
# Find maximum value from Height and Weight feature
print("Max Height:", df["Height"].max())
print("Max Weight:", df["Weight"].max())
```
![Screenshot 2025-04-23 154500](https://github.com/user-attachments/assets/9bc23557-0fe9-49c6-b03d-2c195fe53caa)
```
# Perform MinMax Scaler
minmax = MinMaxScaler()
df_minmax = minmax.fit_transform(df[["Height", "Weight"]])
print("\nMinMaxScaler Result:\n", df_minmax[:5])

```
![Screenshot 2025-04-23 154538](https://github.com/user-attachments/assets/643fe12e-8e06-48af-adb6-748e3539d028)
```
# Perform Standard Scaler
standard = StandardScaler()
df_standard = standard.fit_transform(df[["Height", "Weight"]])
print("\nStandardScaler Result:\n", df_standard[:5])
```
![Screenshot 2025-04-23 154625](https://github.com/user-attachments/assets/a8574b5d-e73c-41ee-8ea0-c6e06ff81b0d)
```
# Perform Normalizer
normalizer = Normalizer()
df_normalized = normalizer.fit_transform(df[["Height", "Weight"]])
print("\nNormalizer Result:\n", df_normalized[:5])

```
![Screenshot 2025-04-23 154718](https://github.com/user-attachments/assets/cf458da7-576f-425b-a04b-92f013f0a98f)
```
# Perform MaxAbsScaler
max_abs = MaxAbsScaler()
df_maxabs = max_abs.fit_transform(df[["Height", "Weight"]])
print("\nMaxAbsScaler Result:\n", df_maxabs[:5])
```
![Screenshot 2025-04-23 154800](https://github.com/user-attachments/assets/913dac25-4f03-48b7-8690-3c646bcf9629)
```
# Perform RobustScaler
robust = RobustScaler()
df_robust = robust.fit_transform(df[["Height", "Weight"]])
print("\nRobustScaler Result:\n", df_robust[:5])
```
![Screenshot 2025-04-23 154845](https://github.com/user-attachments/assets/720cebe9-82d7-425b-aa35-c793d927627f)
```
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE, SelectKBest
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, chi2
# Load Titanic dataset
df = pd.read_csv('/content/titanic_dataset.csv')
# Display column names
print(df.columns)
```
![Screenshot 2025-04-23 154941](https://github.com/user-attachments/assets/c8854e5f-2451-48ca-9a3f-77defaaa4466)
```
# Show shape of dataset
print("Shape:", df.shape)
```
![Screenshot 2025-04-23 155024](https://github.com/user-attachments/assets/9b2db40a-ad5e-41e9-b1b0-6e5d3148d2c1)
```
# Define features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]
# Drop irrelevant columns
df1 = df.drop(["Name", "Sex", "Ticket", "Cabin", "Embarked"], axis=1)
print("Missing Age values before:", df1['Age'].isnull().sum())
```
![Screenshot 2025-04-23 155109](https://github.com/user-attachments/assets/9e128820-461a-4310-b5c0-0f608c7becc0)
```
# Fill null values in Age using forward fill
df1['Age'] = df1['Age'].fillna(method='ffill')

# Check again
print("Missing Age values after:", df1['Age'].isnull().sum())

```
![Screenshot 2025-04-23 155201](https://github.com/user-attachments/assets/ef77afd1-d190-414e-b599-b3ccf4d4c73f)
```
# Apply SelectKBest for top 3 features
feature = SelectKBest(mutual_info_classif, k=3)
# Reorder columns as required
df1 = df1[['PassengerId', 'Fare', 'Pclass', 'Age', 'SibSp', 'Parch', 'Survived']]
# Define feature matrix and target vector
X = df1.iloc[:, 0:6]
y = df1.iloc[:, 6]
# Confirm columns
print("X Columns:", X.columns)
y = y.to_frame()
print("y Columns:", y.columns)
```
![Screenshot 2025-04-23 155302](https://github.com/user-attachments/assets/46422069-296e-4cbc-90c2-2f1199152d22)
```
# Fit SelectKBest
feature.fit(X, y.values.ravel())

```
![Screenshot 2025-04-23 155340](https://github.com/user-attachments/assets/ac3e360c-a95d-4fe4-9ef2-0699b36d0a21)
```
# Get selected feature scores
scores = pd.DataFrame({"Feature": X.columns, "Score": feature.scores_})
print("\nFeature Scores:\n", scores.sort_values(by="Score", ascending=False))
```
![Screenshot 2025-04-23 155416](https://github.com/user-attachments/assets/f674f710-88dc-44bc-bd4c-1357f2fc24b7)

# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
