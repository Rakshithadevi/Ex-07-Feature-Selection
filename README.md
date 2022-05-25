# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE
```
NAME: RAKSHITHA DEVI J
REG NO:212221230082
```
#Importing libraries
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.datasets import load_boston
boston = load_boston()
print(boston['DESCR'])
import pandas as pd
df = pd.DataFrame(boston['data'] )
df.head()
df.columns = boston['feature_names']
df.head()
df['PRICE']= boston['target']
df.head()
df.info()
plt.figure(figsize=(10, 8))
sns.distplot(df['PRICE'], rug=True)
plt.show()
```

#FILTER METHODS
```
X=df.drop("PRICE",1)
y=df["PRICE"]
from sklearn.feature_selection import SelectKBest, chi2
X, y = load_boston(return_X_y=True)
X.shape
#1.Variance Threshold
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold()
selector.fit_transform(X)
#2.Information gain/Mutual Information
from sklearn.feature_selection import mutual_info_regression
mi = mutual_info_regression(X, y);
mi = pd.Series(mi)
mi.sort_values(ascending=False)
mi.sort_values(ascending=False).plot.bar(figsize=(10, 4))
#3.SelectKBest Model
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest,SelectPercentile
skb = SelectKBest(score_func=f_classif, k=2)
X_data_new = skb.fit_transform(X, y)
print('Number of features before feature selection: {}'.format(X.shape[1]))
print('Number of features after feature selection:
{}'.format(X_data_new.shape[1]))
#4.Correlation Coefficient
cor=df.corr()
sns.heatmap(cor,annot=True)
#5.Mean Absolute Difference
mad=np.sum(np.abs(X-np.mean(X,axis=0)),axis=0)/X.shape[0]
plt.bar(np.arange(X.shape[1]),mad,color='teal')
#Processing data into array type.
from sklearn import preprocessing
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)
print(y_transformed)
#6.Chi Square Test
X = X.astype(int)
chi2_selector = SelectKBest(chi2, k=2)
X_kbest = chi2_selector.fit_transform(X, y_transformed)
print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_kbest.shape[1])
#7.SelectPercentile method
X_new = SelectPercentile(chi2, percentile=10).fit_transform(X, y_transformed)
X_new.shape
```


# OUPUT
## Importing required libraries:
![image](https://user-images.githubusercontent.com/94165326/170329433-3fd58495-de1e-4b49-8e1f-88270477aefa.png)

## Analyzing the boston dataset:
![image](https://user-images.githubusercontent.com/94165326/170329477-9c99cb0e-2bf2-4210-b3c8-b95c8203c615.png)
![image](https://user-images.githubusercontent.com/94165326/170329518-7ec33b4d-6872-43be-8381-c7004d0f3249.png)
![image](https://user-images.githubusercontent.com/94165326/170329557-7022a51c-772d-4645-a492-bf487f783108.png)
![image](https://user-images.githubusercontent.com/94165326/170329589-256f0fe9-f29a-4431-95fb-141ea401febf.png)
![image](https://user-images.githubusercontent.com/94165326/170329615-c3a1343c-f065-4e9e-8568-5da2dea51043.png)
![image](https://user-images.githubusercontent.com/94165326/170329655-c72ae568-5a30-47a2-9a9d-c7e452c0fa6c.png)
## Analyzing dataset using Distplot:
![image](https://user-images.githubusercontent.com/94165326/170329687-cd015c66-812b-41aa-91c9-6dafd1089dcc.png)
Filter Methods:
![image](https://user-images.githubusercontent.com/94165326/170330011-9cc7748c-5fe7-476d-939a-c333003b8bed.png)
![image](https://user-images.githubusercontent.com/94165326/170330039-d24b9616-c4b4-4c64-87fd-325ae8038523.png)
![image](https://user-images.githubusercontent.com/94165326/170330112-661c9b5d-aff0-401d-a864-b402e653ee86.png)
![image](https://user-images.githubusercontent.com/94165326/170330167-a58e4937-2e39-4da1-9f00-d465bd934030.png)
![image](https://user-images.githubusercontent.com/94165326/170330201-e3531aba-bd81-40cf-a78f-041935040ebc.png)
![image](https://user-images.githubusercontent.com/94165326/170330241-c4646ddf-878a-4c3e-9012-2bc3b94d1218.png)
![image](https://user-images.githubusercontent.com/94165326/170330284-e13a860d-2b5a-4020-ba3b-078e731239b9.png)
![image](https://user-images.githubusercontent.com/94165326/170330330-9fbd81f3-151d-4d7a-81fb-bc072449fabd.png)
![image](https://user-images.githubusercontent.com/94165326/170330372-4d5fbd76-b851-4311-9711-3b9e7412c20e.png)
![image](https://user-images.githubusercontent.com/94165326/170330421-8ca279d7-5e3e-4a7f-a6e3-394010de8c29.png)

## Wrapper Methods:
![image](https://user-images.githubusercontent.com/94165326/170330538-3dfc8919-d860-47aa-95f4-9a915e536e89.png)
![image](https://user-images.githubusercontent.com/94165326/170330633-e005260e-171a-49d1-b795-f80c42d0934a.png)
![image](https://user-images.githubusercontent.com/94165326/170330664-80e9ea74-6c52-470c-b489-ec5ffabd86cf.png)
![image](https://user-images.githubusercontent.com/94165326/170330711-2ce9954d-29f6-4f91-af21-473639bec467.png)
![image](https://user-images.githubusercontent.com/94165326/170330784-582077d4-7c40-4d1a-bdd2-5012510d0405.png)
![image](https://user-images.githubusercontent.com/94165326/170330834-a1791f58-837d-4e27-b8c0-51240da93a2e.png)
![image](https://user-images.githubusercontent.com/94165326/170330884-400c8aeb-6e2a-4c0c-8e3d-a8342c64ab57.png)
![image](https://user-images.githubusercontent.com/94165326/170330950-1452a7db-3d49-47ff-9531-2938fe5e6444.png)
![image](https://user-images.githubusercontent.com/94165326/170330975-ba19e01c-32f5-458d-a414-83e163691095.png)

## RESULT:
Hence various feature selection techniques are applied to the given data set successfully
and saved the data into a file.


