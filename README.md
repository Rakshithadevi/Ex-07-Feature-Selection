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
![image](https://user-images.githubusercontent.com/94165326/169946821-1a4f8855-76e1-4704-a3d5-5b9533551ea7.png)
![image](https://user-images.githubusercontent.com/94165326/169947002-849d4dab-5bf2-4e0b-9fb8-1f4414a8adac.png)
![image](https://user-images.githubusercontent.com/94165326/169947043-00f51088-9b15-43da-8608-cbc1cda314cb.png)
![image](https://user-images.githubusercontent.com/94165326/169947180-40703226-d391-406e-89aa-fa4563f20405.png)
![image](https://user-images.githubusercontent.com/94165326/169947212-fa7344e3-3f6a-46e3-980a-9cd9b6d52f55.png)
![image](https://user-images.githubusercontent.com/94165326/169947242-b6615d8d-e7a0-4090-b20d-68f564b851ad.png)
![image](https://user-images.githubusercontent.com/94165326/169947262-de4ce4d1-f1a5-49ca-8eb0-87894950e6fd.png)
![image](https://user-images.githubusercontent.com/94165326/169947286-ee5d20e7-f7d1-4655-a092-60c4a4f6dc1f.png)

