# EDA - Analysing the Titanic dataset

In this Notebook we will do an EDA (Exploratory Data Analysis) In the Titanic Dataset ([*link*](https://www.kaggle.com/c/titanic/data))

Following the PDSA Cycle, let's answer the 3 PDSA's questions:
* Objective: Understand the features; Clean the Dataset; Find a good ML algorithm;
* Methodology: Analyze the data, using different visualizantions and metrics; Apply transformations that will bring value to the data; Test a list of ML algorithms, with different parameters, to find out the best ML algorithm for this dataset;
* Espected Results: A better undertanding of the dataset and its limitations; A Cleaning process that will bring more value to the ML phase; Compare the ML Algorithms with diferent evaluations: ROC Curve, Accuracy and F1_score;

Let's start by loading the dataset and checking some informations.


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/titanic/train.csv')

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB


## Explanation about each variable

* PassengerId: unique identifier of the row
* Survived: Survival (0 = No; 1 = Yes)
* Pclass: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
* Name: Passenger Name
* Sex: Sex
* Age: Age
* SibSp: Number of Siblings/Spouses Aboard
* Parch: Number of Parents/Children Aboard
* Ticket: Ticket Number
* Fare: Passenger Fare
* Cabin: Cabin
* Embarked: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)


```python
df.head(8)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



## Understanding the variables

We can see above that there are 891 rows in the Dataset.

The ***PassengerId*** id column is a support column created by the owner of the dataset. Since we already have the dataframe index we can drop it.

Some of those rows have NA values, and they are ('Age','Cabin','Embarked'). Cabin is the most impacted variable with only 204 populated values in a total of 891. Since this variable is so damaged, let's also drop it. About the other 2, we can ignore the NA values while analysing (we still can use then in the ML phase).

There are also 2 columns that, in my opnion, don't bring any value to the analysis: 'Name' and 'Ticket'. So for now let's also drop them.

For the others non-Categorical variables, let's visulize the correlation for a better understanding.


```python
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True)
plt.show()
```


![png](assets/output_6_0.png)


Apparently the 'Survived' column have some correlations with 'Pclass' and 'Fare' columns. Let's check further this in the analysis section.

So Let's check now all the variables 


```python

sns.pairplot(df.loc[:, df.columns != 'PassengerId'], hue="Survived")
```




    <seaborn.axisgrid.PairGrid at 0x7fa58b69ea58>




![png](assets/output_8_1.png)


From the graphic above, it's important to notice that Categorical variables are not ploted.
For the 

I was thinking here, and I didn't get why the Age column was a float column instead of Integer. So let's check some informations:


```python
df.Age.describe()
```




    count    714.000000
    mean      29.699118
    std       14.526497
    min        0.420000
    25%       20.125000
    50%       28.000000
    75%       38.000000
    max       80.000000
    Name: Age, dtype: float64




```python
df.Age.plot.kde()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fa58aee6c88>




![png](assets/output_12_1.png)


The Min value is 0.42, and maybe they are also counting the Passenger Age months.


```python
df.sort_values(by=['Age'], ascending=True).Age
```




    803    0.42
    755    0.67
    644    0.75
    469    0.75
    78     0.83
           ... 
    859     NaN
    863     NaN
    868     NaN
    878     NaN
    888     NaN
    Name: Age, Length: 891, dtype: float64



## Preparing the data for the EDA

So let's drop some columns that we didn see any value in our analysis.


```python
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
```


```python
df.head(8)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>8.4583</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>1</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>51.8625</td>
      <td>S</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>21.0750</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 8 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   Survived  891 non-null    int64  
     1   Pclass    891 non-null    int64  
     2   Sex       891 non-null    object 
     3   Age       714 non-null    float64
     4   SibSp     891 non-null    int64  
     5   Parch     891 non-null    int64  
     6   Fare      891 non-null    float64
     7   Embarked  889 non-null    object 
    dtypes: float64(2), int64(4), object(2)
    memory usage: 55.8+ KB


## Starting the Analysis

So let's start creating some hypothesis and trying to prove them:
1. Did Age played a role in survival?
2. Did Sex played a role in survivel?
3. Did Class played a role in survivel?



### **Question 1**: Did Age played a role in survival?

So lets plot at the same time who survived and who not, related to Age.


```python
sns.distplot(df[df.Survived == 1].Age, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 2}, label = 'Survived')
sns.distplot(df[df.Survived == 0].Age, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 2}, label = 'Didn`t Survive')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fa58aafe438>




![png](assets/output_21_1.png)



```python
ax = sns.boxplot( x='Survived', y='Age', data=df)
```


![png](assets/output_22_0.png)


We can see that the youger from the range of 0-17 have a greater survival rate. For the other hand, the range between 18-35 have a lower survivavel rate. For the range of 36-60, the Survival rate are around the same, with minor difference. And for the range 60+, the survival rate droped again.

So Answering the question, we could say that **Yes**, the Age played a role in the survival rate.

### **Question 2**: Did Sex played a role in survival?

Now let's check the Survival rate compared with the Sex.


```python
ct = pd.crosstab(df.Sex, df.Survived.map({0:'No', 1:'Yes'}))
ct.plot.bar(stacked=True)

ct = pd.crosstab(df.Survived.map({0:'No', 1:'Yes'}), df.Sex)
ct.plot.bar(stacked=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fa5a639d0b8>




![png](assets/output_25_1.png)



![png](assets/output_25_2.png)



```python
df.Survived.sum()/df.Survived.count()
```




    0.3838383838383838



From the above plot we can confirm 2 things:
1. The Male Survival rate are lower than the Female Survival Rate.
2. Only 38% of the passengers survived this tragedy.

So, answering the question: **Yes**, the Sex played a role in the survival rate.

### **Question 3**: Did Class played a role in survival?

We can start checking the Survival rate for each class.


```python
sns.distplot(df[df.Survived == 1].Pclass, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 2}, label = 'Survived')
sns.distplot(df[df.Survived == 0].Pclass, hist = False, kde = True, kde_kws = {'shade': True, 'linewidth': 2}, label = 'Don`t Survived')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fa5a62615c0>




![png](assets/output_29_1.png)



```python
ct = pd.crosstab(df.Pclass, df.Survived.map({0:'No', 1:'Yes'}))
ct.plot.bar(stacked=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fa5a61e32e8>




![png](assets/output_30_1.png)



```python
sns.countplot('Pclass',hue='Survived',data=df)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fa5a61cd550>




![png](assets/output_31_1.png)



```python
for pc in range(1, 4):
    print('Class {0} survival ratio: {1:2.4f}'.format(pc, df[(df.Pclass==pc)].Survived.sum()/df[(df.Pclass==pc)].Survived.count()))
```

    Class 1 survival ratio: 0.6296
    Class 2 survival ratio: 0.4728
    Class 3 survival ratio: 0.2424


It's possible to see a difference in the Survival ratio between the 3 classes of passengers.
So we can also answer the question with an **Yes**, the Pclass played a role in the Survival rate.

## Creating a Machine Learning model

So now let's create a Classifier using some algorithms like:
* Random Forest
* SVM - Linear Kernel
* SVM - Radial Kernel
* XGboost
* Decision Tree
* Naive Bayes

But first we need to prepare the data to be used by the ML algorithms. The preparations are:
* Drop some columns
* Create new support columns
* Fill the NA values or drop the column
* Transform some categorical values into ordenal/numerical values

### Preparing the Data


```python
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
```


```python
df = pd.read_csv('data/titanic/train.csv')

def clean_data(_df):
    # Age - Fill NA
    _df.loc[(_df.Age.isnull()) & (_df.Pclass==1), 'Age'] = 38
    _df.loc[(_df.Age.isnull()) & (_df.Pclass==2), 'Age'] = 30
    _df.loc[(_df.Age.isnull()) & (_df.Pclass==3), 'Age'] = 25

    # Sex - Replace 0 1
    _df.Sex = _df.Sex.map({'female':0, 'male':1})

    # Embarked - Replace 0 1 2 and Fill NA
    _df.loc[_df.Embarked.isnull(), 'Embarked'] = 'S'
    _df.Embarked = _df.Embarked.map({'C':0, 'Q':1, 'S':2})

    # Drop columns that we will not use
    _df = _df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
    
    return _df

df = clean_data(df)

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 8 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   Survived  891 non-null    int64  
     1   Pclass    891 non-null    int64  
     2   Sex       891 non-null    int64  
     3   Age       891 non-null    float64
     4   SibSp     891 non-null    int64  
     5   Parch     891 non-null    int64  
     6   Fare      891 non-null    float64
     7   Embarked  891 non-null    int64  
    dtypes: float64(2), int64(6)
    memory usage: 55.8 KB



```python
train, test = train_test_split(df, test_size=0.3, random_state=0, stratify=df['Survived'])
train_X = train[train.columns[1:]]
train_Y = train.Survived
test_X = test[test.columns[1:]]
test_Y = test.Survived
```


```python
# Random Forest Classifier
model = RandomForestClassifier(n_estimators=1000)
model.fit(train_X, train_Y)
pred = model.predict(test_X)
metrics.accuracy_score(pred, test_Y)
```




    0.8283582089552238




```python
# SVM - RBF Kernel
model = svm.SVC(kernel = 'rbf', C = 2, gamma = 'auto')
model.fit(train_X, train_Y)
pred = model.predict(test_X)
metrics.accuracy_score(pred, test_Y)
```




    0.7313432835820896




```python
# SVM - Linear Kernel
model = svm.SVC(kernel = 'linear', C = 0.1)
model.fit(train_X, train_Y)
pred = model.predict(test_X)
metrics.accuracy_score(pred, test_Y)
```




    0.7910447761194029




```python
# XGBoost
model = XGBClassifier()
model.fit(train_X, train_Y)
pred = model.predict(test_X)
metrics.accuracy_score(pred, test_Y)
```




    0.835820895522388




```python
# Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(train_X, train_Y)
pred = model.predict(test_X)
metrics.accuracy_score(pred, test_Y)
```




    0.8097014925373134




```python
 skein funcio# Naive Bayes - GaussianNB
model = DecisionTreeClassifier()
model.fit(train_X, train_Y)
pred = model.predict(test_X)
metrics.accuracy_score(pred, test_Y)
```




    0.8097014925373134



So now, let's try with the real test database. For this we need to apply the same transformation and cleaning process


```python
train_X = df[df.columns[1:]]
train_Y = df['Survived']
test = pd.read_csv('data/titanic/test.csv')
test_label = pd.read_csv('data/titanic/gender_submission.csv')
```


```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 11 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  418 non-null    int64  
     1   Pclass       418 non-null    int64  
     2   Name         418 non-null    object 
     3   Sex          418 non-null    object 
     4   Age          332 non-null    float64
     5   SibSp        418 non-null    int64  
     6   Parch        418 non-null    int64  
     7   Ticket       418 non-null    object 
     8   Fare         417 non-null    float64
     9   Cabin        91 non-null     object 
     10  Embarked     418 non-null    object 
    dtypes: float64(2), int64(4), object(5)
    memory usage: 36.0+ KB


There is one NA value in the Fare column. The Age and Cabin will be already treated by our cleaning algorithm. So for now, let's populate that column with the mean of Fare for other Passangers for the same PClass.


```python
# test.loc[(~test.Fare.isnull()) & (test.Pclass==3), 'Fare'].mean()
test.loc[(test.Fare.isnull()) & (test.Pclass==3), 'Fare'] = 12.45
```


```python
test = pd.merge(test, test_label, on="PassengerId")

test = clean_data(test)

test_X = test[test.columns[:-1]]
test_Y = test.Survived
```


```python
# Random Forest Classifier
model = RandomForestClassifier(n_estimators=1000)
model.fit(train_X, train_Y)
pred = model.predict(test_X)
metrics.accuracy_score(pred, test_Y)
```




    0.8157894736842105




```python
# Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(train_X, train_Y)
pred = model.predict(test_X)
metrics.accuracy_score(pred, test_Y)
```




    0.7703349282296651




```python
# XGBoost
model = XGBClassifier()
model.fit(train_X, train_Y)
pred = model.predict(test_X)
print(metrics.accuracy_score(pred, test_Y))
pred_xgboost = pred
test_Y_xgboost = test_Y
```

    0.8229665071770335


Let's check the precision and recall of the best classifier. In this case the XGBoost
* Precision: Rate of positivies classified correctly between all the **predicted** positives
* Recall (TPR): Rate of positivies classified correctly between all the **true** positives
* f1_score: Since Precision and Recall show differents behaviors, it's not a good idea to analise them alone. So for this we can use the F_score metric. If the weight is 1, the metric is the f1_score: (2 * Prec(f) * Rec(f))/(Prec(f) + Rec(f))


```python
print('>>> Each Class <<<')
# Calculate the metrics per class.
precision, recall, f1_score, labels = precision_recall_fscore_support(pred_xgboost, test_Y_xgboost, beta=1, average=None)
print(f'Precision:{precision}')
print(f'Recall:{recall}')
print(f'F1_score:{f1_score}')
print(f'Labels:{labels}')
```

    >>> Each Class <<<
    Precision:[0.85338346 0.76973684]
    Recall:[0.86641221 0.75      ]
    F1_score:[0.85984848 0.75974026]
    Labels:[262 156]



```python
print('>>> Micro <<<')
# Calculate metrics globally by counting the total true positives, false negatives and false positives.
precision, recall, f1_score, labels = precision_recall_fscore_support(pred_xgboost, test_Y_xgboost, beta=1, average='micro')
print(f'Precision:{precision}')
print(f'Recall:{recall}')
print(f'F1_score:{f1_score}')
print(f'Labels:{labels}')
```

    >>> Micro <<<
    Precision:0.8229665071770335
    Recall:0.8229665071770335
    F1_score:0.8229665071770335
    Labels:None



```python
print('>>> Macro <<<')
# Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
precision, recall, f1_score, labels = precision_recall_fscore_support(pred_xgboost, test_Y_xgboost, beta=1, average='macro')
print(f'Precision:{precision}')
print(f'Recall:{recall}')
print(f'F1_score:{f1_score}')
print(f'Labels:{labels}')
```

    >>> Macro <<<
    Precision:0.8115601503759399
    Recall:0.8082061068702291
    F1_score:0.8097943722943722
    Labels:None


We could notice that the classes are not balanced. But getting the F1_Score for the 'Macro' Precision and Recall, we can see a rate of around 0.81%

The below graph is the ROC Curve, and it's constructed using the FPR and TPR.


```python
fpr, tpr, thresholds = roc_curve(pred_xgboost, test_Y_xgboost)
print(f'FPR: {fpr[1]}')
print(f'TPR: {tpr[1]}')
```

    FPR: 0.13358778625954199
    TPR: 0.75



```python
plt.plot(fpr,tpr)
plt.plot([0,1],[0,1])
```




    [<matplotlib.lines.Line2D at 0x7fa58f27dd68>]




![png](assets/output_60_1.png)




## Conclusion


In this analysis we could notice some features that impacts the survival rate. They are: Age, Sex and Pclass.

For the classifier point of view, there are some observations:
- XGboost showed the best accuracy with 83.5% in the train data and 82.3% in the test data.
- Random Forest algorithm is also good choice, but it showed that depending how you train the model, the accuracy changes between 80% and 83%
- SVM RBF didn't separate well that data and was placed with the worst accuracy. Apparently the RBF Kernel is not a good choice.
- In general, the best percentage was 82.3%. 

The classifier validation methods are: 
* accuracy_score
* Precision, Recall and F1_score
* ROC Curve


For the future work, I would review the cleaning process. I know that there are more information in the dropped featurs that we could use. Also, we could do some other tries like: Create some support features and merge with external sources. In this disaster, the cabins also played a great role. If some how we could use the position of the ship's cabin
