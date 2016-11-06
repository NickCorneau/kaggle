
# Reading in the Data


```python
import csv as csv
import numpy as np
import pandas as pd

df = pd.read_csv('csv/train.csv',header=0)
df.head()
```




<div>
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
  </tbody>
</table>
</div>



# Identifying columns that aren't Float type or with NaN's


```python
for col in df:
    if (df[col].dtypes != "float64"):
        print col, df[col].dtypes
    if ((len(df[df[col].isnull() == True])) > 0):
        print col, "Contains NaN's"
```

    PassengerId int64
    Survived int64
    Pclass int64
    Name object
    Sex object
    Age Contains NaN's
    SibSp int64
    Parch int64
    Ticket object
    Cabin object
    Cabin Contains NaN's
    Embarked object
    Embarked Contains NaN's


# Dealing with NaN's in Age

I will replace the NaN's in Age with the median age of each passenger based off their respective gender and class.


```python
median_ages = np.zeros((2,3))
median_ages
```




    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]])



Map passenger's gender to 0 or 1 in order to map gender and class later on.


```python
df['Sex'] = df['Sex'].map({'female':0, 'male':1}).astype(int)
df.head()
```




<div>
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
      <td>1</td>
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
      <td>0</td>
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
      <td>0</td>
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
      <td>0</td>
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
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Calculate median age by gender and class.


```python
for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = df[(df['Sex'] == i) & \
                              (df['Pclass'] == j+1)]['Age'].dropna().median()
median_ages
```




    array([[ 35. ,  28. ,  21.5],
           [ 40. ,  30. ,  25. ]])



Replace NaN's with median age for passenger's gender and class


```python
for i in range(0,2):
    for j in range(0,3):
        df.loc[(df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j+1), \
                'Age'] = median_ages[i,j]
df.head()
```




<div>
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
      <td>1</td>
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
      <td>0</td>
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
      <td>0</td>
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
      <td>0</td>
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
      <td>1</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



Confirm Age column is no longer empty


```python
# Checks for empty columns in df
for column in df:
    if (len(df[df[column].isnull() == True])) > 0:
        print column
```

    Cabin
    Embarked


# Converting int64 col's to float


```python
for col in df:
    if (df[col].dtypes == "int64"):
        df[col] = df[col].astype(float)
```

Confirm int's are converted.


```python
for col in df:
    if (df[col].dtypes != "float64"):
        print col, df[col].dtypes
    if ((len(df[df[col].isnull() == True])) > 0):
        print col, "Contains NaN's"
```

    Name object
    Ticket object
    Cabin object
    Cabin Contains NaN's
    Embarked object
    Embarked Contains NaN's



```python
df.head()
```




<div>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>1.0</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0.0</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0.0</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0.0</td>
      <td>35.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>Allen, Mr. William Henry</td>
      <td>1.0</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



# Converting object col's to float

### Name


```python
# Name: Nothing to do with name and can't be converted.
df = df.drop('Name',axis=1)
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>35.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



### Embarked


```python
# Embarked: Only three ports, so map to 0 1 2.
df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(float)
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
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
      <td>1.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>35.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
median_embarked = np.zeros((2,3))
median_embarked
```




    array([[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]])




```python
for i in range(0,2):
    for j in range(0,3):
        median_embarked[i,j] = df[(df['Sex'] == i) & \
                              (df['Pclass'] == j+1)]['Embarked'].dropna().mean()
median_embarked
```




    array([[ 0.48913043,  0.14473684,  0.61805556],
           [ 0.36065574,  0.11111111,  0.34870317]])




```python
# Take the average port embarked by passengers based on their class and sex
for i in range(0,2):
    for j in range(0,3):
        median_embarked[i,j] = round(median_embarked[i,j] * 3)
median_embarked        
```




    array([[ 1.,  0.,  2.],
           [ 1.,  0.,  1.]])




```python
for i in range(0,2):
    for j in range(0,3):
        df.loc[(df.Embarked.isnull()) & (df.Sex == i) & (df.Pclass == j+1), \
                'Embarked'] = median_embarked[i,j]
```

Confirm embarked is all float and no more NaN's.


```python
for col in df:
    if (df[col].dtypes != "float64"):
        print col, df[col].dtypes
    if ((len(df[df[col].isnull() == True])) > 0):
        print col, "Contains NaN's"
```

    Ticket object
    Cabin object
    Cabin Contains NaN's


### Ticket


```python
import re
# Strip all the prefixes
df['Ticket'] = df['Ticket'].map(lambda x: (re.sub(".*? (.+)", "\\1", x)))

# Define function that returns 0 if it can't convert to type
def tryconvert(value, default, *types):
    for t in types:
        try:
            return t(value)
        except ValueError, TypeError:
            continue
    return default

df['Ticket'] = df['Ticket'].map(lambda x: tryconvert(x,0,float))
```

Confirm Ticket is a float.


```python
for col in df:
    if (df[col].dtypes != "float64"):
        print col, df[col].dtypes
    if ((len(df[df[col].isnull() == True])) > 0):
        print col, "Contains NaN's"
```

    Cabin object
    Cabin Contains NaN's


### Cabin


```python
# Drop cabin because not much can be done with it right now.
df = df.drop('Cabin', axis=1)
```

Confirm everything is a float


```python
for col in df:
    if (df[col].dtypes != "float64"):
        print col, df[col].dtypes
    if ((len(df[df[col].isnull() == True])) > 0):
        print col, "Contains NaN's"
```

# Random Forests! Finally!


```python
df = df.drop('Ticket',axis=1)
df = df.drop('PassengerId', axis=1)
```


```python
df
```




<div>
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
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>22.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.2500</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>71.2833</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.9250</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>35.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>53.1000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0500</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.4583</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>54.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>51.8625</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>21.0750</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>27.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>11.1333</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>14.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>30.0708</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>16.7000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>58.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>26.5500</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0500</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>39.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>31.2750</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.8542</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>55.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>16.0000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>29.1250</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13.0000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>31.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>18.0000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>21.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.2250</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>26.0000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>34.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13.0000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0292</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>35.5000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>21.0750</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>31.3875</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.2250</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>19.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>263.0000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>21.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.8792</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.8958</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>861</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>21.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>11.5000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>862</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>48.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>25.9292</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>863</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>21.5</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>69.5500</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>864</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13.0000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>865</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>42.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13.0000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>866</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>27.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>13.8583</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>867</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>50.4958</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>868</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.5000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>869</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>11.1333</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>870</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.8958</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>871</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>47.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>52.5542</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>872</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>33.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>873</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>47.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.0000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>874</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>28.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>24.0000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>875</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.2250</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>876</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.8458</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>877</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>19.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.8958</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>878</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.8958</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>879</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>56.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>83.1583</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>880</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>26.0000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>881</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>33.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.8958</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>882</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>22.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.5167</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>883</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.5000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>884</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0500</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>885</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>39.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>29.1250</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>886</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>27.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13.0000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>887</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>30.0000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>888</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>21.5</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>23.4500</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>889</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>30.0000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>32.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.7500</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 8 columns</p>
</div>




```python
from sklearn.ensemble import RandomForestClassifier

# Clean up TEST DATA
test_df = pd.read_csv('csv/test.csv', header=0)        # Load the test file into a dataframe
train_df = pd.read_csv('csv/train.csv', header=0)
# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports } 

# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)

# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 
```

    /Users/nicholas.corneau/anaconda/envs/titanic/lib/python2.7/site-packages/numpy/lib/arraysetops.py:200: FutureWarning: numpy not_equal will not check object identity in the future. The comparison did not return the same result as suggested by the identity (`is`)) and will change.
      flag = np.concatenate(([True], aux[1:] != aux[:-1]))



```python
# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 

# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = df.values
test_data = test_df.values


print 'Training...'
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print 'Predicting...'
output = forest.predict(test_data).astype(int)


predictions_file = open("csv/myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
```

    Training...
    Predicting...
    Done.



```python

```
