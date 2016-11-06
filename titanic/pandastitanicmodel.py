
# coding: utf-8

# # Numpy

# In[10]:

import csv as csv
import numpy as np

csv_file_object = csv.reader(open('csv/train.csv','rb'))
header = csv_file_object.next()
data=[]

for row in csv_file_object:
    data.append(row)
data = np.array(data)

print(data)


# In[ ]:




# In[11]:

data[0:15,5]


# In[12]:

type(data[0::,5])


# In[13]:

ages_onboard = data[0::,5].astype(np.float)


# # Playing with Pandas

# In[15]:

import pandas as pd

# Always use header = 0 when you know row 0 is the header row.
df = pd.read_csv('csv/train.csv',header=0)

df


# In[16]:

df.head(3)


# In[17]:

df.dtypes


# In[18]:

df.info()


# In[19]:

df.describe()


# # Data Munging

# In[20]:

df['Age'][0:10]


# In[21]:

df['Cabin'][0:10]


# In[22]:

df['Age'].mean()


# In[23]:

df[['Sex','Pclass','Age']]


# In[25]:

df[df['Age']>60][['Sex','Pclass','Age','Survived']]


# In[28]:

df[df['Age'].isnull()][['Sex','Pclass','Age','Survived']]


# In[31]:

for i in range(1,4):
    print i, len(df[(df['Sex']=='male') & (df['Pclass']==i)])


# In[34]:

import pylab as P
df['Age'].hist(bins=16,range=(0,80),alpha=.5)
P.show()


# # Cleaning the Data

# In[38]:

df['Gender']= df['Sex'].map({'female':0, 'male':1}).astype(int)
df.head(3)


# In[40]:

median_ages = np.zeros((2,3))
median_ages


# In[41]:

for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = df[(df['Gender'] == i) &                               (df['Pclass'] == j+1)]['Age'].dropna().median()
median_ages


# In[42]:

df['AgeFill'] = df['Age']
df.head()


# In[45]:

df[df['Age'].isnull()][['Gender','Pclass','Age','AgeFill']].head(10)


# In[47]:

for i in range(0,2):
    for j in range(0,3):
        df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),                 'AgeFill'] = median_ages[i,j]
df[df['Age'].isnull()][['Gender','Pclass','Age','AgeFill']].head(10)


# In[48]:

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
df.describe()


# # Feature Engineering

# In[64]:

df['FamilySize'] = df['SibSp'] + df['Parch']
df['FamilySize'].plot.hist()
df.plot.scatter( x = 'Pclass', y = 'FamilySize')
P.show()


# In[73]:

df['Age*Class'] = df.AgeFill * df.Pclass
df['Age*Class'].hist()
P.show()


# # Final preparation

# In[74]:

df.dtypes


# In[75]:

df.dtypes[df.dtypes.map(lambda x: x=='object')]


# In[81]:

df = df.dropna()


# In[82]:

train_data = df.values
train_data


# In[83]:

data


# In[ ]:



