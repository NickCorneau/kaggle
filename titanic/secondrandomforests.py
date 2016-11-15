
# coding: utf-8

# # Reading in the Data

# In[1]:

import csv as csv
import numpy as np
import pandas as pd

df = pd.read_csv('csv/train.csv',header=0)
df.head()


# # Identifying columns that aren't Float type or with NaN's

# In[2]:

for col in df:
    if (df[col].dtypes != "float64"):
        print col, df[col].dtypes
    if ((len(df[df[col].isnull() == True])) > 0):
        print col, "Contains NaN's"


# # Dealing with NaN's in Age

# I will replace the NaN's in Age with the median age of each passenger based off their respective gender and class.

# In[3]:

median_ages = np.zeros((2,3))
median_ages


# Map passenger's gender to 0 or 1 in order to map gender and class later on.

# In[4]:

df['Sex'] = df['Sex'].map({'female':0, 'male':1}).astype(int)
df.head()


# Calculate median age by gender and class.

# In[5]:

for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = df[(df['Sex'] == i) &                               (df['Pclass'] == j+1)]['Age'].dropna().median()
median_ages


# Replace NaN's with median age for passenger's gender and class

# In[6]:

for i in range(0,2):
    for j in range(0,3):
        df.loc[(df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j+1),                 'Age'] = median_ages[i,j]
df.head()


# Confirm Age column is no longer empty

# In[7]:

# Checks for empty columns in df
for column in df:
    if (len(df[df[column].isnull() == True])) > 0:
        print column


# # Converting int64 col's to float

# In[8]:

for col in df:
    if (df[col].dtypes == "int64"):
        df[col] = df[col].astype(float)


# Confirm int's are converted.

# In[9]:

for col in df:
    if (df[col].dtypes != "float64"):
        print col, df[col].dtypes
    if ((len(df[df[col].isnull() == True])) > 0):
        print col, "Contains NaN's"


# In[10]:

df.head()


# # Converting object col's to float

# ### Name

# In[11]:

# Name: Nothing to do with name and can't be converted.
df = df.drop('Name',axis=1)
df.head()


# ### Embarked

# In[12]:

# Embarked: Only three ports, so map to 0 1 2.
df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(float)
df.head()


# In[13]:

median_embarked = np.zeros((2,3))
median_embarked


# In[14]:

for i in range(0,2):
    for j in range(0,3):
        median_embarked[i,j] = df[(df['Sex'] == i) &                               (df['Pclass'] == j+1)]['Embarked'].dropna().mean()
median_embarked


# In[15]:

# Take the average port embarked by passengers based on their class and sex
for i in range(0,2):
    for j in range(0,3):
        median_embarked[i,j] = round(median_embarked[i,j] * 3)
median_embarked        


# In[16]:

for i in range(0,2):
    for j in range(0,3):
        df.loc[(df.Embarked.isnull()) & (df.Sex == i) & (df.Pclass == j+1),                 'Embarked'] = median_embarked[i,j]


# Confirm embarked is all float and no more NaN's.

# In[17]:

for col in df:
    if (df[col].dtypes != "float64"):
        print col, df[col].dtypes
    if ((len(df[df[col].isnull() == True])) > 0):
        print col, "Contains NaN's"


# ### Ticket

# In[18]:

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


# Confirm Ticket is a float.

# In[19]:

for col in df:
    if (df[col].dtypes != "float64"):
        print col, df[col].dtypes
    if ((len(df[df[col].isnull() == True])) > 0):
        print col, "Contains NaN's"


# ### Cabin

# In[20]:

# Drop cabin because not much can be done with it right now.
df = df.drop('Cabin', axis=1)


# Confirm everything is a float

# In[21]:

for col in df:
    if (df[col].dtypes != "float64"):
        print col, df[col].dtypes
    if ((len(df[df[col].isnull() == True])) > 0):
        print col, "Contains NaN's"


# # Random Forests! Finally!

# In[22]:

df = df.drop('Ticket',axis=1)
df = df.drop('PassengerId', axis=1)


# In[23]:

df


# In[24]:

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


# In[25]:

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


# In[ ]:



