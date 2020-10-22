#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Basic feature engineering with the Titanic Dataset
# Credentials: kasham1991@gmail.com / Karan Sharma

# Know more here https://www.kaggle.com/c/titanic/overview


# In[2]:


# Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Loading the datset

train = pd.read_csv("C://Datasets//titanic_train.csv")
test = pd.read_csv("C://Datasets//titanic_test.csv")


# In[4]:


# The survived column is the predicted variable, it is present in the training set only
# There are a lot of categorical variables

train.head()
#test.head()
#train.shape
#test.shape


# In[5]:


train.columns
#test.columns


# In[6]:


# Looking at the basic statistics
# There exisis null values
#train.describe

train.info()
train.describe().T


# In[7]:


# Removal of null values
# Age, cabin and embarked have null values
# We will deal with these later post categorical to numeric conversion
# If we do it now, we will lose significant amount of data

print(train.isna().sum())
print(test.isna().sum())


# In[8]:


# Converting sex into single numeric with a simple map function

def numeric_sex(data):
    map_sex = {'male': 0,'female': 1}
    data['Sex'].fillna('male', inplace = True)
    data['Sex'] = data['Sex'].map(map_sex)
    data['Sex'].astype(int)


# In[9]:


# Converting fare into single numeric with a simple map function
# The new range of fare will be 0|1|2|3

def numeric_fare(data):
    data.loc[data['Fare'] <= 7.91,'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2
    data.loc[ data['Fare'] > 31, 'Fare'] = 3
    data['Fare'] = data['Fare'].astype(int)


# In[10]:


# Converting cabin into single numeric with a simple map function
# Filling NaN with 0
# Using lambda if x is not equal to 0, label it 1; else label it 0
# Renaming cabin to has cabin

def numeric_cabin(data):
    data['Cabin'].fillna(0, inplace = True)
    data['Has_cabin'] = data['Cabin'].apply(lambda x: 1 if x != 0 else 0)


# In[11]:


# Converting embarked into single numeric with a simple map function
# C = Cherbourg, Q = Queenstown, S = Southampton with 1 | 2 | 3
# Filling NaN with S == 0, since this is the origin of the Titanic

def numeric_embarked(data):
    map_embark = {'S': 0, 'C': 1, 'Q': 2}
    data['Embarked'] = data['Embarked'].fillna('S')
    data['Embarked'] = data['Embarked'].map(map_embark)


# In[12]:


# Converting age into single numeric with a simple map function
# The new range of age will be 0|1|2|3|4

def numeric_age(data):
    data['Age'].fillna(0,inplace=True)
    data.loc[ data['Age'] <= 16, 'Age'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[ data['Age'] > 64, 'Age'] = 4 


# In[13]:


# Creating a new feature set and column FamilySize 
# Is_Alone will be a combination of SibSp and Parch
# Sibsp is the no of siblings/spouses aboard
# Parch is the no of parents/children aboard
# If family size is greater than 0, the passenger is not alone

def numeric_family(data):
    data['Family_size'] = data['SibSp'] + data['Parch']
    data['Family_size'] = data['Family_size'].astype(int)
    data['Is_Alone'] = 0
    data.loc[data['Family_size'] >0,'Is_Alone'] = 0


# In[14]:


# Removing titles from passenger names

def Remove_title(data):
    title = []
    name = data['Name']
    name = name.str.split('.')
    for i in name:
        title.append(i[0].split(',')[1].strip())
    data['Title'] = title
    data['Title'] = data['Title'].replace(['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'the Countess',
       'Jonkheer'], 'Rare')

    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')


# In[15]:


# Converting the titles to single numeric
# The new range of title will be 1|2|3|4|5
# Filling the NaN with 0

def numeric_title(data):
    map_title = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    data['Title'] = data['Title'].map(map_title)
    data['Title'] = data['Title'].fillna(0)


# In[16]:


# Creating a master fucntion 

def numeric_master(dataset):
    numeric_sex(dataset)
    numeric_fare(dataset)
    numeric_cabin(dataset)
    numeric_embarked(dataset)
    numeric_age(dataset)
    numeric_family(dataset)
    Remove_title(dataset)
    numeric_title(dataset)


# In[17]:


# Replacing the remainder NaNs
# Filling fare with median since there are extreme values in that column

train['Embarked'] = train['Embarked'].fillna('S')
numeric_master(train)

test['Fare'] = test['Fare'].fillna(test['Fare'].median())
numeric_master(test)


# In[18]:


# Creating the final dataset
Remove = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch']
train = train.drop(Remove, axis = 1)
test = test.drop(Remove, axis = 1)
#train.head()
#test.head()

x = train.drop('Survived', axis = 1)
y = train['Survived']


# In[19]:


# Splitting the dataset
# Modeling with Logistic Regression
# Creating a classification report

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


# In[20]:


model = LogisticRegression()
model.fit(x_train, y_train)


# In[21]:


y_predict = model.predict(x_test)
print("Accuracy Score is {}".format(accuracy_score(y_test, y_predict)))


# In[22]:


print(classification_report(y_test, y_predict))
print("ROC AUC Score is {}".format(roc_auc_score(y_test, y_predict)))


# In[23]:


# Thank You!

