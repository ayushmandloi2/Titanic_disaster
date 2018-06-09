
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns


# In[75]:


data_train = pd.read_csv("G:/ml/titanic disaster/train.csv")
data_test = pd.read_csv("G:/ml/titanic disaster/test.csv")
combined = [data_train,data_test]


# In[76]:


data_train.count()


# In[77]:


data_train["Age"]=data_train["Age"].fillna(data_train["Age"].median())
data_train["Cabin"]=data_train["Cabin"].fillna(0)
data_train["Embarked"]= data_train["Embarked"].fillna("S")


# In[78]:


data_train.count()


# In[79]:


data_test.count()


# In[80]:


data_test['Age']= data_test['Age'].fillna(data_test['Age'].median())
data_test['Fare']=data_test['Fare'].fillna(data_test['Fare'].median())


# In[81]:


data_test.count()


# In[82]:


data_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[83]:


data_train['AgeBand'] = pd.cut(data_train['Age'], 5)
data_train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
data_test['AgeBand'] = pd.cut(data_test['Age'], 5)


# In[84]:


data_train.loc[data_train['Age']<= 16, "Age"]=0
data_train.loc[(data_train['Age']>= 16) & (data_train['Age']<= 32), "Age"]=1
data_train.loc[(data_train['Age']>= 32) & (data_train['Age']<= 48), "Age"]=2
data_train.loc[(data_train['Age']>= 48) & (data_train['Age']<= 64), "Age"]=3
data_train.loc[(data_train['Age']>= 64), "Age"] = 4
data_test.loc[data_test['Age']<= 16, "Age"]=0
data_test.loc[(data_test['Age']>= 16) & (data_test['Age']<= 32), "Age"]=1
data_test.loc[(data_test['Age']>= 32) & (data_test['Age']<= 48), "Age"]=2
data_test.loc[(data_test['Age']>= 48) & (data_test['Age']<= 64), "Age"]=3
data_test.loc[(data_test['Age']>= 64), "Age"] = 4
data_train.head()


# In[85]:


data_train['Sex'] = data_train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
data_test['Sex'] = data_test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


# In[86]:


data_train= data_train.drop("AgeBand",axis=1)
data_test= data_test.drop("AgeBand",axis=1)



# In[87]:


data_test.head()


# In[88]:


data_train[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[89]:


data_train["FamilySize"] = data_train["SibSp"] + data_train["Parch"] + 1
data_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
data_train = data_train.drop(["SibSp", "Parch"],axis =1)


# In[90]:


data_train['Title'] = data_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(data_train['Title'], data_train['Sex'])


# In[91]:


data_train['Title'] = data_train['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

data_train['Title'] = data_train['Title'].replace('Mlle', 'Miss')
data_train['Title'] = data_train['Title'].replace('Ms', 'Miss')
data_train['Title'] = data_train['Title'].replace('Mme', 'Mrs')
    
title_mapping = {"Mr.": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    
data_train['Title'] = data_train['Title'].map(title_mapping)
data_train['Title'] = data_train['Title'].fillna(0)

data_train.head()
data_train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
data_train =data_train.drop(["Name"], axis =1)


# In[92]:


data_train =data_train.drop(["Embarked","Ticket","Cabin"], axis=1)


# In[93]:



data_test= data_test.drop(["Ticket","Cabin"], axis =1)


# In[94]:


data_test.head()


# In[95]:


data_test["FamilySize"] = data_test["SibSp"] + data_test["Parch"] + 1


# In[96]:


data_test.head()


# In[97]:


data_test = data_test.drop(["Embarked","SibSp","Parch"],axis=1)


# In[98]:


data_test.head()


# In[99]:


data_test['Title'] = data_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(data_test['Title'], data_test['Sex'])
data_test['Title'] = data_test['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

data_test['Title'] = data_test['Title'].replace('Mlle', 'Miss')
data_test['Title'] = data_test['Title'].replace('Ms', 'Miss')
data_test['Title'] = data_test['Title'].replace('Mme', 'Mrs')
    
title_mapping = {"Mr.": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    
data_test['Title'] = data_test['Title'].map(title_mapping)
data_test['Title'] = data_test['Title'].fillna(0)


# In[100]:


data_test = data_test.drop("Name",axis =1)


# In[101]:



X_train = data_train.drop(["Survived","PassengerId"], axis=1)
Y_train = data_train["Survived"]
X_test  = data_test.drop("PassengerId", axis=1).copy()


# In[102]:


X_train.head()


# In[103]:


X_test.head()


# In[104]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train,Y_train)
pred = clf.predict(X_test)


# In[105]:


acc = round(clf.score(X_train, Y_train) * 100, 2)
print acc


# In[106]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train,Y_train)
pred = clf.predict(X_test)
abc = round(clf.score(X_train, Y_train) * 100, 2)
print abc


# In[107]:


submission = pd.DataFrame({
        "PassengerId":data_test["PassengerId"],
        "Survived": pred
    })
submission.to_csv('G:/ml/titanic disaster/submission7.csv', index=False)

