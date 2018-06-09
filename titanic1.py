
# coding: utf-8

# In[307]:


import numpy as np 
import pandas as pd


# In[400]:


train = pd.read_csv("G:/ml/titanic disaster/train.csv")
test = pd.read_csv("G:/ml/titanic disaster/test.csv")


# In[401]:


len(train)
len(train[train['Survived']==1])


# In[402]:


print(100*np.mean(train['Survived'][train['Sex'] =='male']))
print(100*np.mean(train['Survived'][train['Sex']=='female']))


# In[403]:


print(100*np.mean(train['Survived'][train['Pclass']==1]))
print(100*np.mean(train['Survived'][train['Pclass']==3]))


# In[404]:


print(100*np.mean(train['Survived'][train['Age']<18]))
print(100*np.mean(train['Survived'][train['Age']>18]))


# In[405]:


print(100*np.mean(train['Survived'][train['SibSp']>0]))
print(100*np.mean(train['Survived'][train['SibSp']==0]))


# In[406]:


print(100*np.mean(train['Survived'][train['Parch']>0]))
print(100*np.mean(train['Survived'][train['Parch']==0]))


# In[407]:


train['Sex']=train['Sex'].apply(lambda x: 1 if x=='male' else 0)
test['Sex']=test['Sex'].apply(lambda x: 1 if x == 'male' else 0)


# In[408]:


train['Age']=train['Age'].fillna(np.mean(train['Age']))
train['Fare']=train['Fare'].fillna(np.mean(train['Fare']))
test['Age']=test['Age'].fillna(np.mean(test['Age']))
test['Fare']=test['Fare'].fillna(np.mean(test['Fare']))


# In[409]:


train = train[['PassengerId','Sex','Age','Fare','SibSp','Parch','Pclass','Survived']]
test = test[['PassengerId','Sex','Age','Fare','SibSp','Parch','Pclass']]


# In[410]:


X_train= train.drop('Survived', axis =1)

y_train= train['Survived']



# In[411]:


from sklearn.tree import DecisionTreeClassifier
classifier =DecisionTreeClassifier(max_depth = 3)
classifier.fit(X_train,y_train)


# In[412]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,classifier.predict(X_train)))



# In[413]:


pred = classifier.predict(test)


# In[415]:


submission11 = pd.DataFrame({'PassengerId': test['PassengerId'],'Survived': pred})
submission.to_csv('G:/ml/titanic disaster/')

