#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import os


# In[3]:


df = pd.read_csv("breastdata.csv")
df.head(3)


# In[4]:


df.drop(columns=['id','Unnamed: 32'],inplace=True)


# In[5]:


df.info()


# In[6]:


df['diagnosis'].value_counts()


# In[7]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df.iloc[:,1:], df.iloc[:,0],test_size=0.2, random_state=1)


# In[8]:


x_train.shape,x_test.shape


# In[9]:


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)


# In[10]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)


# In[11]:


knn.fit(x_train,y_train)


# In[12]:


y_pred = knn.predict(x_test)


# In[13]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[14]:


from sklearn.model_selection import cross_val_score

knn = KNeighborsClassifier(n_neighbors=5)

np.mean(cross_val_score(knn, df.iloc[:,1:],df.iloc[:,0],scoring='accuracy',cv=5))


# In[15]:


scores = []

for i in range(1,16):
    knn = KNeighborsClassifier(n_neighbors=i)
     
    knn.fit(x_train,y_train)
    
    y_pred = knn.predict(x_test)

    scores.append(accuracy_score(y_test,y_pred))


# In[16]:


import matplotlib.pyplot as plt

plt.plot(range(1,16),scores)
plt.xlabel("k neighbors")
plt.ylabel("accuracy score")


# In[18]:


sns.heatmap(df.isna())


# In[20]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr() , annot=True , cmap= "mako")


# In[ ]:





# In[ ]:




