#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv(r"D:\ai\dataset for practice\unsupervised learing\Mall_Customers.csv")
df.head(2)


# In[3]:


plt.scatter(df.Age,df['Annual Income (k$)'])
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')


# In[4]:


km = AgglomerativeClustering(n_clusters=4)
y_predicted = km.fit_predict(df[['Age','Annual Income (k$)']])
y_predicted


# In[5]:


df['cluster']=y_predicted
df.head()


# In[8]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
df4 = df[df.cluster==3]
plt.scatter(df1.Age,df1['Annual Income (k$)'],color='green')
plt.scatter(df2.Age,df2['Annual Income (k$)'],color='red')
plt.scatter(df3.Age,df3['Annual Income (k$)'],color='black')
plt.scatter(df4.Age,df4['Annual Income (k$)'],color='yellow')

plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.legend()


# In[11]:


km.labels_


# In[ ]:




