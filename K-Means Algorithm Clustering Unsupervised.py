#!/usr/bin/env python
# coding: utf-8

# In[8]:


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


df = pd.read_csv(r"D:\ai\dataset for practice\unsupervised learing\Mall_Customers.csv")
df.head(2)


# In[17]:




plt.scatter(df.Age,df['Annual Income (k$)'])
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')


# In[19]:


km = KMeans(n_clusters=4)
y_predicted = km.fit_predict(df[['Age','Annual Income (k$)']])
y_predicted


# In[20]:



df['cluster']=y_predicted
df.head()


# In[21]:


km.cluster_centers_


# In[22]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
df4 = df[df.cluster==3]
plt.scatter(df1.Age,df1['Annual Income (k$)'],color='green')
plt.scatter(df2.Age,df2['Annual Income (k$)'],color='red')
plt.scatter(df3.Age,df3['Annual Income (k$)'],color='black')
plt.scatter(df4.Age,df4['Annual Income (k$)'],color='yellow')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.legend()


# In[23]:



scaler = MinMaxScaler()

scaler.fit(df[['Annual Income (k$)']])
df['Annual Income (k$)'] = scaler.transform(df[['Annual Income (k$)']])

scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])


# In[25]:


df.head(2)


# In[26]:


plt.scatter(df.Age,df['Annual Income (k$)'])


# In[28]:


km = KMeans(n_clusters=4)
y_predicted = km.fit_predict(df[['Age','Annual Income (k$)']])
y_predicted


# In[29]:


df['cluster']=y_predicted
df.head(2)


# In[30]:


km.cluster_centers_


# In[32]:


df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
df4 = df[df.cluster==3]
plt.scatter(df1.Age,df1['Annual Income (k$)'],color='green')
plt.scatter(df2.Age,df2['Annual Income (k$)'],color='red')
plt.scatter(df3.Age,df3['Annual Income (k$)'],color='black')
plt.scatter(df4.Age,df4['Annual Income (k$)'],color='yellow')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()


# In[33]:


sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age','Annual Income (k$)']])
    sse.append(km.inertia_)


# In[34]:


plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)


# In[35]:


df = pd.read_csv(r"D:\ai\dataset for practice\unsupervised learing\Mall_Customers.csv")
df.head(2)


# In[36]:


plt.scatter(df.Age,df['Spending Score (1-100)'])
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')


# In[ ]:




