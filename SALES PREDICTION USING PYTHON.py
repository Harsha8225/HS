#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv('advertising.csv')
df.head()


# In[3]:


df.shape


# In[5]:


df.describe()


# In[12]:


sns.pairplot(df, x_vars=['TV','Radio','Newspaper'],y_vars='Sales',kind='scatter')
plt.show()


# In[17]:


df['TV'].plot.hist(bins=10,color='gray')


# In[25]:


df['Radio'].plot.hist(bins=10, color='brown')


# In[27]:


df['Newspaper'].plot.hist(bins=10,color='blue')


# In[33]:


sns.heatmap(df.corr(),annot=True)
plt.show()


# In[34]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df[['TV']], df[['Sales']], test_size=0.3,random_state=0)


# In[35]:


print(x_train)


# In[36]:


print(y_train)


# In[37]:


print(x_test)


# In[38]:


print(y_test)


# In[42]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)


# In[43]:


res=model.predict(x_test)
print(res)


# In[44]:


model.coef_


# In[45]:


model.intercept_


# In[46]:


0.05473199* 69.2 + 7.14382225


# In[47]:


plt.plot(res)


# In[48]:


plt.scatter(x_test, y_test)
plt.plot(x_test, 7.14382225 + 0.05473199 * x_test, 'r')
plt.show()


# In[ ]:




