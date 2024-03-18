#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix


# In[4]:


credit_card_data = pd.read_csv("cread.csv")


# In[5]:


credit_card_data.head()


# In[6]:


credit_card_data.shape


# In[7]:


credit_card_data.isnull().sum()


# In[8]:


credit_card_data.describe()


# In[9]:


credit_card_data.info()


# In[11]:


credit_card_data.hist(figsize=(20, 15))
plt.show()


# In[12]:


credit_card_data['Class'].value_counts()


# In[14]:


plt.figure(figsize=(6, 4))
sns.countplot(data=credit_card_data, x='Class')
plt.title('Distribution of Legit Transactions & Fraudulent Transactions')
plt.show()


# In[16]:


sns.distplot(credit_card_data['Time'])
plt.show()


# In[17]:


sns.distplot(credit_card_data['Amount'])
plt.show()


# In[18]:


normal = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[19]:


print(normal.shape)
print(fraud.shape)


# In[20]:


normal.describe()


# In[21]:


fraud.describe()


# In[22]:


credit_card_data.groupby('Class').mean()


# In[23]:


correlation = credit_card_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, cmap="Blues", annot=True)
plt.title('Correlation Matrix Heatmap ')
plt.show()


# In[25]:


for col in credit_card_data.select_dtypes(include = np.number):
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    sns.boxplot(x=col, data =credit_card_data)
    plt.subplot(1,2,2)
    sns.boxplot(x='Class',y=col, data =credit_card_data)
    plt.show()


# In[26]:


X = credit_card_data.drop(columns = 'Class' , axis=1)
Y = credit_card_data['Class']
print(X)
print(Y)


# In[27]:


X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2)


# In[28]:


print(X.shape, X_train.shape , X_test.shape)


# In[29]:


model = LogisticRegression()


# In[30]:


model.fit(X_train, Y_train)


# In[31]:


training_cradit_card_data = model.predict(X_train)
training_cradit_card_prediction = accuracy_score(training_cradit_card_data, 
Y_train)


# In[32]:


print("Accuracy of Training Data : ", training_cradit_card_prediction)


# In[33]:


testing_cradit_card_data = model.predict(X_test)
testing_cradit_card_prediction = accuracy_score(testing_cradit_card_data, 
Y_test)


# In[36]:


input_data = (172782, 0.2195290548808, 0.881245743149381, -0.635890849626074, 0.960927997354267, -0.15297078241792, -1.01430717553581, 0.42712562123977, 0.121340360686438, -0.285669713965033, -0.111639563712404, -1.10923217257381, -0.453234511207357, -1.04694569480814, 1.12267439177599, 1.24351816319406, -1.43189726654146, 0.939328414637529, -0.00237255021158738, 2.89495180691714, 0.0066660264624123, 0.0999358684850581, 0.337119908748991, 0.251790775254804, 0.0576877870152752, -1.5083682536529, 0.144023372480988, 0.181205105338699, 0.21524280501856, 24.05)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
print(prediction)


# In[37]:


if (prediction[0]=='1'):
 print("Credit Card is fraud.")
else:
 print("Credit Card is normal.")


# In[38]:


y_pred = model.predict(X_test)
confusion_matrix = confusion_matrix(Y_test, y_pred.round())
sns.heatmap(confusion_matrix, annot=True, fmt="d", cbar = False)
plt.title("Confusion Matrix")
plt.show()


# In[ ]:




