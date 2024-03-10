#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[9]:


data = pd.read_csv(r'C:\Users\Admin\Downloads\IRIS 1.csv')


# In[10]:


data.head()


# In[11]:


data.tail()


# In[12]:


data.describe()


# In[13]:


data.info()


# In[14]:


data.shape


# In[15]:


count =  data.species.value_counts()
print(count)


# In[16]:


data.isnull().sum()


# In[17]:


lab = data.species.unique().tolist()
lab


# In[18]:


plt.pie(count,labels=lab)
plt.title("Count of Species",fontsize=20)
plt.show()


# In[19]:


plt.subplots(figsize=(7,7))
sns.scatterplot(x="sepal_length",y="sepal_width",data=data,hue="species")
plt.show()


# In[20]:


plt.subplots(figsize=(7,7))
sns.scatterplot(x="petal_length",y="petal_width",data=data,hue="species")
plt.show()


# In[21]:


data.hist(edgecolor='black',figsize=(10,10))
plt.show()


# In[22]:


data1 = data.drop("ID",axis=1)
plot=sns.pairplot(data1,hue="species",diag_kind="hist")
plot.fig.suptitle("Relation of all feature with each other",y=1.1,fontsize=20)
plt.show()


# In[23]:


# Assuming 'species' column is non-numeric, so we drop it
data_numeric = data.drop(columns=['species'])

# Calculate correlation matrix
correlation_matrix = data_numeric.corr()


# In[24]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='species',y='petal_length',data=data)
plt.subplot(2,2,2)
sns.violinplot(x='species',y='petal_width',data=data)
plt.subplot(2,2,3)
sns.violinplot(x='species',y='sepal_length',data=data)
plt.subplot(2,2,4)
sns.violinplot(x='species',y='sepal_width',data=data)


# In[25]:


data = pd.read_csv(r'C:\Users\Admin\Downloads\IRIS 1.csv')
data_numeric = data.drop(columns=['species'])
sns.heatmap(data_numeric.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X = data.drop(["species","ID"],axis=1)
X


# In[28]:


Y = data["species"]
Y


# In[29]:


from sklearn.linear_model import LogisticRegression

# Create an instance of LogisticRegression
model = LogisticRegression(max_iter=1000)

# Now you can use the model for training and prediction


# In[33]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Assuming X and Y are your feature and target variables
# Split the dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# Create an instance of LogisticRegression
model = LogisticRegression(max_iter=1000)

# Fit the model on the training data
model.fit(x_train, y_train)

# Now you can use the model for prediction and evaluation


# In[35]:


from sklearn.metrics import accuracy_score

# Assuming you have already trained your model and have x_train, y_train available

# Calculate the train accuracy score
train_accuracy = model.score(x_train, y_train)

# Alternatively, you can use the accuracy_score function
# y_train_pred = model.predict(x_train)
# train_accuracy = accuracy_score(y_train, y_train_pred)

print("Train Accuracy:", train_accuracy)


# In[36]:


from sklearn.metrics import accuracy_score

# Assuming you have already trained your model and have x_test, y_test available

# Calculate the accuracy score
test_accuracy = model.score(x_test, y_test)

# Alternatively, you can use the accuracy_score function
# y_pred = model.predict(x_test)
# test_accuracy = accuracy_score(y_test, y_pred)

print("Test Accuracy:", test_accuracy)

