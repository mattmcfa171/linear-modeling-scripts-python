#!/usr/bin/env python
# coding: utf-8

# # Homework Assignment: Linear Modeling Script in Python

# ## Load Dependencies

# In[1]:


import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys


# ## Load Data

# In[2]:


dataset = pd.read_csv(sys.argv[1])
print(dataset)


# ## Plot Data

# In[3]:


plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.title('Y vs X')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('py_orig.png')
plt.clf()


# ## Plot Linear Regression

# In[4]:


model = LinearRegression()
model.fit(dataset[['x']], dataset[['y']])


# In[5]:


plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.plot(dataset[['x']], model.predict(dataset[['x']]), color = 'blue')
plt.title('Y vs X')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('py_lm.png')
plt.clf()
