#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[41]:


# Load in dataset from Google Sheets

sheet_id = '1QXuYUiuLj2_3ViDJlsbYkAHWJxN41_OM_GpV0s7Rsc4'

df = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv")

print(df)


# In[43]:


# convert data to arrays for independent and dependent variable

x = df[["Time1"]]
y = df[["Brazil Log GDP per Capita"]]

print(x)


# In[49]:


plt.scatter(x,y) # make scatter plot of independent and dependent variables
plt.xlabel("Year") # x axis is Year
plt.ylabel("Brazil Log GDP per Capita") # y axis is Brazil GDP
plt.show()


# In[45]:


# create instance of LinearRegression object to create line of best fit

model = LinearRegression()
model.fit(x,y)


# In[46]:


# Graph scatter plot with line of best fit

plt.scatter(x,y)
plt.plot(x.values, model.predict(x), color = 'red') # plot line of best fit
plt.xlabel("Year")
plt.ylabel("Brazil Log GDP per Capita")
plt.show()


# In[48]:


# Calculate r squared value

r_squared = model.score(x,y)
print('R-Squared: ', r_squared)


# In[ ]:




