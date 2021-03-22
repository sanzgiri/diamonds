#!/usr/bin/env python
# coding: utf-8

# ### Read dataset into Pandas DataFrame

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.max_columns = 999
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

df = pd.read_csv("diamonds.csv")


# ### Examine Data
# - For first pass, keep only the 4Cs (Carat, Cut, Clarity, Color)
# - Include a fifth "C": Certification
# - Explore feature distribution

# In[2]:


df.head()


# In[3]:


df.describe()


# In[4]:


# Remove unused features: table, depth, measurements, wlink
df = df.drop(['measurements', 'wlink'], axis=1)
# Get counts for each value of a categorical feature in dataset
print df['cut'].value_counts()
print df['color'].value_counts()
print df['clarity'].value_counts()
print df['cert'].value_counts()


# ### Pre-process the data
# - Price: Remove "," "$", convert to float, take log
# - Convert Categorical features Cut, Clarity, Color, Cert to one-hot encoded features.
# - Drop Cut, Clarity, Color, Cert after one-hot encoding

# In[5]:


def xform_price(df):
    df['price'] = df['price'].apply(lambda x:x.replace(',',''))
    df['price'] = df['price'].apply(lambda x:x[1:len(x)])
    df['price'] = df['price'].astype(float)
    return df
    
def xform_4c(df):
    for X in ['cut','clarity','color','cert']:
        X_dummies = pd.get_dummies(df[X], prefix = X)
        X_columns = X_dummies.columns.values[:-1]
        df = df.join(X_dummies[X_columns])
    return df


# In[6]:


df.shape


# In[7]:


df = xform_price(df)
df = xform_4c(df)


# In[8]:


df.head()


# In[9]:


df.shape


# In[10]:


df.describe()


# In[11]:


df = df.drop(['cut','clarity','color','cert'], axis=1)


# # Fit a Linear Regression for "Price" vs. all other features
# - Split dataset to train and test sets (80-20)
# - Train model using "train" dataset
# - Score model on the "test" dataset
# - Features are normalized by sci-kit learn's LinearRegression function
# - Score is R-squared metric for linear regression
# - MSE is mean-square-error between prediction and actual price for test dataset
# - Take a look at regression coefficients. Do they look reasonable?

# In[12]:


from sklearn.model_selection import train_test_split
y = df['price']
X = df.drop(['price'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


from sklearn import linear_model
regr = linear_model.LinearRegression(normalize=True)
regr.fit(X_train, y_train)
score = regr.score(X_test, y_test)
mse = np.mean((regr.predict(X_test) - y_test) ** 2)
print score, mse


# In[14]:


feats = X.columns
feat_dict = dict(zip(feats,regr.coef_))
feat_dict


# ### Can we do better? 
# - Since price varies from \$200 to \$10000, let's try using log(price)

# In[15]:


df['price'] = np.log(df['price'])
y = df['price']
X = df.drop(['price'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regr = linear_model.LinearRegression(normalize=True)
regr.fit(X_train, y_train)
score = regr.score(X_test, y_test)
mse = np.mean((regr.predict(X_test) - y_test) ** 2)
print score, mse


# In[16]:


#feats = X.columns
#feat_dict = dict(zip(feats,regr.coef_))
#feat_dict


# ### Can we do better? Add carat3
# - Define new variable "carat_3" as a measure of the diameter (size) of the diamond
# - diamond density is 3.5 g/cm3
# - 1 carat = 0.2 g
# - Mass (g) = 0.2 * carat
# - Volume = Mass / Density = 0.2 * carat / 3.5 in cm3
# - Size ~ Volume^1/3 ~ carat~1/3

# In[17]:


df['carat_3'] = df['carat'].apply(lambda x:np.power(x,float(1./3)))
y = df['price']
X = df.drop(['price'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regr = linear_model.LinearRegression(normalize=True)
regr.fit(X_train, y_train)
score = regr.score(X_test, y_test)
mse = np.mean((regr.predict(X_test) - y_test) ** 2)
print score, mse


# ### Plot Predicted Price (Blue) vs Actual Price (Red) for a subset of the test dataset 
# - (Ideal Cut, Color J, Clarity SI2)

# In[18]:


df_plot = df[(df.cut_Ideal==1) & (df.color_J==1) & (df.clarity_SI2==1)]
df_train, df_test = train_test_split(df_plot, test_size=0.2, random_state=42)
y_plot = df_test['price']
X_plot = df_test.drop(['price'], axis=1)

plt.rcParams['figure.figsize'] = (10,6)
plt.scatter(X_plot['carat'], np.exp(y_plot), s=5, color='red', label = 'Actual Price')
plt.scatter(X_plot['carat'], np.exp(regr.predict(X_plot)), s=5, color='blue', label = 'Predicted Price')
plt.title('Predicted & Actual Price by Carat size')
plt.xlabel('Carat size')
plt.ylabel('Price ($)')
plt.semilogy()
plt.legend(loc='best')
plt.show()


# In[19]:


df.head()


# In[ ]:




