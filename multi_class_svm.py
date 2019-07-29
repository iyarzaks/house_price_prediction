#!/usr/bin/env python
# coding: utf-8

# In[56]:



import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# In[57]:


df = pd.read_csv('kc_house_data.csv')
df = df.sample(10000)
df.head()


# In[58]:


#removed_features = ['date', 'id']
# year built to age
#yr_renovated to bin
#zip code to categorial
#normalize all columns
#price to categorial


# In[59]:


new_df_columns = list(set(df.columns) - set(['date', 'id','long','lat']))
print (new_df_columns)


# In[60]:


update_df = df[new_df_columns]


# In[61]:


update_df.head()


# In[62]:


update_df['yr_built'] = update_df['yr_built'].apply(lambda x: 2015-x)
update_df['yr_built']


# In[63]:


def year_to_bin(year):
    if year != 0:
        return 1
    else:
        return 0

update_df['yr_renovated'] = update_df['yr_renovated'].apply(year_to_bin)


# In[64]:


update_df = pd.get_dummies(update_df,columns=['zipcode'])


# In[65]:


update_df.head()


# In[66]:


x= update_df.loc[:, update_df.columns != 'price']


# In[67]:


def price_to_class(price):
    if price<250000:
        return 0
    elif price<500000:
        return 1
    elif price<750000:
        return 2
    elif price<1000000:
        return 3
    else:
        return 4
    


# In[68]:


Y = update_df['price']
Y=Y.apply(price_to_class)


# In[69]:


Y


# In[70]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalize_x=scaler.fit_transform(x)


# In[71]:


X = pd.DataFrame(normalize_x)
X.head()


# In[72]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 0) 


# In[73]:


svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 
  
# model accuracy for X_test   
accuracy = svm_model_linear.score(X_test, y_test) 
  
# creating a confusion matrix 
cm = confusion_matrix(y_test, svm_predictions) 
print (accuracy)
print (cm)


# In[33]:


svm_model_linear.coef_[0]

