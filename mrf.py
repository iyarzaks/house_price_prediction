
import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import math
import statistics
import numpy


# In[51]:


df = pd.read_csv('kc_house_data.csv')
df  = df.sample(10000)
df.head()


# In[52]:


distance_dict1 = {}
for i in range(len(df)):
    for j in range (i+1,len(df)):
        distance = math.sqrt(math.pow(df.iloc[i]['lat']-df.iloc[j]['lat'],2)+math.pow(df.iloc[i]['long']-df.iloc[j]['long'],2))
        if distance < 0.012:
            distance_dict1[(i,j)]  = distance


# In[53]:


distance_dict1


# In[58]:


numpy.quantile(l,0.005)


# In[54]:


len(distance_dict1)

