#!/usr/bin/env python
# coding: utf-8

# In[1]:


from stravalib.client import Client
TOKEN = "#####"
client = Client(access_token = TOKEN)
activities = client.get_activities(limit=20000)
activities


# In[2]:


init_cols = ['average_heartrate', 'max_heartrate', 'suffer_score','distance','moving_time', 'type','start_date_local', 'has_heartrate']
data = []
for activity in activities:
    temp = activity.to_dict()
    data.append([temp.get(x) for x in init_cols])
    
import pandas as pd
df = pd.DataFrame(data, columns=init_cols)
df.head()


# In[3]:


def get_minutes(time_str):
    h, m, s = time_str.split(':')
    return 60*int(h)+int(m)+int(s)/60.0

df = df[df['has_heartrate'].isin([True])]
df['distance_mile'] = df['distance'] * 0.000621371
df['time_min'] = df['moving_time'].apply(get_minutes)
df['average_pace'] = df['time_min']/df['distance_mile']
df['start_date_local'] = pd.to_datetime(df['start_date_local'])
df.set_index('start_date_local', inplace=True)
df.head()


# In[4]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-dark-palette')

del df['has_heartrate']
corr = df.corr()

plt.figure(figsize = (12, 8))
sns.heatmap(corr, annot=True, fmt=".2f")


# In[5]:


import numpy as np

def f(group):
    return (group-group.mean()).abs().div(group.std())

grouped = df.groupby('type')
outliers = grouped.transform(f) > 3
df = df[~outliers.any(axis=1)]
df = df[df['average_pace'] != np.inf]

trends = df.copy()
trends.set_index(pd.to_datetime(trends.index), drop=True, inplace = True)
trends['weekday'] = trends.index.map(lambda x: x.weekday)
trends.groupby('weekday').mean()


# In[6]:


import calendar

trends.groupby('weekday').mean()['distance_mile'].plot(kind='bar')
plt.style.use('seaborn-dark-palette')
plt.xticks(list(range(7)), list(calendar.day_name),rotation='horizontal')
plt.xlabel('')
plt.ylabel('Distance in miles')
plt.title('Average distance by day of week')


# In[7]:


trends.groupby('weekday').mean()['suffer_score'].plot(kind='bar')
plt.style.use('seaborn-dark-palette')
plt.xticks(list(range(7)), list(calendar.day_name),rotation='horizontal')
plt.xlabel('')
plt.ylabel('Suffer Score')
plt.title('Suffer score by day of week')


# In[8]:


trends.groupby('weekday').mean()['average_pace'].plot(kind='bar')
plt.style.use('seaborn-dark-palette')
plt.xticks(list(range(7)), list(calendar.day_name),rotation='horizontal')
plt.xlabel('')
plt.ylabel('Average Pace')
plt.title('Average pace by day of week')


# In[9]:


df.drop(['type'], axis = 1 , inplace = True)
del df['moving_time']
del df['distance']
df.head()
cols = ['suffer_score', 'average_pace', 'time_min', 'max_heartrate', 'average_heartrate', 'distance_mile']
sns.pairplot(x_vars=cols, y_vars = cols, data =df, size = 6)


# In[10]:


import sklearn
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(df)

from sklearn.cluster import KMeans
model = KMeans(n_clusters = 4)
model.fit(X)
df['Cluster'] = model.labels_
sns.pairplot(x_vars=cols, y_vars=cols, hue = 'Cluster', data =df)


# In[ ]:




