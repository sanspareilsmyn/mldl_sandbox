#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# https://www.kaggle.com/werooring/basic-eda-lgbm-modeling-public-score-0-87714


# In[2]:


import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# from plotnine import *

from category_encoders.ordinal import OrdinalEncoder
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier

import random
import gc
import os

# In[4]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# In[7]:


train.shape, test.shape

# In[9]:


train.describe(include='O')


# # Numerical Data

# ## Univariate Data Visualization

# In[12]:


def get_min_max_avg(df, feature):
    print('Feature: ', feature)
    print('--------------------------------------')
    print('The max value is:', df[feature].max())
    print('The min value is:', df[feature].min())
    print('The average value is:', df[feature].mean())
    print('The median value is:', df[feature].median())


# In[13]:


def plot_hist(df, feature, max_ylim, bins=10):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(df[feature], bins=bins)
    ax.set_ylim(0, max_ylim)
    ax.set_title(feature + ' distribution (bins=' + str(bins) + ')')


# In[14]:


train.info()

# ### 1) Age

# In[15]:


get_min_max_avg(train, 'age')

# In[16]:


plot_hist(train, 'age', 4000, 15)

# ### 2) Final Weight

# In[17]:


get_min_max_avg(train, 'fnlwgt')

# In[18]:


plot_hist(train, 'fnlwgt', 13000, 20)

# ### 3) Education num

# In[19]:


get_min_max_avg(train, 'education_num')

# In[20]:


plot_hist(train, 'education_num', 9000, 16)

# ### 4) Capital Gain

# In[21]:


get_min_max_avg(train, 'capital_gain')

# In[25]:


plot_hist(train, 'capital_gain', 25000, 20)

# ### 5) Capital Loss

# In[27]:


get_min_max_avg(train, 'capital_loss')

# In[30]:


plot_hist(train, 'capital_loss', 25000, 20);

# ### 6) Hours per week

# In[31]:


get_min_max_avg(train, 'hours_per_week')

# In[34]:


plot_hist(train, 'hours_per_week', 15000, 20)

# ## Multivariate Data Visualization

# ### 1) Age vs income

# In[51]:


sns.set_theme(style="whitegrid")

sns.violinplot(data=train, x="age", y="income", split=True, inner='quart',
               linewidth=1,
               palette={">50K": "b", "<=50K": ".85"})
sns.despine(left=True)

# In[62]:


sns.boxplot(x='income', y='age', data=train)

# ### 2) fnlwgt vs income

# In[63]:


sns.boxplot(x='income', y='fnlwgt', data=train)

# ### 3) education_num vs income

# In[64]:


sns.boxplot(x='income', y='education_num', data=train)

# ### 4) capital_gain vs income

# In[65]:


sns.boxplot(x='income', y='capital_gain', data=train)

# ### 5) capital_loss vs income

# In[66]:


sns.boxplot(x='income', y='capital_loss', data=train)

# ### 6) capital_net vs income

# In[72]:


train['capital_net'] = train['capital_gain'] - train['capital_loss']

# In[73]:


train['capital_net'].value_counts()

# In[74]:


sns.boxplot(x='income', y='capital_net', data=train)

# ### 7) Hours per week vs income

# In[75]:


sns.boxplot(x='income', y='hours_per_week', data=train)

# # Categorical Data

# In[76]:


for col in train.columns:
    if train[col].dtype == 'object':
        all_categories = train[col].unique()
        print(f'Column "{col}" has {len(all_categories)} unique categroies')
        print('The categories are:', ', '.join(all_categories))
        print()

    # In[77]:

for col in train.columns:
    if train[col].dtype == 'object':
        categories = train[col].unique()
        print(f'The number of unique values in [{col}]: {len(categories)}')


# In[78]:


def get_unique_values(df, feature):
    all_categories = train[feature].unique()
    print(f'Column "{feature}" has {len(all_categories)} unique categroies')
    print('------------------------------------------')
    print('\n'.join(all_categories))


# ### 1) Workclass

# In[79]:


get_unique_values(train, 'workclass')

# In[90]:


fig, ax = plt.subplots(1, 1, figsize=(15, 10))
col = 'workclass'
value_counts = train[col].value_counts()
sns.countplot(x=col, data=train, palette="Set2", edgecolor='black', order=value_counts.index)

for i, v in value_counts.reset_index().iterrows():
    ax.text(i - 0.1, v[col] + 150, v[col])

# In[93]:


fig, ax = plt.subplots(1, 1, figsize=(15, 10))
value_counts = train[col].value_counts()
sns.countplot(x=col, hue='income', data=train, palette="Set2", edgecolor='black', order=value_counts.index);

# ### 2) Education

# In[94]:


get_unique_values(train, 'education')

# In[96]:


fig, ax = plt.subplots(1, 1, figsize=(20, 10))
col = 'education'
value_counts = train[col].value_counts()
sns.countplot(x=col, data=train, palette="Set2", edgecolor='black', order=value_counts.index)

for i, v in value_counts.reset_index().iterrows():
    ax.text(i - 0.1, v[col] + 150, v[col])

# In[97]:


fig, ax = plt.subplots(1, 1, figsize=(15, 7))
value_counts = train[col].value_counts()
sns.countplot(x=col, hue='income', data=train, palette="Set2", edgecolor='black', order=value_counts.index);

# ### 3) Marital status

# In[98]:


get_unique_values(train, 'marital_status')

# In[99]:


fig, ax = plt.subplots(1, 1, figsize=(20, 10))
col = 'marital_status'
value_counts = train[col].value_counts()
sns.countplot(x=col, data=train, palette="Set2", edgecolor='black', order=value_counts.index)

for i, v in value_counts.reset_index().iterrows():
    ax.text(i - 0.1, v[col] + 150, v[col])

# In[100]:


fig, ax = plt.subplots(1, 1, figsize=(15, 7))
value_counts = train[col].value_counts()
sns.countplot(x=col, hue='income', data=train, palette="Set2", edgecolor='black', order=value_counts.index);

# In[102]:


train.loc[train[col] == 'Married-AF-spouse', 'income'].value_counts()

# ### 4) Occupation

# In[104]:


get_unique_values(train, 'occupation')

# In[105]:


fig, ax = plt.subplots(1, 1, figsize=(20, 10))
col = 'occupation'
value_counts = train[col].value_counts()
sns.countplot(x=col, data=train, palette="Set2", edgecolor='black', order=value_counts.index)

for i, v in value_counts.reset_index().iterrows():
    ax.text(i - 0.1, v[col] + 150, v[col])

# In[107]:


fig, ax = plt.subplots(1, 1, figsize=(15, 7))
value_counts = train[col].value_counts()
sns.countplot(y=col, hue='income', data=train, palette="Set2", edgecolor='black', order=value_counts.index);

# ### 5) Relationship

# In[109]:


get_unique_values(train, 'relationship')

# In[110]:


fig, ax = plt.subplots(1, 1, figsize=(20, 10))
col = 'relationship'
value_counts = train[col].value_counts()
sns.countplot(x=col, data=train, palette="Set2", edgecolor='black', order=value_counts.index)

for i, v in value_counts.reset_index().iterrows():
    ax.text(i - 0.1, v[col] + 150, v[col])

# In[111]:


fig, ax = plt.subplots(1, 1, figsize=(15, 7))
value_counts = train[col].value_counts()
sns.countplot(x=col, hue='income', data=train, palette="Set2", edgecolor='black', order=value_counts.index);

# ### 6) Race

# In[112]:


get_unique_values(train, 'race')

# In[113]:


fig, ax = plt.subplots(1, 1, figsize=(20, 10))
col = 'race'
value_counts = train[col].value_counts()
sns.countplot(x=col, data=train, palette="Set2", edgecolor='black', order=value_counts.index)

for i, v in value_counts.reset_index().iterrows():
    ax.text(i - 0.1, v[col] + 150, v[col])

# In[114]:


fig, ax = plt.subplots(1, 1, figsize=(15, 7))
value_counts = train[col].value_counts()
sns.countplot(x=col, hue='income', data=train, palette="Set2", edgecolor='black', order=value_counts.index);

# ### 7) Sex

# In[115]:


get_unique_values(train, 'sex')

# In[116]:


fig, ax = plt.subplots(1, 1, figsize=(20, 10))
col = 'sex'
value_counts = train[col].value_counts()
sns.countplot(x=col, data=train, palette="Set2", edgecolor='black', order=value_counts.index)

for i, v in value_counts.reset_index().iterrows():
    ax.text(i - 0.1, v[col] + 150, v[col])

# In[117]:


fig, ax = plt.subplots(1, 1, figsize=(15, 7))
value_counts = train[col].value_counts()
sns.countplot(x=col, hue='income', data=train, palette="Set2", edgecolor='black', order=value_counts.index);

# ### 8) Native Country

# In[119]:


get_unique_values(train, 'native_country')

# In[120]:


fig, ax = plt.subplots(1, 1, figsize=(20, 10))
col = 'native_country'
value_counts = train[col].value_counts()
sns.countplot(x=col, data=train, palette="Set2", edgecolor='black', order=value_counts.index)

for i, v in value_counts.reset_index().iterrows():
    ax.text(i - 0.1, v[col] + 150, v[col])

# In[121]:


fig, ax = plt.subplots(1, 1, figsize=(15, 7))
value_counts = train[col].value_counts()
sns.countplot(x=col, hue='income', data=train, palette="Set2", edgecolor='black', order=value_counts.index);

# In[128]:


fig, ax = plt.subplots(1, 1, figsize=(15, 7))
train_us = train[train['native_country'] == 'United-States']
col = 'native_country'
sns.countplot(x=col, hue='income', data=train_us, palette="Set2", edgecolor='black');

# In[130]:


fig, ax = plt.subplots(1, 1, figsize=(15, 7))
train_other = train[train['native_country'] != 'United-States']
col = 'native_country'
sns.countplot(y=col, hue='income', data=train_other, palette="Set2", edgecolor='black');

# ## Modeling

# In[144]:


# train.drop(['id'], axis=1, inplace=True)
train.drop(['capital_net'], axis=1, inplace=True)
# test.drop(['id'], axis=1, inplace=True)


# In[145]:


y = train['income'] != '<=50K'
X = train.drop(['income'], axis=1)

# In[146]:


print(X.columns)
print(test.columns)

# In[147]:


LE_encoder = OrdinalEncoder(list(X.columns))

X = LE_encoder.fit_transform(X, y)
test = LE_encoder.transform(test)

# In[148]:


NFOLDS = 5
folds = KFold(n_splits=NFOLDS)

# In[149]:


columns = X.columns
splits = folds.split(X, y)
y_preds = np.zeros(test.shape[0])

# In[150]:


feature_importances = pd.DataFrame()
feature_importances['feature'] = columns

# In[152]:


model = LGBMClassifier(objective='binary', verbose=5, random_state=91)

# In[153]:


for fold_n, (train_index, valid_index) in enumerate(splits):
    print('Fold: ', fold_n + 1)
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    evals = [(X_train, y_train), (X_valid, y_valid)]
    model.fit(X_train, y_train, eval_metric='f1', eval_set=evals, verbose=True)

    feature_importances[f'fold_{fold_n + 1}'] = model.feature_importances_

    y_preds += model.predict(test).astype(int) / NFOLDS

    del X_train, X_valid, y_train, y_valid
    gc.collect()

# In[154]:


feature_importances

# In[155]:


y_preds

# In[ ]:




