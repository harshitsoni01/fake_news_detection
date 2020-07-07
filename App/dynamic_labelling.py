#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import joblib


# In[34]:
print("Enter name of csv file to label")
print("NOTE: put file in csv_files directory")
file_name = input("")
file_location = f'../csv_files/{file_name}'
df_read = pd.read_csv(file_location)


# In[35]:


# In[36]:

df_pred = pd.DataFrame(columns=['title', 'label'])


# In[47]:


df1 = pd.DataFrame(columns=['title', 'label'])


# In[48]:


df1.append(['title'], "hello")


# In[55]:


pipeline = joblib.load('pipeline_final.sav')


# In[63]:

print("Labelling...")
for index, row in df_read.iterrows():
    text = row['title']
    text2 = [text]
    pred = pipeline.predict(text2)
    df1 = df1.append({'title': text, 'label': pred}, ignore_index=True)
    # print(f'{text}: {pred}')


# In[64]:


df1


# In[66]:

file_save_location = f'../csv_files/{file_name}_labelled.csv'
df1.to_csv('../csv_files/example1_labelled.csv')


# In[ ]:
