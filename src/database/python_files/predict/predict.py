#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import pickle
import sys

# path_to_module = '/Users/Indy/Desktop/coding/Dementia_proj/src/database/python_files/'
path_to_module = '/var/www/html/python/mysql_connect/python_files'
sys.path.append(path_to_module)

from os import listdir, walk
from os.path import isfile, join

from predict.preprocessing import prepare_impure_label
from predict.classifier_alg import combine_2
# In[1]:


from tqdm import tqdm


# # Load Cleaned Data

basepath = '/var/www/html/python/mysql_connect/'
datapath = basepath + 'DDC_Data/'
mypath = basepath + 'DDC_Data/raw/'


def predict_label(df_all_p, cleaned_data_path=''):

    filename = basepath + 'model/knn_model_patients.pkl'

    model = pickle.load(open(filename,'rb'))

    subj_range = np.hstack((np.arange(2001,2002),np.arange(3001,3006)))
    all_patients = [str(i) for i in subj_range]


    # # Load Data


    # df_all_p_sorted = pd.read_csv(cleaned_data_path, index_col=0)
    df_all_p_sorted = df_all_p

    # # Predict Labels


    cols = ['x','y','z']

    X_all = np.array(df_all_p_sorted[cols].to_dict('split')['data'])
    y_all = np.zeros(X_all.shape[0])

    y_pred_all = []
    grouped = df_all_p_sorted.groupby('ID')

    T = 0.16
    freq = 1/T
    n_onehour = int(60*60*freq)

    for x in grouped:
        label_grp = x[0]

        df_grp = grouped.get_group(label_grp)
        print(df_grp.shape[0])

        X_all_p = np.array(df_grp[cols].to_dict('split')['data'])
        y_all_p = np.zeros(X_all_p.shape[0])

        X_impure, y_impure = prepare_impure_label(X_all_p, y_all_p)
        y_pred = model.predict(X_impure)

        window_length = 60
        y_pred_fill = np.hstack(([y_pred[0] for i in range(window_length-1)], y_pred))

        y_pred_all.append(y_pred_fill)

    y_pred_all = np.hstack(y_pred_all)

    y_pred_walk = np.array(combine_2(X_all, y_pred_all))

    df_all_p_sorted['y_pred'] = pd.Series(y_pred_walk)

    return df_all_p_sorted

# In[74]:
def export_predicted_data(df_all_p_sorted, predicted_data_path):
    df_all_p_sorted.to_csv(predicted_data_path)


# In[ ]:


#     for i in range(n_onehour, df_grp.shape[0], n_onehour):
#         df_onehour = df_grp[i-n_onehour:i]
#         print(i-n_onehour, i)

#         cols = ['x','y','z']

#         X_all_p = np.array(df_onehour[cols].to_dict('split')['data'])
#         y_all_p = np.zeros(X_all_p.shape[0])

#         X_impure, y_impure = prepare_impure_label(X_all_p, y_all_p)
#         y_pred = model.predict(X_impure)

#         window_length = 60
#         y_pred_fill = np.hstack(([y_pred[0] for i in range(window_length-1)], y_pred))

#         y_pred_all.append(y_pred_fill)

#     if(len(y_pred_all)>0):
#         y_pred_temp = np.hstack(y_pred_all)
#     else:
#         y_pred_temp = np.array(y_pred_all)
#     print(y_pred_temp.shape, len(y_pred_all), df_grp.shape)

#     if(y_pred_temp.shape[0]<df_grp.shape[0]):
#         df_onehour = df_grp[y_pred_temp.shape[0]:]
