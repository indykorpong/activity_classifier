#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

from os import listdir, walk
from os.path import isfile, join
from sklearn.preprocessing import MinMaxScaler

from ..load_dataset import load_acc, load_hr, load_timer, merge_acc_and_hr

# # Load Raw Data

mypath = '../DDC_Data/raw/'
basepath = '../'
datapath = '../DDC_Data/'

# In[3]:


subj_range = np.hstack((np.arange(2001,2002),np.arange(3001,3006)))

all_patients = [str(i) for i in subj_range]


# In[4]:

def load_all_data(all_patients):


    df_all_p = pd.DataFrame()

    for subject_id in all_patients:
        print("Loading {0}'s data".format(subject_id))

        acc_filepath = mypath + subject_id + '/' + subject_id + '-log_acc.csv'
        df_raw = pd.read_csv(acc_filepath, header=None, names=['x','y','z','timestamp'])

        df_timer, rec_date, start_time, end_time = load_timer(subject_id)
        df_filt = load_acc(subject_id, rec_date, start_time, end_time)
        df_hr = load_hr(subject_id, rec_date, start_time, end_time)

        df1 = merge_acc_and_hr(df_filt, df_hr)

        cols = ['x','y','z']
        xyz_ = df1[cols].to_dict('split')['data']
        xyz_new = MinMaxScaler().fit_transform(xyz_)

        for i in range(len(cols)):
            df1[cols[i]] = pd.Series(xyz_new.transpose()[i])

        X_i_p = np.array(df1[cols].to_dict(orient='split')['data'])
        subj_i_p = np.array([subject_id for i in range(X_i_p.shape[0])])

        df_all_p = df_all_p.append(df1, sort=False)

    print('Finished Loading')

    return df_all_p


def export_cleaned_data(df_all_p, cleaned_data_path):
    df_all_p = df_all_p.reset_index(drop=True)

    label_list = ['sit', 'sleep', 'stand', 'walk']

    df_label = pd.DataFrame({
        'label': [i for i in range(len(label_list))],
        'activity name': label_list
    })

    df_all_p.to_csv(cleaned_data_path)

    csv_path = datapath + 'cleaned/label_names.csv'
    df_label.to_csv(csv_path)
