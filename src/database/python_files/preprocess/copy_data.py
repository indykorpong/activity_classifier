#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import time
import datetime
import sys

path_to_module = 'C:/Users/Indy/Desktop/python_files/'
sys.path.append(path_to_module)

from datetime import timedelta, date
from os import listdir, walk
from os.path import isfile, join
from sklearn.preprocessing import MinMaxScaler

from load_data.load_dataset import load_acc, load_hr, load_timer, merge_acc_and_hr, calc_sec, calc_ts

# # Load Raw Data

datapath = 'DDC_Data/'
basepath = ''

# In[3]:


subj_range = np.hstack((np.arange(2001,2002),np.arange(3001,3006)))

all_patients = [str(i) for i in subj_range]


# In[4]:
def load_raw_data():

    mypath = 'DDC_Data/raw/'
    basepath = ''

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

    df_all_p = df_all_p.reset_index(drop=True)

    return df_all_p

# # Copy Data

def copy_one_day(df_all_p):
    df_day = pd.DataFrame()

    T = 0.16
    freq = 1/T
    oneday = 24*60*60

    while(df_day.shape[0]<=int(oneday*freq)):
        df_day = df_day.append(df_all_p, sort=False)

    df_day = df_day[:int(oneday*freq)]
    df_day = df_day.reset_index(drop=True)

    df_day['ID'] = pd.Series(['9999' for i in range(df_day.shape[0])])

    date_format = '%Y-%m-%d'

    dt = '2019-03-28'
    date_t = datetime.datetime.fromtimestamp(time.mktime(time.strptime(dt, date_format)))
    midnight = '00:00:00.000'

    time_list = np.array([(date_t + timedelta(seconds=calc_sec(midnight)+(T*i))).strftime(date_format) + ' ' + 
                calc_ts(calc_sec(midnight)+(T*i)) for i in range(int(oneday*freq))])

    df_day['timestamp'] = pd.Series(time_list)

    return df_day

# In[45]:

def copy_one_month(df_all_p):
    df_day = pd.DataFrame()

    T = 0.16
    freq = 1/T
    oneday = 24*60*60

    while(df_day.shape[0]<=int(30*oneday*freq)):
        df_day = df_day.append(df_all_p, sort=False)

    df_day = df_day[:int(oneday*freq)]
    df_day = df_day.reset_index(drop=True)

    df_day['ID'] = pd.Series(['9999' for i in range(df_day.shape[0])])

    date_format = '%Y-%m-%d'

    dt = '2019-03-28'
    date_t = datetime.datetime.fromtimestamp(time.mktime(time.strptime(dt, date_format)))
    midnight = '00:00:00.000'

    time_list = np.array([(date_t + timedelta(seconds=calc_sec(midnight)+(T*i))).strftime(date_format) + ' ' + 
                calc_ts(calc_sec(midnight)+(T*i)) for i in range(int(30*oneday*freq))])

    df_day['timestamp'] = pd.Series(time_list)
    print(time_list.shape)

    return df_day


    # # Store Cleaned Data in CSV

def export_copied_data(df_day, cleaned_data_path):

    df_day.to_csv(cleaned_data_path)


cleaned_data_path_day = datapath + 'cleaned/cleaned_data_9999_day.csv'
cleaned_data_path_month = datapath + 'cleaned/cleaned_data_9999_month.csv'

df_all_p = load_raw_data()
df_day = copy_one_day(df_all_p)
export_copied_data(df_day, cleaned_data_path_day)

df_month = copy_one_month(df_all_p)
export_copied_data(df_month, cleaned_data_path_month)