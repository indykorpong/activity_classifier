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
import csv

from io import StringIO

on_server = int(sys.argv[1])

at_home = 'C:'

if(on_server==0):
    path_to_module = at_home + '/Users/Indy/Desktop/coding/Dementia_proj/src/database/python_files/'

elif(on_server==1):
    path_to_module = '/var/www/html/python/mysql_connect/python_files'

sys.path.append(path_to_module)
os.chdir(path_to_module)

# # Set data path

if(on_server==0):
    basepath = at_home + '/Users/Indy/Desktop/coding/Dementia_proj/'
else:
    basepath = '/var/www/html/python/mysql_connect/'
    
datapath = basepath + 'DDC_Data/'
mypath = basepath + 'DDC_Data/raw/'

from datetime import timedelta, date
from os import listdir, walk
from os.path import isfile, join
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from load_data.load_dataset import load_all_data, load_acc, load_hr, load_timer, merge_acc_and_hr, calc_sec, calc_ts
from insert_db.insert_db import get_patients_acc_hr

# In[3]:


subj_range = np.hstack((np.arange(2001,2002),np.arange(3001,3006)))

all_patients = [str(i) for i in subj_range]


# In[4]:
def load_raw_data(all_patients, date_to_retrieve, mydb, mycursor):

    df_acc, df_hr = get_patients_acc_hr(all_patients, date_to_retrieve, mydb, mycursor)

    df1 = merge_acc_and_hr(df_acc, df_hr)

    cols = ['x','y','z']
    xyz_ = df1[cols].to_dict('split')['data']
    xyz_new = MinMaxScaler().fit_transform(xyz_)

    for i in range(len(cols)):
        df1[cols[i]] = pd.Series(xyz_new.transpose()[i])

    X_i_p = np.array(df1[cols].to_dict(orient='split')['data'])
    subj_i_p = np.array([subject_id for i in range(X_i_p.shape[0])])

    print('Finished Loading')

    df_1 = df_1.reset_index(drop=True)

    return df_1

def load_raw_data_2(all_patients):
    df_all_p, X_all_p, y_all_p, subj_all_p, ts_all_p, hr_all_p = load_all_data(all_patients)
    df_all_p = df_all_p.reset_index(drop=True)

    return df_all_p

# # Copy Data

# In[45]:

T = 0.16
freq = 1/T
oneday = 24*60*60
n_days = 1
copy_length = int(n_days*oneday*freq)
# copy_length = 30000

def copy_dataframe(df_all_p):
    output = StringIO()
    writer = csv.writer(output, delimiter=',')

    list_df = df_all_p.to_dict(orient='split')['data']

    
    df_length = int(len(list_df))

    # write first row or index row

    cols = list(df_all_p.columns.values)
    writer.writerow(cols)
    print(copy_length, df_length)

    for c in cols:
        if(df_all_p[c].dtypes!=str):
            print('changed')
            df_all_p[c] = df_all_p[c].astype(str)

    # start copying

    pbar = tqdm(total=copy_length)
    count = 0

    while(True):
        for i in range(len(list_df)):
            writer.writerow(list_df[i])

        count += df_length
        pbar.update(df_length)

        if(count>=copy_length):
            break

    pbar.close()
    
    output.seek(0) # we need to get back to the start of the BytesIO

    df_large = pd.read_csv(output)

    df_large = df_large[:int(n_days*oneday*freq)]
    df_large = df_large.reset_index(drop=True)

    print(df_large.head(5))

    return df_large

def copy_one_month(df_all_p):
    # df_day = pd.DataFrame()
    df_day = copy_dataframe(df_all_p)

    df_day['ID'] = pd.Series(['9999' for i in range(df_day.shape[0])])

    datetime_format = '%Y-%m-%d %H:%M:%S.%f'

    dt = '2019-04-03 00:00:00.000'
    date_t = datetime.datetime.fromtimestamp(time.mktime(time.strptime(dt, datetime_format)))
    midnight = '00:00:00.000'
    
    time_list = np.array([(date_t + timedelta(seconds=calc_sec(midnight)+(T*i))).strftime(datetime_format) for i in range(copy_length)])

    df_day['timestamp'] = pd.Series(time_list)
    print(time_list.shape)

    return df_day

    # # Store Cleaned Data in CSV

def export_copied_data(df_day, cleaned_data_path):

    df_day.to_csv(cleaned_data_path)