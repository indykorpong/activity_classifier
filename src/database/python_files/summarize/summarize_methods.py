# # Import Libraries

# In[1]:

import pickle
import ruptures as rpt
import numpy as np
import pandas as pd
import sys
import os

from datetime import date, datetime, timedelta

from load_data.load_dataset import calc_sec, calc_ts

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

label_list = ['sit', 'sleep', 'stand', 'walk']
label_dict = {'sit': 0, 'sleep': 1, 'stand': 2, 'walk': 3}

date_format = '%Y-%m-%d'
time_format = '%H:%M:%S.%f'

onehour = 60*60
n_hours = 24    # hours in 1 day

midnight = datetime.strptime('00:00:00.000', time_format)

def get_df_continuous_list(df_i):
    df_cont_list = []

    keep_idx = 0

    for i in range(df_i.shape[0]):
        keep_lb = df_i.loc[keep_idx, 'y_pred']

        if(df_i.loc[i, 'y_pred']!=keep_lb):
            df_cont_list.append(df_i[keep_idx:i])
            keep_idx = i

    return df_cont_list

def separate_hourly(df_cont_list):
    df_sep_list = []

    for i in range(len(df_cont_list)):
        df_segment = df_cont_list[i]
        print('df segment')
        print(df_segment)
        print('========')
        df_segment['time'] = df_segment['time'].apply(lambda x: datetime.strptime(x, time_format))
        print(':D')

        start_time_of_segment = df_segment.loc[0, 'time']
        end_time_of_segment = df_segment.loc[len(df_segment)-1, 'time']

        for h in range(n_hours):
            base_time = midnight + timedelta(hours=h)
            end_time = midnight + timedelta(hours=h+1)

            if(base_time<=start_time_of_segment and end_time>=end_time_of_segment):
                df_sep_list.append(df_segment)
                break
            elif(base_time<=start_time_of_segment and end_time<end_time_of_segment):
                df_seg_1 = pd.DataFrame()
                for j in range(df_segment.shape[0]):
                    if(df_segment.loc[j, 'time']<=end_time):
                        df_seg_1 = df_seg_1.append(df_segment[j])
                    else:
                        keep_idx = j
                        break

                df_sep_list.append(df_seg_1)

                if(df_seg_1.shape[0]<df_segment.shape[0]):
                    print('???')
                    df_seg_2 = pd.DataFrame()
                    for j in range(keep_idx, df_segment.shape[0]):
                        df_seg_2 = df_seg_2.append(df_segment[j])

                    df_cont_list.append(df_seg_2)

    return df_sep_list

def get_df_summary_all(df_all, all_patients):
    df_summary = pd.DataFrame()
    df_act_period = pd.DataFrame()

    df_all['date'] = df_all['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df_all['time'] = df_all['timestamp'].apply(lambda x: x.strftime('%H:%M:%S.%f'))

    cols = ['ID','date','time','x','y','z','HR','y_pred']
    df_all = df_all[cols]

    for subject_i in all_patients:
        filter_ = [str(df_all.loc[i, 'ID'])==subject_i for i in range(df_all.shape[0])]
        df_all_i = df_all[filter_]

        df_cont_list = get_df_continuous_list(df_all_i)
        print(df_cont_list)
        df_sep_list = separate_hourly(df_cont_list)

        print(df_sep_list)