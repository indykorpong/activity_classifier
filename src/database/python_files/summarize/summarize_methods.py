# # Import Libraries

# In[1]:

import pickle
import ruptures as rpt
import numpy as np
import pandas as pd
import sys
import os

from queue import Queue
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
    time_col_idx = 2
    cols = ['ID', 'date', 'time', 'x', 'y', 'z', 'HR', 'y_pred']

    q = Queue()

    for i in range(len(df_cont_list)):
        df_segment = df_cont_list[i].copy()
        df_segment['time'] = df_segment['time'].apply(lambda x: datetime.strptime(x, time_format))
        q.put(df_segment)

    while(not q.empty()):
        df_segment = q.get()

        print('df segment')
        print(df_segment)
        print('========')
        
        print(':D')

        start_time_of_segment = df_segment.iloc[0, time_col_idx]
        end_time_of_segment = df_segment.iloc[len(df_segment)-1, time_col_idx]
        print(start_time_of_segment)
        print(midnight + timedelta(hours=1))

        for h in range(n_hours):
            base_time = midnight + timedelta(hours=h)
            end_time = midnight + timedelta(hours=h+1)

            if(base_time<=start_time_of_segment and end_time>=end_time_of_segment):
                df_sep_list.append(df_segment)
                break
            elif(base_time<=start_time_of_segment and end_time<end_time_of_segment):
                df_seg_1 = pd.DataFrame()
                for j in range(df_segment.shape[0]):
                    if(df_segment.iloc[j, time_col_idx]<=end_time):
                        df_seg_1 = df_seg_1.append(df_segment.iloc[j])
                    else:
                        keep_idx = j
                        break

                if(df_seg_1.empty):
                    df_sep_list.append(df_seg_1)
                else:
                    df_sep_list.append(df_seg_1[cols])

                print(df_seg_1.shape[0], df_segment.shape[0])

                if(df_seg_1.shape[0]!=0 and df_seg_1.shape[0]<df_segment.shape[0]):
                    print('???')
                    df_seg_2 = pd.DataFrame()
                    for j in range(keep_idx, df_segment.shape[0]):
                        df_seg_2 = df_seg_2.append(df_segment.iloc[j])

                    if(df_seg_2.empty):
                        q.put(df_seg_2)
                    else:
                        q.put(df_seg_2[cols])

    return df_sep_list

def get_act_period(df_sep_list):
    df_act_period_i = pd.DataFrame()
    cols = ['ID', 'Date', 'TimeFrom', 'TimeUntil', 'Label']

    for i in range(len(df_sep_list)):
        df_sep_list_i = df_sep_list[i].copy()
        df_sep_list_i = df_sep_list_i.reset_index(drop=True)

        if(not df_sep_list_i.empty):
            df_temp = pd.DataFrame({
                'ID': int(df_sep_list_i.loc[0, 'ID']),
                'Date': df_sep_list_i.loc[0, 'date'],
                'TimeFrom': df_sep_list_i.loc[0, 'time'],
                'TimeUntil': df_sep_list_i.loc[df_sep_list_i.shape[0]-1, 'time'],
                'Label': int(df_sep_list_i.loc[0, 'y_pred'])
            }, index=[0])
            df_act_period_i = df_act_period_i.append(df_temp)

    df_act_period_i = df_act_period_i.reset_index(drop=True)

    return df_act_period_i[cols]

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

        df_act_period_i = get_act_period(df_sep_list)
        df_act_period = df_act_period.append(df_act_period_i)
        print('::::::::')
        print(df_act_period_i)

    for subject_i in all_patients:
        filter_ = [str(df_act_period.loc[i, 'ID'])==subject_i for i in range(df_act_period.shape[0])]
        df_act_period_i = df_act_period[filter_]
    
        for h in range(n_hours):
            base_time = midnight + timedelta(hours=h)
            end_time = midnight + timedelta(hours=h+1)

            df_temp = pd.DataFrame({
                'ID': df_act_period_i.loc[0, 'ID'],
                'Date': df_act_period_i.loc[0, 'Date'],
                'TimeFrom': base_time.strftime(time_format),
                'TimeUntil': end_time.strftime(time_format),
                'ActualFrom': None,
                'ActualUntil': None,
                'DurationSit': midnight,
                'DurationSleep': midnight,
                'DurationStand': midnight,
                'DurationWalk': midnight,
                'TotalDuration': midnight,
                'CountSit': 0,
                'CountSleep': 0,
                'CountStand': 0,
                'CountWalk': 0,
                'CountInactive': 0,
                'CountActive': 0,
                'CountTotal': 0,
                'CountSit': 0,
                'DurationPerTransition': midnight
            }, index=[0])

            df_summary = df_summary.append(df_temp)

        df_summary = df_summary.reset_index(drop=True)

        for h in range(n_hours):
            base_time = midnight + timedelta(hours=h)
            end_time = midnight + timedelta(hours=h+1)

            tick = 0
            for i in range(df_act_period_i.shape[0]):
                if(base_time<=df_act_period_i.loc[i, 'TimeFrom'] and 
                end_time>=df_act_period_i.loc[i, 'TimeUntil'] and tick==0):
                    for j in range(df_summary.shape[0]):
                        if(df_summary.loc[j, 'TimeFrom']==base_time.strftime(time_format)):
                            df_summary.loc[j, 'ActualFrom'] = df_act_period_i.loc[i, 'TimeFrom']
                            tick = 1
                            
                elif(base_time<=df_act_period_i.loc[i, 'TimeFrom'] and 
                end_time>=df_act_period_i.loc[i, 'TimeUntil'] and tick==1):
                    for j in range(df_summary.shape[0]):
                        if(df_summary.loc[j, 'TimeUntil']==end_time.strftime(time_format)):
                            df_summary.loc[j, 'ActualUntil'] = df_act_period_i.loc[i, 'TimeUntil']

        print(df_summary)