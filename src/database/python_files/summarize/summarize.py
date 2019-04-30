# # Import Libraries

import pickle
import ruptures as rpt
import numpy as np
import pandas as pd
import sys
import os

from queue import Queue
from datetime import date, datetime, timedelta

# path_to_module = '/var/www/html/python/mysql_connect/python_files'

# sys.path.append(path_to_module)
# os.chdir(path_to_module)

# # # Set data path

# basepath = '/var/www/html/python/mysql_connect/'
    
# datapath = basepath + 'DDC_Data/'
# mypath = basepath + 'DDC_Data/raw/'

date_format = '%Y-%m-%d'
time_format = '%H:%M:%S.%f'
datetime_format = '{} {}'.format(date_format, time_format)

n_hours = 24    # hours in 1 day

midnight = datetime.strptime('00:00:00.000', time_format)
midnight_time = midnight.time()

zero_date = datetime(1900, 1, 1)

act_label_list = [0, 1, 2, 3]       # sit sleep stand walk
act_duration_cols = ['DurationSit', 'DurationSleep', 'DurationStand', 'DurationWalk']
act_count_cols = ['CountSit', 'CountSleep', 'CountStand', 'CountWalk']

def get_df_continuous_list(df_i):
    df_cont_list = []

    keep_idx = 0

    for i in range(df_i.shape[0]):

        if(df_i.loc[i, 'y_pred']!=df_i.loc[keep_idx, 'y_pred'] or \
            df_i.loc[i, 'date']!=df_i.loc[keep_idx, 'date']):
            df_cont_list.append(df_i[keep_idx:i])
            keep_idx = i
        elif(i==df_i.shape[0]-1):
            df_cont_list.append(df_i[keep_idx:df_i.shape[0]])

    return df_cont_list

def separate_hourly(df_cont_list):
    df_sep_list = []
    time_col_idx = 2
    cols = ['UserID', 'date', 'time', 'x', 'y', 'z', 'HR', 'y_pred']

    q = Queue()

    for i in range(len(df_cont_list)):
        df_segment = df_cont_list[i].copy()
        df_segment['time'] = df_segment['time'].apply(lambda x: datetime.strptime(x, time_format))
        q.put(df_segment)

    while(not q.empty()):
        df_segment = q.get()

        start_time_of_segment = df_segment.iloc[0, time_col_idx]
        end_time_of_segment = df_segment.iloc[len(df_segment)-1, time_col_idx]

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

                if(df_seg_1.shape[0]!=0 and df_seg_1.shape[0]<df_segment.shape[0]):
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
    cols = ['UserID', 'Date', 'ActualFrom', 'ActualUntil', 'Label']

    for i in range(len(df_sep_list)):
        df_sep_list_i = df_sep_list[i].copy()
        df_sep_list_i = df_sep_list_i.reset_index(drop=True)

        if(not df_sep_list_i.empty):
            df_temp = pd.DataFrame({
                'UserID': int(df_sep_list_i.loc[0, 'UserID']),
                'Date': df_sep_list_i.loc[0, 'date'],
                'ActualFrom': df_sep_list_i.loc[0, 'time'],
                'ActualUntil': df_sep_list_i.loc[df_sep_list_i.shape[0]-1, 'time'],
                'Label': int(df_sep_list_i.loc[0, 'y_pred'])
            }, index=[0])
            df_act_period_i = df_act_period_i.append(df_temp)

    if(not df_act_period_i.empty):
        df_act_period_i = df_act_period_i.sort_values(by=['Date','ActualFrom'], axis=0)
        df_act_period_i = df_act_period_i.reset_index(drop=True)

        return df_act_period_i[cols]

    return df_act_period_i

def get_df_summary(df_act_period):
    df_summary_all = pd.DataFrame()
    
    grouped = df_act_period.groupby(by='Date')

    for x in grouped:
        date_group = x[0]
        df_act_period_i = grouped.get_group(date_group)
        df_act_period_i = df_act_period_i.reset_index(drop=True)

        df_summary = pd.DataFrame()

        for h in range(n_hours):
            base_time = midnight + timedelta(hours=h)
            end_time = midnight + timedelta(hours=h+1)

            df_temp = pd.DataFrame({
                'UserID': df_act_period_i.loc[0, 'UserID'],
                'Date': df_act_period_i.loc[0, 'Date'],
                'TimeFrom': base_time,       # starting timestamp (hourly, datetime)
                'TimeUntil': end_time,       # ending timestamp (hourly, datetime)
                'ActualFrom': midnight,
                'ActualUntil': midnight,
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
                'CountActiveToInactive': 0,
                'DurationPerAction': midnight
            }, index=[0])

            df_summary = df_summary.append(df_temp)

        df_summary = df_summary.reset_index(drop=True)

        for h in range(n_hours):
            base_time = midnight + timedelta(hours=h)
            end_time = midnight + timedelta(hours=h+1)

            tick = 0
            for i in range(df_act_period_i.shape[0]):

                if(base_time<=df_act_period_i.loc[i, 'ActualFrom'] and 
                    end_time>=df_act_period_i.loc[i, 'ActualUntil'] and tick==0):
                    for j in range(df_summary.shape[0]):
                        if(df_summary.loc[j, 'TimeFrom']==base_time):
                            df_summary.loc[j, 'ActualFrom'] = df_act_period_i.loc[i, 'ActualFrom']
                            tick = 1
                            
                if(base_time<=df_act_period_i.loc[i, 'ActualFrom'] and 
                    end_time>=df_act_period_i.loc[i, 'ActualUntil'] and tick==1):
                    for j in range(df_summary.shape[0]):
                        if(df_summary.loc[j, 'TimeUntil']==end_time):
                            df_summary.loc[j, 'ActualUntil'] = df_act_period_i.loc[i, 'ActualUntil']

                elif(end_time<df_act_period_i.loc[i, 'ActualUntil'] and tick==1):
                    break

            act_duration_list = [midnight for i in range(4)]
            act_count_list = [0 for i in range(4)]

            for i in range(df_act_period_i.shape[0]):
                if(base_time<=df_act_period_i.loc[i, 'ActualFrom'] and 
                    end_time>=df_act_period_i.loc[i, 'ActualUntil']):
                    act_duration_list[df_act_period_i.loc[i, 'Label']] += \
                        df_act_period_i.loc[i, 'ActualUntil'] - df_act_period_i.loc[i, 'ActualFrom']
                    act_count_list[df_act_period_i.loc[i, 'Label']] += 1

                transition_found = 0
                if(i>0 and (df_act_period_i.loc[i, 'Label']==0 or df_act_period_i.loc[i, 'Label']==1) and 
                    (df_act_period_i.loc[i-1, 'Label']==2 or df_act_period_i.loc[i-1, 'Label']==3)):
                    transition_found = 1

                elif(i>0 and (df_act_period_i.loc[i, 'Label']==2 or df_act_period_i.loc[i, 'Label']==3) and 
                    (df_act_period_i.loc[i-1, 'Label']==0 or df_act_period_i.loc[i-1, 'Label']==1)):
                    transition_found = 1

                if(transition_found==1):
                    for j in range(df_summary.shape[0]):
                        if(df_summary.loc[j, 'TimeFrom']==base_time):
                            df_summary.loc[j, 'CountActiveToInactive'] += 1

            for j in range(df_summary.shape[0]):
                if(df_summary.loc[j, 'TimeFrom']==base_time):
                    for lb in act_label_list:
                        df_summary.loc[j, act_duration_cols[lb]] = act_duration_list[lb]
                        df_summary.loc[j, act_count_cols[lb]] = act_count_list[lb]
                        if(lb==0 or lb==1):         # sit or sleep
                            df_summary.loc[j, 'CountInactive'] += act_count_list[lb]
                        elif(lb==2 or lb==3):
                            df_summary.loc[j, 'CountActive'] += act_count_list[lb]

                    df_summary.loc[j, 'TotalDuration'] = np.sum([(df_summary.loc[j, act_duration_cols[lb]] - zero_date) for lb in act_label_list])
                    df_summary.loc[j, 'TotalDuration'] += zero_date

                    df_summary.loc[j, 'CountTotal'] = np.sum([df_summary.loc[j, act_count_cols[lb]] for lb in act_label_list])

                    if(df_summary.loc[j, 'CountTotal']!=0):
                        total_td = df_summary.loc[j, 'TotalDuration'] - zero_date
                        micro_seconds = total_td.microseconds + 1000000 * (total_td.seconds + 86400 * total_td.days)
                        duration_per_action = micro_seconds/ df_summary.loc[j, 'CountTotal']

                        df_summary.loc[j, 'DurationPerAction'] = zero_date + timedelta(microseconds=duration_per_action)

                    break
                    
            cols = ['UserID', 'Date', 'TimeFrom', 'TimeUntil', 'ActualFrom', 'ActualUntil', 
                'DurationSit', 'DurationSleep', 'DurationStand', 'DurationWalk', 'TotalDuration', 
                'CountSit', 'CountSleep', 'CountStand', 'CountWalk', 
                'CountInactive', 'CountActive', 'CountTotal', 
                'CountActiveToInactive', 'DurationPerAction']
        
            df_summary = df_summary.reindex(columns=cols)


        df_summary_all = df_summary_all.append(df_summary)

    return df_summary_all

def get_summary(df_all):

    df_act_period = pd.DataFrame()

    df_all['date'] = df_all['timestamp'].apply(lambda x: x.date())
    df_all['time'] = df_all['timestamp'].apply(lambda x: x.strftime('%H:%M:%S.%f'))

    cols = ['UserID','date','time','x','y','z','HR','y_pred']
    df_all = df_all[cols]

    df_cont_list = get_df_continuous_list(df_all)

    df_sep_list = separate_hourly(df_cont_list)

    df_act_period_i = get_act_period(df_sep_list)
    df_act_period = df_act_period.append(df_act_period_i)
    
    cols = ['UserID', 'Date', 'ActualFrom', 'ActualUntil', 'Label']
    df_act_period = df_act_period.reindex(columns=cols)

    df_summary = get_df_summary(df_act_period)

    df_act_period['ActualFrom'] = df_act_period['ActualFrom'].apply(lambda x: x.strftime(time_format))
    df_act_period['ActualUntil'] = df_act_period['ActualUntil'].apply(lambda x: x.strftime(time_format))    

    return df_summary, df_act_period
    