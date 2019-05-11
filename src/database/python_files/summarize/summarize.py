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

zero_datetime = datetime(1900, 1, 1)
zero_date = date(1900, 1, 1)

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

def recursive_segment(df_segment, df_hour_dict, a, b):
    if(df_segment.shape[0]<=0 or df_segment.empty):
        print('here2')
        return df_hour_dict, a

    keep = 0
    df_segment = df_segment.reset_index(drop=True)

    print('a and b:', a, b)
    for j in range(df_segment.shape[0]):
        if(keep<df_segment.shape[0] and df_segment.loc[keep, 'timestamp']>a and df_segment.loc[df_segment.shape[0]-1, 'timestamp']<=b):
            df_hour_dict[a] = pd.concat([df_hour_dict[a], df_segment[keep:df_segment.shape[0]]], ignore_index=True)
            keep = df_segment.shape[0]
        elif(j>0 and j-keep>0 and df_segment.loc[keep, 'timestamp']>a and df_segment.loc[j-1, 'timestamp']<=b 
            and df_segment.loc[j, 'timestamp']>b):

            df_hour_dict[a] = pd.concat([df_hour_dict[a], df_segment[keep:j]], ignore_index=True)
            df_remainder = df_segment[j:]

            if(df_remainder.shape[0]>0 or not df_remainder.empty):
                return recursive_segment(df_remainder, df_hour_dict, a+timedelta(hours=1), b+timedelta(hours=1))

    return {}, zero_datetime

def separate_hourly(df_cont_list):
    df_sep_list = []

    for i in range(len(df_cont_list)):
        df_segment = df_cont_list[i].copy()
        df_segment['time'] = df_segment['time'].apply(lambda x: datetime.strptime(x, time_format))
        df_segment = df_segment.reset_index(drop=True)

        print('date:', df_segment.loc[0, 'date'])
        days_delta = (df_segment.loc[0, 'date'] - zero_date).days

        df_hour_dict = {}

        for h in range(n_hours):
            hour_start = midnight + timedelta(hours=h) + timedelta(days=days_delta)
            df_hour_dict[hour_start] = pd.DataFrame()

        a_next = midnight + timedelta(days=days_delta)
        for h in range(n_hours):
            a = midnight + timedelta(hours=h) + timedelta(days=days_delta)      # hour start
            b = midnight + timedelta(hours=h+1) + timedelta(days=days_delta)    # hour end
            # print(a, a_next, b)

            if(a_next>a):
                a = a_next

            df_hour_dict_i, a_next = recursive_segment(df_segment, df_hour_dict, a, b)
            if(a_next!=zero_datetime):
                for t in range(a, a_next, timedelta(hours=1)):
                    df_hour_dict[t] = df_hour_dict_i[t]
            
        for h in range(n_hours):
            a = midnight + timedelta(hours=h) + timedelta(days=days_delta)      # hour start
            
            if(len(df_hour_dict[a])>0):
                df_sep_list.append(df_hour_dict[a])

    print('df sep list length:', len(df_sep_list))
    print('df sep list[0] length:', len(df_sep_list[0]))
    print('df sep list[-1] length:', len(df_sep_list[-1]))

    return df_sep_list


def get_act_period(df_sep_list):
    df_act_period_i = pd.DataFrame()
    cols = ['UserID', 'Date', 'ActualFrom', 'ActualUntil', 'Label']

    for i in range(len(df_sep_list)):
        df_sep_list_i = df_sep_list[i]

        if(len(df_sep_list_i)!=0):
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

                    df_summary.loc[j, 'TotalDuration'] = np.sum([(df_summary.loc[j, act_duration_cols[lb]] - zero_datetime) for lb in act_label_list])
                    df_summary.loc[j, 'TotalDuration'] += zero_datetime

                    df_summary.loc[j, 'CountTotal'] = np.sum([df_summary.loc[j, act_count_cols[lb]] for lb in act_label_list])

                    if(df_summary.loc[j, 'CountTotal']!=0):
                        total_td = df_summary.loc[j, 'TotalDuration'] - zero_datetime
                        micro_seconds = total_td.microseconds + 1000000 * (total_td.seconds + 86400 * total_td.days)
                        duration_per_action = micro_seconds/ df_summary.loc[j, 'CountTotal']

                        df_summary.loc[j, 'DurationPerAction'] = zero_datetime + timedelta(microseconds=duration_per_action)

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

    # df_all['timestamp'] = df_all['timestamp'].apply(lambda x: datetime.strptime(x, datetime_format))
    df_all['date'] = df_all['timestamp'].apply(lambda x: x.date())
    df_all['time'] = df_all['timestamp'].apply(lambda x: x.strftime('%H:%M:%S.%f'))

    cols = ['UserID','timestamp','date','time','x','y','z','HR','y_pred']
    df_all = df_all[cols]
    
    print('df all shape', df_all.shape)

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
    