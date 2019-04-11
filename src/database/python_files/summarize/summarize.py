#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import sys

on_server = True

if(not on_server):
    path_to_module = '/Users/Indy/Desktop/coding/Dementia_proj/src/database/python_files/'
else:
    path_to_module = '/var/www/html/python/mysql_connect/python_files'
sys.path.append(path_to_module)

from os import listdir, walk
from os.path import isfile, join

from summarize.activity_summary import get_df_summary_all, get_floor_start, get_ceil_finish
from load_data.load_dataset import calc_sec, calc_ts
from predict.preprocessing import convert_time_to_string

def get_duration_per_act(i, df_summary_all):
        from_actual = df_summary_all.loc[i, 'from actual']
        to_actual = df_summary_all.loc[i, 'to actual']
        total_i = df_summary_all.loc[i, 'total count']
        duration_actual = calc_sec(to_actual) - calc_sec(from_actual)

        duration_per_act = duration_actual/total_i

        return convert_time_to_string(duration_per_act)

def get_summarized_data(df_all_p, predicted_data_path=''):
        # # Load Predicted Data

        subj_range = np.hstack((np.arange(2001,2002),np.arange(3001,3006)))
        all_patients = [str(i) for i in subj_range]

        df_all_p_sorted = df_all_p

        df_date = df_all_p_sorted.copy()
        df_date['date'] = df_date['timestamp'].apply(lambda x: x.split(' ')[0])
        df_date['time'] = df_date['timestamp'].apply(lambda x: x.split(' ')[1])

        cols = ['ID','date','time','x','y','z','HR','y_pred']
        df_date = df_date[cols]


        # # Summarize Data

        df_summary_all, df_act_period = get_df_summary_all(all_patients, df_date)

        cols = ['ID', 'date', 'from', 'to', 'y_pred']
        df_act_period = df_act_period[cols]
        df_act_period = df_act_period.reset_index(drop=True)

        df_summary_all = df_summary_all.reset_index(drop=True)

        actual_from_all = []
        for i in range(len(df_summary_all)):
                keep_start = -1

                for j in range(len(df_act_period)):
                        floor_start = get_floor_start(df_act_period.loc[j, 'from'])

                        if(df_act_period.loc[j, 'date']==df_summary_all.loc[i, 'date'] and
                        df_act_period.loc[j, 'ID']==df_summary_all.loc[i, 'ID']):

                                if(floor_start>keep_start and floor_start==calc_sec(df_summary_all.loc[i, 'from'])):

                                        actual_from_all.append(df_act_period.loc[j, 'from'])
                                        keep_start = calc_sec(df_act_period.loc[j, 'from'])

                                elif(floor_start<=keep_start):
                                        break

        actual_to_all = []
        for i in range(len(df_summary_all)-1, -1, -1):
                keep_finish = calc_sec(df_summary_all.loc[i, 'to'])

                for j in range(len(df_act_period)-1, -1, -1):

                        ceil_finish = get_ceil_finish(df_act_period.loc[j, 'to'])


                        if(df_act_period.loc[j, 'date']==df_summary_all.loc[i, 'date'] and
                        df_act_period.loc[j, 'ID']==df_summary_all.loc[i, 'ID']):

                                if(keep_finish==ceil_finish):

                                        actual_to_all.append(df_act_period.loc[j, 'to'])
                                        break

        df_summary_all['from actual'] = pd.Series(actual_from_all)
        df_summary_all['to actual'] = pd.Series(actual_to_all[::-1])

        duration_per_act = [get_duration_per_act(i, df_summary_all) for i in range(len(df_summary_all))]

        df_summary_all['duration per action'] = pd.Series(duration_per_act)

        return df_summary_all, df_act_period


def export_summarized_data(df_summary_all, df_act_period, all_day_summary_path, act_period_path):
        cols = ['ID', 'date', 'from', 'to', 'from actual', 'to actual',
                'sit', 'sleep', 'stand', 'walk', 'total',
                'sit count', 'sleep count', 'stand count', 'walk count',
                'inactive count', 'active count', 'total count', 'transition count', 'duration per action']
        df_summary_all = df_summary_all[cols]

        cols = ['ID', 'date', 'from', 'to', 'y_pred']
        df_act_period = df_act_period[cols]

        df_summary_all.to_csv(all_day_summary_path)
        df_act_period.to_csv(act_period_path)
