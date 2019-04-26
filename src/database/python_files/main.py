#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
import os
import math
import mysql.connector
import sys

path_to_module = '/var/www/html/python/mysql_connect/python_files'

sys.path.append(path_to_module)
os.chdir(path_to_module)

# # Set data path

basepath = '/var/www/html/python/mysql_connect/'
    
datapath = basepath + 'DDC_Data/'
mypath = basepath + 'DDC_Data/raw/'

from os import listdir, walk
from os.path import isfile, join
from datetime import datetime, timedelta

from predict.predict import predict_label

from summarize.summarize import get_summary

from load_data.load_data import load_raw_data

from insert_db.insert_db import connect_to_database, reset_status, get_distinct_user_ids, update_summarized_flag
from insert_db.insert_db import insert_db_act_period, insert_db_hourly_summary, insert_db_act_log, insert_db_status

# # All Day Data
datetime_format = "%Y-%m-%d %H:%M:%S.%f"

def time_str_now():
    return datetime.now().strftime(datetime_format)[:-3]

status_started = 0
status_stopped = 1
status_error = -1

def get_all_day_result(all_patients):
    # Load data

    all_status = reset_status()

    all_status[0] = status_started
    start_time = time_str_now()
    print(start_time)

    for p in all_patients:
        # StartTime, EndTime, UserID, ProcessName, StartingData, EndingData, ProcessStatus
        insert_db_status(start_time, None, p, 'LOAD DATA', None, None, all_status[0])
        # try:
        df_all_p = load_raw_data(p)
        print(df_all_p.head(5))
        print('...')
        print(df_all_p.tail(5))
        print('...')

        all_status[0] = status_stopped
        
        # except:
        #     all_status[0] = status_error 

        if(df_all_p.empty):
            print('empty')
            all_status[0] = status_error

        stop_time = time_str_now()
        insert_db_status(start_time, stop_time, p, 'LOAD DATA', df_all_p.loc[0, 'timestamp'].strftime(datetime_format), \
            df_all_p.loc[df_all_p.shape[0]-1, 'timestamp'].strftime(datetime_format), all_status[0])

        print(all_status)

        if(df_all_p.empty):
            return
        
        # Predict

        # chunk_length = 1000
        # if(df_all_p.shape[0]>chunk_length):
        #     loop_length = df_all_p.shape[0] - chunk_length
        # else:
        loop_length = df_all_p.shape[0]
        chunk_length = loop_length

        for i in range(0, loop_length, chunk_length):
            all_status = reset_status()
            print('reset status:', all_status)

            df_chunk = df_all_p[i:i+chunk_length]
            df_chunk = df_chunk.reset_index(drop=True)
            
            window_length = 60
            if(df_chunk.shape[0]<window_length):
                continue

            all_status[1] = status_started
            start_time = time_str_now()
            insert_db_status(start_time, None, p, 'PREDICT', None, None, all_status[1])
            # try:
            print("started predicting")
            df_all_p_sorted = predict_label(df_chunk)
            print(df_all_p_sorted.head(5))
            print()
            print(df_all_p_sorted.tail(5))
            print('...')
            insert_db_act_log(df_all_p_sorted)
            print("finished predicting")

            all_status[1] = status_stopped
            
            # except:
            #     all_status[1] = status_error

            stop_time = time_str_now()
            print(all_status, start_time, stop_time)
            insert_db_status(start_time, stop_time, p, 'PREDICT', df_all_p_sorted.loc[0, 'timestamp'].strftime(datetime_format), \
                df_all_p_sorted.loc[df_all_p_sorted.shape[0]-1, 'timestamp'].strftime(datetime_format), all_status[1])

            # # Analyze Predicted Results

            all_status[2] = status_started
            start_time = time_str_now()
            insert_db_status(start_time, None, p, 'SUMMARIZE RESULTS', None, None, all_status[2])
            # try:
            df_summary_all, df_act_period = get_summary(df_all_p_sorted)

            print('finished summarizing')
            insert_db_hourly_summary(df_summary_all)
            insert_db_act_period(df_act_period)

            all_status[2] = status_stopped
            update_summarized_flag(p)

            # except:
            #   all_status[2] = status_error
            
            stop_time = time_str_now()
            insert_db_status(start_time, stop_time, p, 'SUMMARIZE RESULTS', df_all_p_sorted.loc[0, 'timestamp'].strftime(datetime_format), \
                df_all_p_sorted.loc[df_all_p_sorted.shape[0]-1, 'timestamp'].strftime(datetime_format), all_status[2])

            print('all status', all_status)

def delete_from_table():
    mydb, mycursor = connect_to_database()
    sql = "update cu_amd.acc_log_2 set loaded_flag=NULL;"
    mycursor.execute(sql)

    sql = "update cu_amd.hr_log_2 set loaded_flag=NULL;"
    mycursor.execute(sql)

    sql = "delete from cu_amd.ActivityLog;"
    mycursor.execute(sql)

    sql = "delete from cu_amd.ActivityPeriod;"
    mycursor.execute(sql)

    sql = "delete from cu_amd.HourlyActivitySummary;"
    mycursor.execute(sql)
    mydb.commit()

def main_function():
    all_patients = get_distinct_user_ids()
    print(all_patients)

    delete_from_table()     # clear table

    get_all_day_result(all_patients)

if(__name__=='__main__'):
    main_function()
