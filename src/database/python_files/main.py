#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle
import os
import sys
import math
import mysql.connector
import schedule
import time

from datetime import datetime, timedelta

path_to_module = '/var/www/html/python/mysql_connect/python_files'
# path_to_module = '/Users/Indy/Desktop/coding/Dementia_proj/src/database/python_files'

sys.path.append(path_to_module)
os.chdir(path_to_module)

# # Set data path

basepath = '/var/www/html/python/mysql_connect/'
# basepath = '/Users/Indy/Desktop/coding/Dementia_proj/src/database/python_files'
    
datapath = basepath + 'DDC_Data/'
mypath = basepath + 'DDC_Data/raw/'

from predict.predict import predict_label
from summarize.summarize import get_summary
from load_data.load_data import load_raw_data

from insert_db.insert_db import connect_to_database, clear_table, reset_status, get_distinct_user_ids
from insert_db.insert_db import update_summarized_flag, get_unpredicted_data, get_unsummarized_data
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
    
    for user_id in all_patients:
        all_status = reset_status()
        all_status[0] = status_started
        start_time = time_str_now()
        print(start_time)

        # StartTime, EndTime, UserID, ProcessName, StartingData, EndingData, ProcessStatus
        insert_db_status(start_time, None, user_id, 'LOAD DATA', None, None, all_status[0])
        # try:
        df_all_p = load_raw_data(user_id)

        print("loaded data of user: {}".format(user_id))
        print('all status:', all_status)
        print('head')
        print(df_all_p.head())

        if(not df_all_p.empty):
            insert_db_act_log(df_all_p)

        all_status[0] = status_stopped
        
        # except:
        #     all_status[0] = status_error 

        stop_time = time_str_now()

        if(df_all_p.empty and all_status[0]!=status_error):
            print('no new data')
            all_status[0] = status_error
            
            insert_db_status(start_time, stop_time, user_id, 'LOAD DATA', None, \
                None, all_status[0])

            continue

        else:
            insert_db_status(start_time, stop_time, user_id, 'LOAD DATA', df_all_p.loc[0, 'timestamp'], \
                df_all_p.loc[df_all_p.shape[0]-1, 'timestamp'], all_status[0])

        print(all_status)
            
        
        # Predict

        df_to_predict = get_unpredicted_data(user_id)
        
        loop_length = df_all_p.shape[0]
        chunk_length = loop_length

        for i in range(0, loop_length, chunk_length):
            all_status = reset_status()
            print('reset status:', all_status)

            df_chunk = df_to_predict[i:i+chunk_length]
            df_chunk = df_chunk.reset_index(drop=True)
            
            window_length = 60
            if(df_chunk.shape[0]<window_length):
                continue

            all_status[1] = status_started
            start_time = time_str_now()
            insert_db_status(start_time, None, user_id, 'PREDICT', None, None, all_status[1])
            # try:
            print("started predicting")
            
            df_all_p_sorted = predict_label(df_chunk)
            insert_db_act_log(df_all_p_sorted, update=True)

            print("finished predicting")

            all_status[1] = status_stopped
            
            # except:
            #     all_status[1] = status_error

            stop_time = time_str_now()
            print(all_status, start_time, stop_time)
            insert_db_status(start_time, stop_time, user_id, 'PREDICT', df_all_p_sorted.loc[0, 'timestamp'], \
                df_all_p_sorted.loc[df_all_p_sorted.shape[0]-1, 'timestamp'], all_status[1])

            # # Analyze Predicted Results

            all_status[2] = status_started
            start_time = time_str_now()
            insert_db_status(start_time, None, user_id, 'SUMMARIZE RESULTS', None, None, all_status[2])

            df_to_summarize = get_unsummarized_data(user_id)
            print('df to summarize')
            print(df_to_summarize.head())
            print()
            print(df_to_summarize.tail())

            # try:
            df_summary_all, df_act_period = get_summary(df_to_summarize)

            print('finished summarizing')
            insert_db_hourly_summary(df_summary_all)
            insert_db_act_period(df_act_period)

            all_status[2] = status_stopped
            update_summarized_flag(user_id)

            # except:
            #     all_status[2] = status_error
            
            stop_time = time_str_now()
            insert_db_status(start_time, stop_time, user_id, 'SUMMARIZE RESULTS', df_all_p_sorted.loc[0, 'timestamp'], \
                df_all_p_sorted.loc[df_all_p_sorted.shape[0]-1, 'timestamp'], all_status[2])

            print('all status', all_status)

def main_function():
    all_patients = get_distinct_user_ids()
    print(all_patients)

    clear_table()     # delete every row in some tables

    get_all_day_result(all_patients)

if(__name__=='__main__'):
    # schedule_time = "19:17"
    # schedule.every().day.at(schedule_time).do(main_function)

    # while 1:
    #     schedule.run_pending()
    #     time.sleep(1)
    #     print('waiting')

    main_function()