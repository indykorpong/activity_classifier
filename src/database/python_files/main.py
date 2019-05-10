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

# path_to_module = '/var/www/html/python/mysql_connect/python_files'
path_to_module = '/Users/Indy/Desktop/coding/Dementia_proj/src/database/python_files'

sys.path.append(path_to_module)
os.chdir(path_to_module)

# # Set data path

# basepath = '/var/www/html/python/mysql_connect/'
basepath = '/Users/Indy/Desktop/coding/Dementia_proj/'
    
datapath = basepath + 'DDC_Data/'
mypath = basepath + 'DDC_Data/raw/'

from predict.predict import predict_label
from summarize.summarize import get_summary
from load_data.load_data import load_raw_data

from insert_db.insert_db import get_sql_connection
from insert_db.insert_db import connect_to_database, create_temp_table, reset_status, get_distinct_user_ids
from insert_db.insert_db import update_summarized_flag, get_unpredicted_data, get_unsummarized_data
from insert_db.insert_db import insert_db_act_period, insert_db_hourly_summary, insert_db_act_log, insert_db_status

datetime_format = "%Y-%m-%d %H:%M:%S.%f"

# Get the current timestamp
def time_str_now():
    return datetime.now().strftime(datetime_format)[:-3]

status_started = 0
status_stopped = 1
status_error = -1

def get_all_day_result(all_patients):
    # Load data, predict activities, and get summary for each patient

    for user_id in all_patients:

        #### Load and preprocess data

        # Get the reset status of all process status (load, predict, summarize status respectively)
        all_status = reset_status()
        all_status[0] = status_started

        start_time = time_str_now()

        # Insert process status into AuditLog table in database
        # StartTime, EndTime, UserID, ProcessName, StartingData, EndingData, ProcessStatus
        insert_db_status(start_time, None, user_id, 'LOAD DATA', None, None, all_status[0])
        # try:

        # Load raw data from accelerometer_log table in database and preprocess the data
        df_all_p = load_raw_data(user_id)

        print("loaded data of user: {}".format(user_id))
        print('all status:', all_status)

        # If the preprocessed data is not empty for the patient, insert them into
        # ActivityLog table in database
        if(not df_all_p.empty):
            insert_db_act_log(df_all_p)

        all_status[0] = status_stopped
        
        # except:
        #     all_status[0] = status_error 

        stop_time = time_str_now()

        # If there are no new data from smartwatch, insert error status into AuditLog table in database
        if(df_all_p.empty and all_status[0]!=status_error):
            print('no new data')
            all_status[0] = status_error
            insert_db_status(start_time, stop_time, user_id, 'LOAD DATA', None, \
                None, all_status[0])

            continue

        else:
            insert_db_status(start_time, stop_time, user_id, 'LOAD DATA', df_all_p.loc[0, 'timestamp'], \
                df_all_p.loc[df_all_p.shape[0]-1, 'timestamp'], all_status[0])

        print('all status:', all_status)
            
        #### Predict

        # Get the unpredicted data from ActivityLog table in database
        df_to_predict = get_unpredicted_data(user_id)
        df_to_predict = df_to_predict.reset_index(drop=True)
        
        all_status = reset_status()
        print('reset status:', all_status)
        
        # If the data is smaller than the time window length, then skip
        window_length = 60
        if(df_to_predict.shape[0]<window_length):
            continue

        all_status[1] = status_started
        start_time = time_str_now()
        insert_db_status(start_time, None, user_id, 'PREDICT', None, None, all_status[1])

        # try:
        print("started predicting on user: {}".format(user_id))
        
        # Predict activity labels of the patient
        df_all_p_sorted = predict_label(df_to_predict)

        # Update the Label field in ActivityLog table in database
        insert_db_act_log(df_all_p_sorted, update=True)

        print("finished predicting on user: {}".format(user_id))

        all_status[1] = status_stopped
        
        # except:
        #     all_status[1] = status_error

        stop_time = time_str_now()
        print(all_status, start_time, stop_time)
        insert_db_status(start_time, stop_time, user_id, 'PREDICT', df_all_p_sorted.loc[0, 'timestamp'], \
            df_all_p_sorted.loc[df_all_p_sorted.shape[0]-1, 'timestamp'], all_status[1])

        #### Summarize

        all_status[2] = status_started
        start_time = time_str_now()
        insert_db_status(start_time, None, user_id, 'SUMMARIZE RESULTS', None, None, all_status[2])

        # Get the unsummarized data from ActivityLog table in database
        df_to_summarize = get_unsummarized_data(user_id)

        # try:
        print("started summarizing on user: {}".format(user_id))

        # Summarize the predicted results and get activity summary and activity period for the patient
        df_summary_all, df_act_period = get_summary(df_to_summarize)

        # Insert the summary and activity period into HourlyActivitySummary table 
        # and ActivityPeriod table in database respectively
        insert_db_hourly_summary(df_summary_all)
        insert_db_act_period(df_act_period)

        print('finished summarizing on user: {}'.format(user_id))

        all_status[2] = status_stopped

        # Set the SummarizedFlag in ActivityLog table in database to TRUE(1)
        update_summarized_flag(user_id)

        # except:
        #     all_status[2] = status_error
        
        stop_time = time_str_now()
        insert_db_status(start_time, stop_time, user_id, 'SUMMARIZE RESULTS', df_all_p_sorted.loc[0, 'timestamp'], \
            df_all_p_sorted.loc[df_all_p_sorted.shape[0]-1, 'timestamp'], all_status[2])

        print('all status:', all_status)

def main_function():
    all_patients = get_distinct_user_ids()
    print('all patients:', all_patients)

    create_temp_table()     # Import data from original tables to temporary tables

    get_all_day_result(all_patients)

def get_acc_data(user_id):
    cnx = get_sql_connection()

    sql = "SELECT UserID, DateAndTime, X, Y, Z, HR, Label FROM ActivityLog WHERE UserID={};".format(user_id)

    df_to_predict = pd.read_sql(sql, cnx)
    df_to_predict = df_to_predict.rename(columns={
        'DateAndTime': 'timestamp',
        'X': 'x',
        'Y': 'y',
        'Z': 'z',
        'ActivityIndex': 'AI',
        'Label': 'y_pred'
    })

    return df_to_predict

if(__name__=='__main__'):
    # schedule_time = "00:00"

    # # Schedule the program to run main function every XXXX
    # schedule.every().minute.do(main_function)
    # # schedule.every().day.at(schedule_time).do(main_function)

    # while 1:
    #     schedule.run_pending()
    #     time.sleep(1)
    #     print('waiting')

    # main_function()

    all_patients = get_distinct_user_ids()
    print(all_patients)

    f, ax = plt.subplots(nrows=len(all_patients), ncols=1)

    for i, subject_id in enumerate(all_patients):
        df_acc = get_acc_data(subject_id)
        df_acc = df_acc.set_index('timestamp')
        print(df_acc)

        cols = ['x', 'y', 'z']
        ax[i] = df_acc[cols].plot(use_index=True)

    plt.savefig('acc_plot.png', dpi=200)
    plt.show()
        