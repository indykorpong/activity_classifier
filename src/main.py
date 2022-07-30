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
# path_to_module = '/Users/Indy/Desktop/coding/Dementia_proj/src'

sys.path.append(path_to_module)
os.chdir(path_to_module)

# # Set data path

basepath = '/var/www/html/python/mysql_connect/'
# basepath = '/Users/Indy/Desktop/coding/Dementia_proj/'

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

def clean_table_data():
    mydb, mycursor = connect_to_database()
    sql_list = [
        "SET GLOBAL connect_timeout=28800", 
        "SET GLOBAL wait_timeout=28800", 
        "SET GLOBAL interactive_timeout=28800"
        ]

    for s in sql_list:
        mycursor.execute(s)
    mydb.commit()

def get_all_day_result(all_patients, batch_size=10000):
    # Load data, predict activities, and get summary for each patient

    for user_id in all_patients:

        #### Load and preprocess data

        all_status = reset_status()
        all_status[0] = status_started

        start_time = time_str_now()

        # StartTime, EndTime, UserID, ProcessName, StartingData, EndingData, ProcessStatus
        insert_db_status(start_time, None, user_id, 'LOAD DATA', None, None, all_status[0])
        # try:

        df_all_p = load_raw_data(user_id)

        print("loaded data of user: {}".format(user_id))
        print('all status:', all_status)
        print('df_all_p shape:', df_all_p.shape)

        if(not df_all_p.empty):
            df_all_p['timestamp'] = df_all_p['timestamp'].apply(lambda x: np.datetime64(x))
            insert_db_act_log(df_all_p)
            print('inserted to ActivityLog')

        all_status[0] = status_stopped
        
        # except:
        #     all_status[0] = status_error 

        stop_time = time_str_now()

        if(df_all_p.empty and all_status[0]!=status_error):
            print('no new data')
            all_status[0] = status_error
            insert_db_status(start_time, stop_time, user_id, 'LOAD DATA', None, \
                None, all_status[0])

        else:
            insert_db_status(start_time, stop_time, user_id, 'LOAD DATA', df_all_p.loc[0, 'timestamp'], \
                df_all_p.loc[df_all_p.shape[0]-1, 'timestamp'], all_status[0])

        print('all status:', all_status)

        #### Predict
    
        # Get the unpredicted data from ActivityLog table in database
        mydb, mycursor = connect_to_database()
        current_idx_sql = "SELECT Idx FROM ActivityLog WHERE UserID={} and (LoadedFlag=False or LABEL IS NULL) ORDER BY Idx ASC LIMIT 1;".format(user_id)
        
        # mycursor.execute(act_log_2_sql)
        mycursor.execute(current_idx_sql)
        result = mycursor.fetchone()
        print('rowcount:', mycursor.rowcount)
        if(mycursor.rowcount>0):
            current_idx = result[0]
        else:
            continue

        last_res_sql = "SELECT @LastResultIdx := Idx FROM ActivityLog WHERE (LoadedFlag=False or LABEL IS NULL) and UserID={} ORDER BY Idx DESC LIMIT 1;".format(user_id)
        mycursor.execute(last_res_sql)
        result = mycursor.fetchone()
        if(mycursor.rowcount>0):
            last_idx = result[0]
        else:
            last_idx = 0
        print('last idx:', last_idx)

        mydb.commit()

        while(current_idx<last_idx):
            print('start fetching unpredicted data of user:', user_id)
            df_to_predict, new_current_idx = get_unpredicted_data(user_id, current_idx)
            print('current idx:', current_idx)
            # print('new current idx:', new_current_idx)

            if(df_to_predict.empty):
                print('empty dataframe')
                break
            # if(current_idx>last_idx):
            #     break

            df_to_predict = df_to_predict.reset_index(drop=True)
            print('df to predict shape', df_to_predict.shape)
            print(df_to_predict.head(3))
            print(df_to_predict.tail(3))
            
            current_idx = new_current_idx
            print('current idx 2:', current_idx)

            all_status = reset_status()
            print('reset status:', all_status)
            print('finished fetching unpredicted data of user:', user_id)

            H = 10          # window size of bai's equation
            if(df_to_predict.shape[0]<H):
                break       # stop fetching data and break from loop because the data is too short

            all_status[1] = status_started
            start_time = time_str_now()
            insert_db_status(start_time, None, user_id, 'PREDICT', None, None, all_status[1])

            # try:
            print("started predicting on user: {}".format(user_id))
            
            # Predict activity labels of the patient
            cols = ['x', 'y', 'z']
            for c in cols:
                filt = df_to_predict[c].notnull()
                df_to_predict = df_to_predict[filt]
            df_to_predict = df_to_predict.reset_index(drop=True)

            df_all_p_sorted = predict_label(df_to_predict)
            df_all_p_sorted['timestamp'] = df_all_p_sorted['timestamp'].apply(lambda x: np.datetime64(x))
            print(df_all_p_sorted.head(3))

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

            # try:
            print("started summarizing on user: {}".format(user_id))

            # Summarize the predicted results and get activity summary and activity period for the patient
            df_summary_all, df_act_period = get_summary(df_all_p_sorted)

            # Insert the summary and activity period into HourlyActivitySummary table 
            # and ActivityPeriod table in database respectively
            insert_db_hourly_summary(df_summary_all)
            insert_db_act_period(df_act_period)

            print('finished summarizing on user: {}'.format(user_id))

            all_status[2] = status_stopped

            # Set the SummarizedFlag in ActivityLog table in database to TRUE(1)
            update_summarized_flag(user_id, current_idx)

            # except:
            #     all_status[2] = status_error
            
            stop_time = time_str_now()
            
            start_data = df_all_p_sorted.loc[0, 'timestamp'].strftime(datetime_format)
            end_data = df_all_p_sorted.loc[df_all_p_sorted.shape[0]-1, 'timestamp'].strftime(datetime_format)

            insert_db_status(start_time, stop_time, user_id, 'SUMMARIZE RESULTS', start_data, \
                end_data, all_status[2])

            print('all status:', all_status)

def main_function():
    all_patients = get_distinct_user_ids()
    # all_patients = [17]
    print('all user ids:', all_patients)

    create_temp_table()     # Import data from original tables to temporary tables
    clean_table_data()

    get_all_day_result(all_patients)

if(__name__=='__main__'):
    schedule_time = "00:00"

    # Schedule the program to run main function every XXXX
    # schedule.every().day.at(schedule_time).do(main_function)
    schedule.every().minute.do(main_function)

    while 1:
        schedule.run_pending()
        time.sleep(1)
        print('.')

    # main_function()
        