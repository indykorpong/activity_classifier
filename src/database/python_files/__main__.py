#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt

import pickle
import os
import math
import mysql.connector
import sys

on_server = True

if(not on_server):
    path_to_module = '/Users/Indy/Desktop/coding/Dementia_proj/src/database/python_files/'
else:
    path_to_module = '/var/www/html/python/mysql_connect/python_files'

sys.path.append(path_to_module)

from os import listdir, walk
from os.path import isfile, join
from datetime import datetime

from preprocess.data_preprocess import load_all_data, export_cleaned_data
from predict.predict import predict_label, export_predicted_data
from summarize.summarize import get_summarized_data, export_summarized_data

from preprocess.copy_data import load_raw_data, load_raw_data_2, copy_one_day, copy_one_month, export_copied_data
from insert_db.insert_db import insert_db_act_period, insert_db_all_day_summary, insert_db_patient, insert_db_status, reset_error_bool

# # Set data path

if(not on_server):
    basepath = '/Users/Indy/Desktop/coding/Dementia_proj/'
else:
    basepath = '/var/www/html/python/mysql_connect/'
    
datapath = basepath + 'DDC_Data/'
mypath = basepath + 'DDC_Data/raw/'

# # Connect to MySQL Database

def connect_to_database():
    
    if(not on_server):
        user = 'root'
        passwd = "1amdjvr'LN"
    else:
        user = 'php'
        passwd = 'HOD8912+php'

    mydb = mysql.connector.connect(
        host='localhost',
        user=user,
        passwd=passwd,
        database='cu_amd'
        )

    print(mydb)

    mycursor = mydb.cursor()

    return mydb, mycursor

# # All Day Data

# In[6]:

def time_str_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

status_started = 0
status_stopped = 1
status_error = -1


def get_all_day_result(mydb, mycursor, all_status):
    # Load data
    load_error, predict_error, summarize_error = reset_error_bool()

    all_status[0] = status_started
    start_time = time_str_now()
    print(start_time)
    insert_db_status('LOAD DATA', start_time, None, all_status[0], mydb, mycursor)
    try:
        df_all_p = load_raw_data_2()
        print(df_all_p.head(5))
        df_large = copy_one_month(df_all_p)

        all_status[0] = status_stopped
        stop_time = time_str_now()
        insert_db_status('LOAD DATA', start_time, stop_time, all_status[0], mydb, mycursor)
    except:
        load_error = True
    
    if(load_error):
        all_status[0] = status_error
        stop_time = time_str_now()
        insert_db_status('LOAD DATA', start_time, stop_time, all_status[0], mydb, mycursor)
        
    # Predict

    chunk_length = 10000
    for i in range(0, df_large.shape[0]-chunk_length, chunk_length):
        load_error, predict_error, summarize_error = reset_error_bool()

        df_chunk = df_large[i:i+chunk_length]
        df_chunk = df_chunk.reset_index(drop=True)
        
        all_status[1] = status_started
        start_time = time_str_now()
        insert_db_status('PREDICT', start_time, None, all_status[1], mydb, mycursor)
        # try:
        print("started predicting")
        df_all_p_sorted = predict_label(df_chunk, i)
        print(df_all_p_sorted.head(5))
        insert_db_patient(df_all_p_sorted, mydb, mycursor)
        print("finished predicting")

        all_status[1] = status_stopped
        stop_time = time_str_now()
        print(all_status[1], start_time, stop_time)
        insert_db_status('PREDICT', start_time, stop_time, all_status[1], mydb, mycursor)
        # except:
        #     print("prediction error")
        #     predict_error = True

        if(predict_error):
            all_status[1] = status_stopped
            stop_time = time_str_now()
            print(all_status[1], start_time, stop_time)
            insert_db_status('PREDICT', start_time, stop_time, all_status[1], mydb, mycursor)

        # # Analyze Predicted Results

        all_status[2] = status_started
        start_time = time_str_now()
        insert_db_status('SUMMARIZE RESULTS', start_time, None, all_status[2], mydb, mycursor)
        try:
            df_summary_all, df_act_period = get_summarized_data(df_all_p_sorted)
            print('finished summarizing')
            insert_db_all_day_summary(df_summary_all, mydb, mycursor)
            insert_db_act_period(df_act_period, mydb, mycursor)

            all_status[2] = status_stopped
            stop_time = time_str_now()
            insert_db_status('SUMMARIZE RESULTS', start_time, stop_time, all_status[2], mydb, mycursor)
        except:
            summarize_error = True

        if(summarize_error):
            all_status[2] = status_error
            stop_time = time_str_now()
            insert_db_status('SUMMARIZE RESULTS', start_time, stop_time, all_status[2], mydb, mycursor)

    return all_status

def main_function():
    mydb, mycursor = connect_to_database()

    status_load = 0
    status_predict = 0
    status_summary = 0
    all_status = [status_load, status_predict, status_summary]

    all_status = get_all_day_result(mydb, mycursor, all_status)

# print(df_summary_all)

if(__name__=='__main__'):
    main_function()
