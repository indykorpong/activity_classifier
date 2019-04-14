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

print('argv[1] =', on_server)


from os import listdir, walk
from os.path import isfile, join
from datetime import datetime

from preprocess.data_preprocess import load_all_data, export_cleaned_data
from predict.predict import predict_label, export_predicted_data
from summarize.summarize import get_summarized_data, export_summarized_data

from preprocess.copy_data import load_raw_data, load_raw_data_2, copy_one_month, export_copied_data
from insert_db.insert_db import connect_to_database, insert_db_act_period, insert_db_all_day_summary, insert_db_patient, insert_db_status, reset_status


# # All Day Data

# In[6]:

def time_str_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

status_started = 0
status_stopped = 1
status_error = -1
status_sleep = -2

def get_all_day_result(all_status, all_patients, date_to_retrieve):
    # Load data
    all_status = reset_status()

    all_status[0] = status_started
    start_time = time_str_now()
    print(start_time)
    insert_db_status('LOAD DATA', start_time, None, all_status[0])
    # try:
    df_all_p = load_raw_data(all_patients, date_to_retrieve)
    print(df_all_p.head(5))
    # df_large = copy_one_month(df_all_p)
    df_large = df_all_p

    all_status[0] = status_stopped
    
    # except:
    #     all_status[0] = status_error 

    stop_time = time_str_now()
    insert_db_status('LOAD DATA', start_time, stop_time, all_status[0])
    print(all_status)
        
    # Predict

    chunk_length = 100000
    if(df_large.shape[0]>chunk_length):
        loop_length = df_large.shape[0] - chunk_length
    else:
        loop_length = df_large.shape[0]

    for i in range(0, loop_length, chunk_length):
        all_status = reset_status()

        df_chunk = df_large[i:i+chunk_length]
        df_chunk = df_chunk.reset_index(drop=True)
        
        all_status[1] = status_started
        start_time = time_str_now()
        insert_db_status('PREDICT', start_time, None, all_status[1])
        # try:
        print("started predicting")
        df_all_p_sorted = predict_label(df_chunk, i)
        print(df_all_p_sorted.head(5))
        insert_db_patient(df_all_p_sorted)
        print("finished predicting")

        all_status[1] = status_stopped
        
        # except:
        #     all_status[1] = status_error

        stop_time = time_str_now()
        print(all_status, start_time, stop_time)
        insert_db_status('PREDICT', start_time, stop_time, all_status[1])

        # # Analyze Predicted Results

        all_status[2] = status_started
        start_time = time_str_now()
        insert_db_status('SUMMARIZE RESULTS', start_time, None, all_status[2])
        # try:
        df_summary_all, df_act_period = get_summarized_data(df_all_p_sorted, all_patients)
        print('finished summarizing')
        insert_db_all_day_summary(df_summary_all)
        insert_db_act_period(df_act_period)

        all_status[2] = status_stopped
        
        # except:
        #   all_status[2] = status_error
        
        stop_time = time_str_now()
        insert_db_status('SUMMARIZE RESULTS', start_time, stop_time, all_status[2])

        print('all status', all_status)

    return all_status

def main_function():
    status_load = 0
    status_predict = 0
    status_summary = 0
    all_status = [status_load, status_predict, status_summary]

    # insert_db_status('SUMMARIZE RESULTS', datetime.now(), datetime.now(), status_error)

    all_patients = ['11']
    date_to_retrieve = '2019-04-10'
    all_status = get_all_day_result(all_status, all_patients, date_to_retrieve)

# print(df_summary_all)

if(__name__=='__main__'):
    main_function()
