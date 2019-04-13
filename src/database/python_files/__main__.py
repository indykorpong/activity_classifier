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
from insert_db.insert_db import insert_db_act_period, insert_db_all_day_summary, insert_db_patient, insert_db_status, reset_status




# # Connect to MySQL Database

def connect_to_database():
    
    if(not on_server and at_home==''):
        user = 'root'
        passwd = "1amdjvr'LN"
    elif(not on_server and at_home=='C:'):
        user = 'root'
        passwd = ''
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
status_sleep = -2

def get_all_day_result(all_status, all_patients, date_to_retrieve, mydb, mycursor):
    # Load data
    all_status = reset_status()

    all_status[0] = status_started
    start_time = time_str_now()
    print(start_time)
    insert_db_status('LOAD DATA', start_time, None, all_status[0], mydb, mycursor)
    # try:
    df_all_p = load_raw_data(all_patients, date_to_retrieve, mydb, mycursor)
    print(df_all_p.head(5))
    df_large = copy_one_month(df_all_p)

    all_status[0] = status_stopped
    
    # except:
    #     all_status[0] = status_error 

    stop_time = time_str_now()
    insert_db_status('LOAD DATA', start_time, stop_time, all_status[0], mydb, mycursor)
    print(all_status)
        
    # Predict

    chunk_length = 100000
    for i in range(0, df_large.shape[0]-chunk_length, chunk_length):
        all_status = reset_status()

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
        
        # except:
        #     all_status[1] = status_error

        stop_time = time_str_now()
        print(all_status, start_time, stop_time)
        insert_db_status('PREDICT', start_time, stop_time, all_status[1], mydb, mycursor)

        # # Analyze Predicted Results

        all_status[2] = status_started
        start_time = time_str_now()
        insert_db_status('SUMMARIZE RESULTS', start_time, None, all_status[2], mydb, mycursor)
        # try:
        df_summary_all, df_act_period = get_summarized_data(df_all_p_sorted, all_patients)
        print('finished summarizing')
        insert_db_all_day_summary(df_summary_all, mydb, mycursor)
        insert_db_act_period(df_act_period, mydb, mycursor)

        all_status[2] = status_stopped
        
        # except:
        #   all_status[2] = status_error
        
        stop_time = time_str_now()
        insert_db_status('SUMMARIZE RESULTS', start_time, stop_time, all_status[2], mydb, mycursor)

        print('all status', all_status)

    return all_status

def main_function():
    mydb, mycursor = connect_to_database()

    status_load = 0
    status_predict = 0
    status_summary = 0
    all_status = [status_load, status_predict, status_summary]

    insert_db_status('SUMMARIZE RESULTS', datetime.now(), datetime.now(), status_error, mydb, mycursor)

    all_patients = ['11']
    date_to_retrieve = '2019-04-10'
    all_status = get_all_day_result(all_status, all_patients, date_to_retrieve, mydb, mycursor)

# print(df_summary_all)

if(__name__=='__main__'):
    main_function()
