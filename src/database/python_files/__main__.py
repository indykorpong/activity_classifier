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

from os import listdir, walk
from os.path import isfile, join

from preprocess.data_preprocess import load_all_data, export_cleaned_data
from predict.predict import predict_label, export_predicted_data
from summarize.summarize import get_summarized_data, export_summarized_data

from preprocess.copy_data import load_raw_data, copy_one_day, export_copied_data

# # Set data path

datapath = '../DDC_Data/'
basepath = '../'

# # Connect to MySQL Database

def connect_to_database():
    mydb = mysql.connector.connect(
        host='localhost',
        user='php',
        passwd='HOD8912+php',
        database='test_indy'
        )

    print(mydb)

    mycursor = mydb.cursor()

    return mydb, mycursor


# # Load Dataset

def get_all_patients_result():

    subj_range = np.hstack((np.arange(2001,2002),np.arange(3001,3006)))
    all_patients = [str(i) for i in subj_range]

    # In[37]:


    cleaned_data_path = datapath + 'cleaned/cleaned_data_' + all_patients[0] + '_to_' + all_patients[-1] + '.csv'
    predicted_data_path = datapath + 'prediction/predicted_data_' + all_patients[0] + '_to_' + all_patients[-1] + '.csv'
    all_day_summary_path = datapath + 'summary/all_day_summary_' + all_patients[0] + '_to_' + all_patients[-1] + '.csv'
    act_period_path = datapath + 'summary/activity_period_' + all_patients[0] + '_to_' + all_patients[-1] + '.csv'


    df_all_p_sorted = load_all_data(all_patients)
    export_cleaned_data(df_all_p_sorted, cleaned_data_path)

    # # Predict Labels

    df_all_p_sorted = predict_label(predicted_data_path)
    export_predicted_data(df_all_p_sorted, predicted_data_path)

    # In[39]:

    df_summary_all, df_act_period = get_summarized_data(predicted_data_path)
    export_summarized_data(df_summary_all, df_act_period, all_day_summary_path, act_period_path)

    return df_all_p_sorted, df_summary_all, df_act_period

# # All Day Data

# In[6]:



def get_all_day_result():
    cleaned_data_path = datapath + 'cleaned/cleaned_data_9999.csv'
    predicted_data_path = datapath + 'prediction/predicted_data_9999.csv'
    all_day_summary_path = datapath + 'summary/all_day_summary_9999.csv'
    act_period_path = datapath + 'summary/activity_period_9999.csv'

    df_all_p = load_raw_data()
    df_day = copy_one_day(df_all_p)
    export_copied_data(df_day, cleaned_data_path)

    df_all_p_sorted = predict_label(predicted_data_path)
    export_predicted_data(df_all_p_sorted, predicted_data_path)


    # # Analyze Predicted Results


    df_summary_all, df_act_period = get_summarized_data(predicted_data_path)
    export_summarized_data(df_summary_all, df_act_period, all_day_summary_path, act_period_path)

    return df_all_p_sorted, df_summary_all, df_act_period


# # Insert to Database

# In[37]:

def insert_to_database(mydb, mycursor, df_all_p_sorted, df_summary_all, df_act_period):

    sql = "INSERT INTO Patient (ID, Dateandtime, X, Y, Z, HR, Label) VALUES (%s, %s, %s, %s, %s, %s, %s)"

    for row in zip(df_all_p_sorted['ID'],
                df_all_p_sorted['timestamp'],
                df_all_p_sorted['x'],
                df_all_p_sorted['y'],
                df_all_p_sorted['z'],
                df_all_p_sorted['HR'],
                df_all_p_sorted['y_pred']):

        mycursor.execute(sql, row)

    mydb.commit()

    sql = "INSERT INTO AllDaySummary (ID, Date, TimeFrom, TimeUntil, ActualFrom, ActualUntil,    DurationSit, DurationSleep, DurationStand, DurationWalk, TotalDuration,    CountSit, CountSleep, CountStand, CountWalk,    CountActive, CountInactive,    CountTotalActiveness, CountTransition, DurationPerTransition)    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

    for row in zip(df_summary_all['ID'],
                df_summary_all['date'],
                df_summary_all['from'],
                df_summary_all['to'],
                df_summary_all['from actual'],
                df_summary_all['to actual'],
                df_summary_all['sit'],
                df_summary_all['sleep'],
                df_summary_all['stand'],
                df_summary_all['walk'],
                df_summary_all['total'],
                df_summary_all['sit count'],
                df_summary_all['sleep count'],
                df_summary_all['stand count'],
                df_summary_all['walk count'],
                df_summary_all['inactive count'],
                df_summary_all['active count'],
                df_summary_all['total count'],
                df_summary_all['transition count'],
                df_summary_all['duration per action']):

        mycursor.execute(sql, row)

    mydb.commit()


    sql = "INSERT INTO ActivityPeriod (ID, Date, TimeFrom, TimeUntil, Label)    VALUES (%s, %s, %s, %s, %s)"

    for row in zip(df_act_period['ID'],
                df_act_period['date'],
                df_act_period['from'],
                df_act_period['to'],
                df_act_period['y_pred']):

        mycursor.execute(sql, row)

    mydb.commit()

def main_function():
    mydb, mycursor = connect_to_database()
    df_all_p_sorted, df_summary_all, df_act_period = get_all_patients_result()
    insert_to_database(mydb, mycursor, df_all_p_sorted, df_summary_all, df_act_period)

# print(df_summary_all)

if(__name__=='__main__'):
    main_function()
