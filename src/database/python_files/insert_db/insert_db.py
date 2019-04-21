import mysql.connector
import numpy as np
import pandas as pd
import csv
import sys

from datetime import datetime, timedelta
from mysql.connector import FieldType
from sqlalchemy import create_engine

status_started = 0
status_stopped = 1
status_error = -1
status_sleep = -2

on_server = int(sys.argv[1])
at_home = ''

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

def get_sql_connection():
    user = 'php'
    passwd = 'HOD8912+php'
    host = 'localhost'
    port = '3306'
    schema = 'cu_amd'
    engine = create_engine('mysql+mysqlconnector://{}:{}@{}:{}/{}'.format(user, passwd, host, port, schema), echo=False)

    return engine

def reset_status():
    load_status = status_sleep
    predict_status = status_sleep
    summarize_status = status_sleep
    return [load_status, predict_status, summarize_status]

def set_safe_updates(enable):
    mydb, mycursor = connect_to_database()

    if(enable):
        sql = "SET SQL_SAFE_UPDATES = 1;"
    else:
        sql = "SET SQL_SAFE_UPDATES = 0;"
    
    mycursor.execute(sql)
    mydb.commit()

def insert_db_status(process_name, start_time, stop_time, status):
    mydb, mycursor = connect_to_database()
    set_safe_updates(False)

    datetime_format = '%Y-%m-%d %H:%i:%S.%f'

    insert_sql = "INSERT INTO cu_amd.Logging (StartTime, StopTime, ProcessName, ProcessStatus) VALUES (DATE_FORMAT(%s, %s), %s, %s, %s)"
    update_sql = "UPDATE cu_amd.Logging SET ProcessStatus = %s WHERE StartTime = %s AND ProcessName = %s"
    stoptime_sql = "UPDATE cu_amd.Logging SET StopTime = %s WHERE StartTime = %s AND ProcessName = %s"
    
    error = False

    print(process_name, status)

    # try:
    if(status==status_started):
        sql = insert_sql
        values = (start_time, datetime_format, stop_time, process_name, status)

    elif(status==status_stopped or status==status_error):
        sql = update_sql
        values = (status, start_time, process_name)

        values2 = (stop_time, start_time, process_name)

    # except:
    #     sql = update_sql
    #     values = (status_error, start_time, process_name)
    #     error = True
    
    print(sql)
    print('values:', values)
    
    mycursor.execute(sql, values)
    
    if(status==status_error or status==status_stopped or error):
        mycursor.execute(update_sql, values)
        print('values:', values)

        mycursor.execute(stoptime_sql, values2)
        print('values2:', values2)

    mydb.commit()

    set_safe_updates(True)
    print('inserted status')

def insert_db_patient(df_all_p_sorted):
    cnx = get_sql_connection()
    
    df_all_p_sorted = df_all_p_sorted.rename(columns={
        'timestamp': 'DateAndTime',
        'x': 'X',
        'y': 'Y',
        'z': 'Z',
        'AI': 'ActivityIndex',
        'y_pred': 'Label'
    })
    df_all_p_sorted.to_sql('Patient', cnx, schema='cu_amd', if_exists='append', index=False, index_label=None, chunksize=100, dtype=None)


def insert_db_all_day_summary(df_summary_all):
    if(df_summary_all.empty):
        print('df summary all is empty')
        return

    cols = ['ID', 'date', 'from', 'to', 'from actual', 'to actual', 
            'sit', 'sleep', 'stand', 'walk', 'total', 
            'sit count', 'sleep count', 'stand count', 'walk count', 
            'inactive count', 'active count', 'total count', 
            'transition count', 'duration per action']

    cnx = get_sql_connection()
    df_summary_all = df_summary_all[cols]
    df_summary_all = df_summary_all.rename(columns={
        'date': 'Date',
        'from': 'TimeFrom',
        'to': 'TimeUntil',
        'from actual': 'ActualFrom',
        'to actual': 'ActualUntil',
        'sit': 'DurationSit',
        'sleep': 'DurationSleep',
        'stand': 'DurationStand',
        'walk': 'DurationWalk',
        'total': 'TotalDuration',
        'sit count': 'CountSit',
        'sleep count': 'CountSleep',
        'stand count': 'CountStand',
        'walk count': 'CountWalk',
        'inactive count': 'CountInactive',
        'active count': 'CountActive',
        'total count': 'CountTotalActiveness',
        'transition count': 'CountTransition',
        'duration per action': 'DurationPerTransition'
    })
    
    df_summary_all.to_sql('AllDaySummary', cnx, schema='cu_amd', if_exists='replace', index=False, index_label=None, chunksize=100, dtype=None)


def insert_db_act_period(df_act_period):
    if(df_act_period.empty):
        print('df activity period is empty')
        return

    cols = ['ID', 'date', 'from', 'to', 'y_pred']
    
    cnx = get_sql_connection()
    df_act_period = df_act_period[cols]
    df_act_period = df_act_period.rename(columns={
        'date': 'Date',
        'from': 'TimeFrom',
        'to': 'TimeUntil',
        'y_pred': 'Label'
    })
    df_act_period.to_sql('ActivityPeriod', cnx, schema='cu_amd', if_exists='replace', index=False, index_label=None, chunksize=100, dtype=None)

def get_patients_acc_hr(all_patients, date_to_retrieve):
    mydb, mycursor = connect_to_database()

    df_acc = pd.DataFrame()
    df_hr = pd.DataFrame()

    for p in all_patients:
        date_format = '%Y-%m-%d'
        next_date = (datetime.strptime(date_to_retrieve, date_format) + timedelta(days=1)).strftime(date_format)

        sql = "SELECT * FROM cu_amd.acc_log_2 where user_id='{}' \
        and (event_timestamp > DATE_FORMAT('{}', '%Y-%m-%d')) \
        and (event_timestamp < DATE_FORMAT('{}', '%Y-%m-%d'));".format(p, date_to_retrieve, next_date)
        print(sql)

        mycursor.execute(sql)
        records = mycursor.fetchall()

        print(p, date_to_retrieve)
        print('length: ', mycursor.rowcount)

        if(mycursor.rowcount==0):
            continue

        xyz = []
        timestamp = []
        user_id = []

        for row in records:
            xyz.append([row[1], row[2], row[3]])
            timestamp.append(row[4])
            user_id.append(row[6])
        xyz = np.array(xyz)
        print(xyz.shape)

        df_i = pd.DataFrame({'ID': user_id,
                            'timestamp': timestamp,
                            'x': xyz.transpose()[0],
                            'y': xyz.transpose()[1],
                            'z': xyz.transpose()[2]})

        df_acc = df_acc.append(df_i)

        sql2 = "SELECT * FROM cu_amd.hr_log where user_id='{}' \
        and (event_timestamp > DATE_FORMAT('{}', '%Y-%m-%d')) \
        and (event_timestamp < DATE_FORMAT('{}', '%Y-%m-%d'));".format(p, date_to_retrieve, next_date)

        mycursor.execute(sql2)
        records = mycursor.fetchall()
        print('length: ', mycursor.rowcount)

        hr = []
        timestamp = []
        user_id = []

        for row in records:
            hr.append(row[1])
            timestamp.append(row[2])
            user_id.append(row[4])

        df_hr_i = pd.DataFrame({'ID': user_id,
                            'timestamp': timestamp,
                            'hr': hr})

        df_hr = df_hr.append(df_hr_i)

    return df_acc, df_hr

def select_from_logging():
    mydb, mycursor = connect_to_database()
    sql = "SELECT * FROM Logging;"

    mycursor.execute(sql)
    records = mycursor.fetchall()

    print(sql)
    for r in records:
        print(r)