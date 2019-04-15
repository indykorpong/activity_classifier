import mysql.connector
import numpy as np
import pandas as pd
import csv
import sys

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

    update_sql = "UPDATE Logging SET ProcessStatus = %s WHERE StartTime = DATE_FORMAT(%s, '%y-%m-%d %H:%i:%S.%f') AND ProcessName = %s"
    insert_sql = "INSERT INTO Logging (StartTime, StopTime, ProcessName, ProcessStatus) VALUES (DATE_FORMAT(%s, '%y-%m-%d %H:%i:%S.%f'), %s, %s, %s)"
    error = False

    print(process_name, status)

    try:
        if(status==status_stopped or status_error):
            sql = update_sql

            values = (stop_time, start_time, process_name)
            values2 = (status, start_time, process_name)
        
        elif(status==status_started):
            sql = insert_sql
            values = (start_time, stop_time, process_name, status)

    except:
        sql = update_sql
        
        values = (stop_time, start_time, process_name)
        values2 = (status_error, start_time, process_name)
        error = True
    
    mycursor.execute(sql, values)
    
    if(status==status_error or status==status_stopped or error):
        mycursor.execute(update_sql, values2)

    mydb.commit()

    set_safe_updates(True)

def insert_db_patient(df_all_p_sorted):
    # mydb, mycursor = connect_to_database()
    # sql = "INSERT INTO Patient (ID, DateAndTime, X, Y, Z, HR, ActivityIndex, Label) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"

    # for row in zip(df_all_p_sorted['ID'],
    #             df_all_p_sorted['timestamp'],
    #             df_all_p_sorted['x'],
    #             df_all_p_sorted['y'],
    #             df_all_p_sorted['z'],
    #             df_all_p_sorted['HR'],
    #             df_all_p_sorted['AI'],
    #             df_all_p_sorted['y_pred']):

    #     mycursor.execute(sql, row)

    # mydb.commit()

    cnx = get_sql_connection()
    
    df_all_p_sorted = df_all_p_sorted.rename(columns={
        'timestamp': 'DateAndTime',
        'x': 'X',
        'y': 'Y',
        'z': 'Z',
        'AI': 'ActivityIndex',
        'y_pred': 'Label'
    })
    df_all_p_sorted.to_sql('Patient', cnx, schema='cu_amd', if_exists='replace', index=False, index_label=None, chunksize=100, dtype=None)


def insert_db_all_day_summary(df_summary_all):
    mydb, mycursor = connect_to_database()
    sql = "INSERT INTO AllDaySummary (ID, Date, TimeFrom, TimeUntil, ActualFrom, ActualUntil,    DurationSit, DurationSleep, DurationStand, DurationWalk, TotalDuration,    CountSit, CountSleep, CountStand, CountWalk,    CountInactive, CountActive,    CountTotalActiveness, CountTransition, DurationPerTransition)    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

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

def insert_db_act_period(df_act_period):
    mydb, mycursor = connect_to_database()

    sql = "INSERT INTO ActivityPeriod (ID, Date, TimeFrom, TimeUntil, Label)    VALUES (%s, %s, %s, %s, %s)"

    for row in zip(df_act_period['ID'],
                df_act_period['date'],
                df_act_period['from'],
                df_act_period['to'],
                df_act_period['y_pred']):

        mycursor.execute(sql, row)

    mydb.commit()

def get_patients_acc_hr(all_patients, date_to_retrieve):
    mydb, mycursor = connect_to_database()

    df_acc = pd.DataFrame()
    df_hr = pd.DataFrame()

    table = 'acc_log_2'
    # table = 'accelerometer_log'

    for p in all_patients:
        sql = "SELECT * FROM cu_amd.{} where user_id='{}' and (event_timestamp > DATE_FORMAT('{}', '%y-%m-%d'));".format(table, p, date_to_retrieve)
        print(sql)
        mycursor.execute(sql)
        records = mycursor.fetchall()

        print(p, date_to_retrieve)
        print('length: ', mycursor.rowcount)

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

        sql2 = "SELECT * FROM cu_amd.hr_log where user_id='{}' and (event_timestamp > DATE_FORMAT('{}', '%y-%m-%d'));".format(p, date_to_retrieve)

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

    # mydb.commit()

    return df_acc, df_hr

