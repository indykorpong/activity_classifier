import mysql.connector
import numpy as np
import pandas as pd
import csv
import sys

from datetime import datetime, timedelta
from sqlalchemy import create_engine

status_started = 0
status_stopped = 1
status_error = -1

# # Connect to MySQL Database

def connect_to_database():
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
    load_status = status_started
    predict_status = status_started
    summarize_status = status_started
    return [load_status, predict_status, summarize_status]

def set_safe_updates(enable):
    mydb, mycursor = connect_to_database()

    if(enable):
        sql = "SET SQL_SAFE_UPDATES = 1;"
    else:
        sql = "SET SQL_SAFE_UPDATES = 0;"
    
    mycursor.execute(sql)
    mydb.commit()

# StartTime, EndTime, UserID, ProcessName, StartingData, EndingData, ProcessStatus
def insert_db_status(start_time, end_time, user_id, process_name, starting_data, ending_data, status):
    mydb, mycursor = connect_to_database()
    set_safe_updates(False)

    datetime_format = '%Y-%m-%d %H:%i:%S.%f'

    insert_sql = "INSERT INTO cu_amd.AuditLog (StartTime, EndTime, UserID, ProcessName, ProcessStatus) VALUES (DATE_FORMAT(%s, %s), %s, %s, %s, %s)"
    update_sql = "UPDATE cu_amd.AuditLog SET ProcessStatus = %s WHERE StartTime = %s AND ProcessName = %s"
    stoptime_sql = "UPDATE cu_amd.AuditLog SET EndTime = %s WHERE StartTime = %s AND ProcessName = %s"
    start_data_sql = "UPDATE cu_amd.AuditLog SET StartingData=DATE_FORMAT(%s, %s) WHERE UserID=%s AND StartTime=%s AND ProcessName=%s;"
    end_data_sql = "UPDATE cu_amd.AuditLog SET EndingData=DATE_FORMAT(%s, %s) WHERE UserID=%s AND StartTime=%s AND ProcessName=%s;"
    
    error = False

    print(process_name, status)

    # try:
    if(status==status_started):
        sql = insert_sql
        values = (start_time, datetime_format, end_time, user_id, process_name, status)

    if(status==status_stopped or status==status_error):
        sql = update_sql
        values = (status, start_time, process_name)
        values2 = (end_time, start_time, process_name)

    if(status==status_stopped):
        start_data_values = (starting_data, datetime_format, user_id, start_time, process_name)
        end_data_values = (ending_data, datetime_format, user_id, start_time, process_name)

        mycursor.execute(start_data_sql, start_data_values)
        mycursor.execute(end_data_sql, end_data_values)

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

def insert_db_act_log(df_all_p_sorted):
    cnx = get_sql_connection()
    
    df_all_p_sorted = df_all_p_sorted.rename(columns={
        'timestamp': 'DateAndTime',
        'x': 'X',
        'y': 'Y',
        'z': 'Z',
        'AI': 'ActivityIndex',
        'y_pred': 'Label'
    })
    df_all_p_sorted.to_sql('ActivityLog', cnx, schema='cu_amd', if_exists='append', index=False, index_label=None, chunksize=100, dtype=None)


def insert_db_hourly_summary(df_summary_all):
    if(df_summary_all.empty):
        print('df summary all is empty')
        return

    cnx = get_sql_connection()

    time_cols = ['TimeFrom', 'TimeUntil', 'ActualFrom', 'ActualUntil', 
        'DurationSit', 'DurationSleep', 'DurationStand', 'DurationWalk', 'TotalDuration', 
        'DurationPerAction']

    time_format = '%H:%M:%S.%f'

    # print('check time cols', df_summary_all.head())

    for c in time_cols:
        df_summary_all[c] = df_summary_all[c].apply(lambda x: x.strftime(time_format))

    print(df_summary_all.dtypes)
    
    df_summary_all.to_sql('HourlyActivitySummary', cnx, schema='cu_amd', if_exists='append', index=False, index_label=None, chunksize=100, dtype=None)


def insert_db_act_period(df_act_period):
    if(df_act_period.empty):
        print('df activity period is empty')
        return

    cnx = get_sql_connection()

    df_act_period.to_sql('ActivityPeriod', cnx, schema='cu_amd', if_exists='append', index=False, index_label=None, chunksize=100, dtype=None)

def get_patients_acc_hr(user_id):
    mydb, mycursor = connect_to_database()

    df_acc = pd.DataFrame()
    df_hr = pd.DataFrame()

    # date_format = '%Y-%m-%d'

    sql = "SELECT * FROM cu_amd.acc_log_2 WHERE loaded_flag IS NULL and user_id='{}';".format(user_id)

    mycursor.execute(sql)
    records = mycursor.fetchall()
    print('length: ', mycursor.rowcount)

    if(mycursor.rowcount!=0):
        xyz = []
        timestamp = []
        user_ids = []

        for row in records:
            xyz.append([row[1], row[2], row[3]])
            timestamp.append(row[0])
            user_ids.append(row[4])
        xyz = np.array(xyz)
        print(xyz.shape)

        df_i = pd.DataFrame({'UserID': user_ids,
                            'timestamp': timestamp,
                            'x': xyz.transpose()[0],
                            'y': xyz.transpose()[1],
                            'z': xyz.transpose()[2]})

        df_acc = df_acc.append(df_i)

        sql2 = "SELECT * FROM cu_amd.hr_log_2 WHERE loaded_flag IS NULL and user_id='{}';".format(user_id)

        mycursor.execute(sql2)
        records = mycursor.fetchall()
        print('length: ', mycursor.rowcount)

        hr = []
        timestamp = []
        user_ids = []

        for row in records:
            hr.append(row[1])
            timestamp.append(row[0])
            user_ids.append(row[2])

        print(len(user_ids))

        df_hr_i = pd.DataFrame({'UserID': user_ids,
                            'timestamp': timestamp,
                            'hr': hr})

        df_hr = df_hr.append(df_hr_i)

        # update the values of loaded_flag to 1
        sql3 = "UPDATE cu_amd.acc_log_2 SET loaded_flag=TRUE WHERE loaded_flag IS NULL and user_id={};".format(user_id)
        sql4 = "UPDATE cu_amd.hr_log_2 SET loaded_flag=TRUE WHERE loaded_flag IS NULL and user_id={};".format(user_id)

        mycursor.execute(sql3)
        mycursor.execute(sql4)
        mydb.commit()

    return df_acc, df_hr

def get_distinct_user_ids():
    mydb, mycursor = connect_to_database()

    sql = "SELECT DISTINCT user_id from acc_log_2;"

    mycursor.execute(sql)
    records = mycursor.fetchall()

    return list([r[0] for r in records])

def update_summarized_flag(user_id):
    mydb, mycursor = connect_to_database()
    
    sql = "UPDATE cu_amd.ActivityLog SET SummarizedFlag=TRUE WHERE SummarizedFlag IS NULL and user_id={};".format(user_id)

    mycursor.execute(sql)
    mydb.commit()