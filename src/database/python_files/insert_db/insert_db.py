import mysql.connector
import numpy as np
import pandas as pd
import csv
import sys

from datetime import datetime, timedelta
from sqlalchemy import create_engine
from mysql.connector.errors import IntegrityError

status_started = 0
status_stopped = 1
status_error = -1

# # Connect to MySQL Database

def connect_to_database():
    user = 'php'
    passwd = 'HOD8912+php'
    # user = 'root'
    # passwd = "1amdjvr'LN"

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
    # user = 'root'
    # passwd = "1amdjvr'LN"

    host = 'localhost'
    port = '3306'
    schema = 'cu_amd'
    engine = create_engine('mysql+mysqlconnector://{}:{}@{}:{}/{}'.format(user, passwd, host, port, schema), echo=False)

    return engine

def clear_table():
    mydb, mycursor = connect_to_database()

    sql = ["insert into acc_log_2 (event_timestamp, x, y, z, user_id) \
        select distinct event_timestamp, x, y, z, user_id from accelerometer_log \
        where event_timestamp > \
        (SELECT max(event_timestamp) as max_timestamp \
        FROM acc_log_2);"]

    for s in sql:
        mycursor.execute(s)

    mydb.commit()

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

    try:
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

    except:
        sql = update_sql
        values = (status_error, start_time, process_name)
        error = True
    
    mycursor.execute(sql, values)
    
    if(status==status_error or status==status_stopped or error):
        mycursor.execute(update_sql, values)

        mycursor.execute(stoptime_sql, values2)

    mydb.commit()

    set_safe_updates(True)
    print('inserted status')

def insert_db_act_log(df_all_p, update=False):
    # cnx = get_sql_connection()
    # df_all_p.to_sql('ActivityLog', cnx, schema='cu_amd', if_exists='append', index=False, index_label=None, chunksize=100, dtype=None)

    mydb, mycursor = connect_to_database()

    datetime_format = '%Y-%m-%d %H:%M:%S.%f'

    df_all_p['timestamp'] = df_all_p['timestamp'].apply(lambda x: x.strftime(datetime_format))
    df_all_p = df_all_p.rename(columns={
        'timestamp': 'DateAndTime',
        'x': 'X',
        'y': 'Y',
        'z': 'Z',
        'AI': 'ActivityIndex',
        'y_pred': 'Label'
    })

    if(not update):
        sql = "INSERT INTO ActivityLog (UserID, DateAndTime, X, Y, Z, HR) \
            VALUES (%s, %s, %s, %s, %s, %s)"

        values = zip(df_all_p['UserID'], df_all_p['DateAndTime'], \
            df_all_p['X'], df_all_p['Y'], df_all_p['Z'], df_all_p['HR'])

    else:
        sql = "INSERT INTO ActivityLog (UserID, DateAndTime) \
            VALUES (%s, %s) \
            ON DUPLICATE KEY UPDATE \
            X=%s, Y=%s, Z=%s, HR=%s, \
            ActivityIndex=%s, Label=%s;"

        values = zip(df_all_p['UserID'], df_all_p['DateAndTime'], \
            df_all_p['X'], df_all_p['Y'], df_all_p['Z'], df_all_p['HR'], \
            df_all_p['ActivityIndex'], df_all_p['Label'])

        print('update')
    
    for z in values:
        mycursor.execute(sql, z)
    mydb.commit()

def insert_db_hourly_summary(df_summary_all):
    if(df_summary_all.empty):
        print('df hourly summary is empty')
        return
    
    mydb, mycursor = connect_to_database()

    date_format = '%Y-%m-%d'
    df_summary_all['Date'] = df_summary_all['Date'].apply(lambda x: x.strftime(date_format))

    time_cols = ['TimeFrom', 'TimeUntil', 'ActualFrom', 'ActualUntil', 
        'DurationSit', 'DurationSleep', 'DurationStand', 'DurationWalk', 'TotalDuration', 
        'DurationPerAction']

    time_format = '%H:%M:%S.%f'

    for c in time_cols:
        df_summary_all[c] = df_summary_all[c].apply(lambda x: x.strftime(time_format))

    sql = "INSERT INTO HourlyActivitySummary (UserID, Date, TimeFrom, TimeUntil, \
        ActualFrom, ActualUntil, DurationSit, DurationSleep, DurationStand, DurationWalk, TotalDuration, \
        CountSit, CountSleep, CountStand, CountWalk, \
        CountInactive, CountActive, CountTotal, \
        CountActiveToInactive,DurationPerAction) \
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"

    sql2 = "UPDATE cu_amd.HourlyActivitySummary \
        SET \
            ActualFrom=%s, \
            ActualUntil=%s, \
            DurationSit=%s, \
            DurationSleep=%s, \
            DurationStand=%s, \
            DurationWalk=%s, \
            TotalDuration=%s, \
            CountSit=%s, \
            CountSleep=%s, \
            CountStand=%s, \
            CountWalk=%s, \
            CountInactive=%s, \
            CountActive=%s, \
            CountTotal=%s, \
            CountActiveToInactive=%s, \
            DurationPerAction=%s \
        WHERE UserID=%s AND Date=%s \
        AND TimeFrom=%s \
        AND TimeUntil=%s \
        AND ActualFrom=%s \
        AND ActualUntil=%s;"

    values = zip(df_summary_all['UserID'], df_summary_all['Date'], \
        df_summary_all['TimeFrom'], df_summary_all['TimeUntil'], df_summary_all['ActualFrom'], df_summary_all['ActualUntil'], \
        df_summary_all['DurationSit'], df_summary_all['DurationSleep'], df_summary_all['DurationStand'], df_summary_all['DurationWalk'], \
        df_summary_all['TotalDuration'], df_summary_all['CountSit'], df_summary_all['CountSleep'], df_summary_all['CountStand'], df_summary_all['CountWalk'], \
        df_summary_all['CountInactive'], df_summary_all['CountActive'], df_summary_all['CountTotal'], \
        df_summary_all['CountActiveToInactive'], df_summary_all['DurationPerAction'])

    for z in values:
        print(z)
        try:
            print('insert')
            mycursor.execute(sql, z)
        except IntegrityError:
            print('update')
            mycursor.execute(sql2, z)


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

def get_unpredicted_data(user_id):
    cnx = get_sql_connection()

    sql = "SELECT UserID, DateAndTime, X, Y, Z, HR, Label FROM ActivityLog WHERE Label IS NULL and UserID={};".format(user_id)

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

def get_unsummarized_data(user_id):
    cnx = get_sql_connection()

    sql = "SELECT UserID, DateAndTime, X, Y, Z, HR, Label FROM ActivityLog WHERE SummarizedFlag IS NULL and UserID={};".format(user_id)

    df_to_summarize = pd.read_sql(sql, cnx)
    df_to_summarize = df_to_summarize.rename(columns={
        'DateAndTime': 'timestamp',
        'X': 'x',
        'Y': 'y',
        'Z': 'z',
        'ActivityIndex': 'AI',
        'Label': 'y_pred'
    })

    return df_to_summarize

def get_distinct_user_ids():
    mydb, mycursor = connect_to_database()

    sql = "SELECT DISTINCT user_id from acc_log_2;"

    mycursor.execute(sql)
    records = mycursor.fetchall()

    return list([r[0] for r in records])

def update_summarized_flag(user_id):
    mydb, mycursor = connect_to_database()
    
    sql = "UPDATE cu_amd.ActivityLog SET SummarizedFlag=TRUE WHERE SummarizedFlag IS NULL and UserID={};".format(user_id)

    mycursor.execute(sql)
    mydb.commit()

def get_user_profile(user_id):
    mydb, mycursor = connect_to_database()

    get_user_profile_sql = "SELECT MinX, MinY, MinZ, MaxX, MaxY, MaxZ FROM UserProfile WHERE UserID={};".format(user_id)

    mycursor.execute(get_user_profile_sql)
    user_profile = mycursor.fetchall()

    if(len(user_profile)==0):
        get_min_max_sql = "SELECT MIN(x), MIN(y), MIN(z), MAX(x), MAX(y), MAX(z) FROM acc_log_2 WHERE user_id={};".format(user_id)
        
        mycursor.execute(get_min_max_sql)
        user_profile = mycursor.fetchall()

        insert_user_profile_sql = "INSERT INTO UserProfile (UserID, MinX, MinY, MinZ, MaxX, MaxY, MaxZ) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        value_list = np.hstack(([user_id],list(user_profile[0])))
        values = tuple(value_list)

        mycursor.execute(insert_user_profile_sql, values)
        mydb.commit()

        mycursor.execute(get_user_profile_sql)
        user_profile = mycursor.fetchall()

    return user_profile