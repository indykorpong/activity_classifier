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

## Connect to MySQL database using mysql connector
class Datetime64Converter(mysql.connector.conversion.MySQLConverter):
    """ A mysql.connector Converter that handles datetime64 types """

    def _timestamp_to_mysql(self, value):
        return value.strftime('%Y-%m-%d %H:%M:%S.%f').encode('ascii')

def connect_to_database():
    user = 'php'
    passwd = 'HOD8912+php'

    mydb = mysql.connector.connect(
        host='localhost',
        user=user,
        passwd=passwd,
        database='cu_amd'
        )

    print('connected to database')
    mydb.set_converter_class(Datetime64Converter)
    mycursor = mydb.cursor(buffered=True)

    return mydb, mycursor

## Connect to MySQL database using sqlalchemy

def get_sql_connection():
    user = 'php'
    passwd = 'HOD8912+php'

    host = 'localhost'
    port = '3306'
    schema = 'cu_amd'
    engine = create_engine('mysql+mysqlconnector://{}:{}@{}:{}/{}'.format(user, passwd, host, port, schema), echo=False)

    return engine

## Create temporary tables (acc_log_2, hr_log_2) from original tables (accelerometer_log, hr_log) 

def create_temp_table():
    mydb, mycursor = connect_to_database()

    sql = ["insert into acc_log_2 (event_timestamp, x, y, z, user_id) \
        select distinct event_timestamp, x, y, z, user_id from accelerometer_log \
        where event_timestamp > \
        (SELECT max(event_timestamp) as max_timestamp \
        FROM acc_log_2);",
        "insert into hr_log_2 (event_timestamp, HR, user_id) \
        select distinct event_timestamp, hr, user_id from hr_log \
        where event_timestamp > \
        (SELECT max(event_timestamp) as max_timestamp \
        FROM hr_log_2);"]

    for s in sql:
        mycursor.execute(s)

    mydb.commit()

# Set all process status to starting status(0)

def reset_status():
    load_status = status_started
    predict_status = status_started
    summarize_status = status_started
    return [load_status, predict_status, summarize_status]

## Set sql safe updates parameter to True or False

def set_safe_updates(enable):
    mydb, mycursor = connect_to_database()

    if(enable):
        sql = "SET SQL_SAFE_UPDATES = 1;"
    else:
        sql = "SET SQL_SAFE_UPDATES = 0;"
    
    mycursor.execute(sql)
    mydb.commit()

## Set or insert process status in AuditLog table in database

# StartTime, EndTime, UserID, ProcessName, StartingData, EndingData, ProcessStatus
def insert_db_status(start_time, end_time, user_id, process_name, starting_data, ending_data, status):
    mydb, mycursor = connect_to_database()
    set_safe_updates(False)

    datetime_format = '%Y-%m-%d %H:%i:%S.%f'

    insert_sql = "INSERT INTO cu_amd.AuditLog (StartProcessTime, EndProcessTime, UserID, ProcessName, ProcessStatus) VALUES (DATE_FORMAT(%s, %s), %s, %s, %s, %s)"
    update_sql = "UPDATE cu_amd.AuditLog SET ProcessStatus = %s WHERE StartProcessTime = %s AND ProcessName = %s"
    stoptime_sql = "UPDATE cu_amd.AuditLog SET EndProcessTime = %s WHERE StartProcessTime = %s AND ProcessName = %s"
    start_data_sql = "UPDATE cu_amd.AuditLog SET StartData=DATE_FORMAT(%s, %s) WHERE UserID=%s AND StartProcessTime=%s AND ProcessName=%s;"
    end_data_sql = "UPDATE cu_amd.AuditLog SET EndData=DATE_FORMAT(%s, %s) WHERE UserID=%s AND StartProcessTime=%s AND ProcessName=%s;"
    
    error = False

    print(process_name, status)

    # Set the query string and values to insert into AuditLog table
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
        # sql = update_sql
        # values = (status_error, start_time, process_name)
        # error = True
    
    # Try executing the sql query with values. If fails, set status to error status
    try:
        mycursor.execute(sql, values)
    except:
        status = status_error        

    # If the status is stopping status or error status, also execute the UPDATE ProcessStatus query
    if(status==status_error or status==status_stopped):
        mycursor.execute(update_sql, values)
        mycursor.execute(stoptime_sql, values2)

    mydb.commit()

    set_safe_updates(True)
    print('inserted status')

## Insert the patient's preprocessed or predicted data into ActivityLog table in database
def insert_db_act_log(df_all_p, update=False):
    mydb, mycursor = connect_to_database()

    # datetime_format = '%Y-%m-%d %H:%M:%S.%f'

    # Convert timestamp column from datetime format to string format
    # df_all_p['timestamp'] = df_all_p['timestamp'].apply(lambda x: x + timedelt)
    df_all_p = df_all_p.rename(columns={
        'timestamp': 'DateAndTime',
        'x': 'X',
        'y': 'Y',
        'z': 'Z',
        'AI': 'ActivityIndex',
        'y_pred': 'Label'
    })
    # df_all_p = df_all_p.drop(columns=['timestamp'])
    print('df act log:', df_all_p.head(3))
    print('df act log:', df_all_p.tail(3))
    print(df_all_p['DateAndTime'].dtypes)

    if(not update):
        # df_all_p.to_sql('ActivityLog', con=cnx, schema='cu_amd', if_exists='append', index=False, index_label=None, chunksize=10000)

        sql = "INSERT INTO ActivityLog (UserID, DateAndTime, X, Y, Z, HR) \
            VALUES (%s, %s, %s, %s, %s, %s)"

        values = zip(df_all_p['UserID'], df_all_p['DateAndTime'], \
            df_all_p['X'], df_all_p['Y'], df_all_p['Z'], df_all_p['HR'])

        for val in values:
            mycursor.execute(sql, val)

    # If True, update the values of ActivityIndex and Label
    else:
        values = zip(df_all_p['UserID'], df_all_p['DateAndTime'], \
            df_all_p['X'], df_all_p['Y'], df_all_p['Z'], df_all_p['HR'], \
            df_all_p['ActivityIndex'], df_all_p['Label'])

        for val in values:
            u_id = val[0]; dt = val[1]; x_i = val[2]; y_i = val[3]; z_i = val[4]
            hr_i = val[5]; ai_i = val[6]; label_i = val[7]
            
            ai_sql = "UPDATE ActivityLog SET ActivityIndex={} WHERE UserID={} and DateAndTime='{}';".format(ai_i, u_id, dt)
            label_sql = "UPDATE ActivityLog SET Label={} WHERE UserID={} and DateAndTime='{}';".format(label_i, u_id, dt)

            mycursor.execute(ai_sql)
            mycursor.execute(label_sql)

        print('update activity log')
    
    mydb.commit()

## Insert the hourly activity summary of the patient into HourlyActivitySummary table in database

def insert_db_hourly_summary(df_summary_all):
    # If there is no summary or the dataframe is empty, do nothing
    if(df_summary_all.empty):
        print('df hourly summary is empty')
        return
    
    mydb, mycursor = connect_to_database()

    # Convert Date column from datetime format to string format
    date_format = '%Y-%m-%d'
    df_summary_all['Date'] = df_summary_all['Date'].apply(lambda x: x.strftime(date_format))

    time_cols = ['TimeFrom', 'TimeUntil', 'ActualFrom', 'ActualUntil', 
        'DurationSit', 'DurationSleep', 'DurationStand', 'DurationWalk', 'TotalDuration', 
        'DurationPerAction']

    time_format = '%H:%M:%S.%f'

    for c in time_cols:
        df_summary_all[c] = df_summary_all[c].apply(lambda x: x.strftime(time_format))

    insert_sql = "INSERT INTO HourlyActivitySummary (UserID, Date, TimeFrom, TimeUntil, \
        ActualFrom, ActualUntil, DurationSit, DurationSleep, DurationStand, DurationWalk, TotalDuration, \
        CountSit, CountSleep, CountStand, CountWalk, \
        CountInactive, CountActive, CountTotal, \
        CountActiveToInactive,DurationPerAction) \
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, \
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"

    update_sql = "SET @NewActualFrom := time(%s); \
            SET @OldActualFrom := (select ActualFrom from cu_amd.HourlyActivitySummary where TimeFrom=%s and UserID=%s and Date=%s); \
            \
            SET @NewActualUntil := time(%s); \
            SET @OldActualUntil := (select ActualUntil from cu_amd.HourlyActivitySummary where TimeFrom=%s and UserID=%s and Date=%s); \
            \
            SET @NewDurationSit := time(%s); \
            SET @OldDurationSit := (select DurationSit from cu_amd.HourlyActivitySummary where TimeFrom=%s and UserID=%s and Date=%s); \
            \
            SET @NewDurationSleep := time(%s); \
            SET @OldDurationSleep := (select DurationSleep from cu_amd.HourlyActivitySummary where TimeFrom=%s and UserID=%s and Date=%s); \
            \
            SET @NewDurationStand := time(%s); \
            SET @OldDurationStand := (select DurationStand from cu_amd.HourlyActivitySummary where TimeFrom=%s and UserID=%s and Date=%s); \
            \
            SET @NewDurationWalk := time(%s); \
            SET @OldDurationWalk := (select DurationWalk from cu_amd.HourlyActivitySummary where TimeFrom=%s and UserID=%s and Date=%s); \
            \
            SET @NewTotalDuration := time(%s); \
            SET @OldTotalDuration := (select TotalDuration from cu_amd.HourlyActivitySummary where TimeFrom=%s and UserID=%s and Date=%s); \
            \
            SET @NewCountSit = %s; \
            SET @OldCountSit = (select CountSit from cu_amd.HourlyActivitySummary where TimeFrom=%s and UserID=%s and Date=%s); \
            \
            SET @NewCountSleep = %s; \
            SET @OldCountSleep = (select CountSleep from cu_amd.HourlyActivitySummary where TimeFrom=%s and UserID=%s and Date=%s); \
            \
            SET @NewCountStand = %s; \
            SET @OldCountStand = (select CountStand from cu_amd.HourlyActivitySummary where TimeFrom=%s and UserID=%s and Date=%s); \
            \
            SET @NewCountWalk = %s;   \
            SET @OldCountWalk = (select CountWalk from cu_amd.HourlyActivitySummary where TimeFrom=%s and UserID=%s and Date=%s); \
            \
            SET @NewCountInactive = %s; \
            SET @OldCountInactive = (select CountInactive from cu_amd.HourlyActivitySummary where TimeFrom=%s and UserID=%s and Date=%s); \
            \
            SET @NewCountActive = %s;  \
            SET @OldCountActive = (select CountActive from cu_amd.HourlyActivitySummary where TimeFrom=%s and UserID=%s and Date=%s); \
            \
            SET @NewCountTotal = %s;  \
            SET @OldCountTotal = (select CountTotal from cu_amd.HourlyActivitySummary where TimeFrom=%s and UserID=%s and Date=%s);  \
            \
            SET @NewCountActToInact = %s;  \
            SET @OldCountActToInact = (select CountActiveToInactive from cu_amd.HourlyActivitySummary where TimeFrom=%s and UserID=%s and Date=%s); \
            \
            SET @NewDurationPAction = %s;  \
            SET @OldDurationPAction = (select DurationPerAction from cu_amd.HourlyActivitySummary where TimeFrom=%s and UserID=%s and Date=%s); \
            \
            UPDATE cu_amd.HourlyActivitySummary  \
            SET  \
                ActualFrom=IF(@OldActualFrom='00:00:00.000', @NewActualFrom, @OldActualFrom),  \
                ActualUntil=IF(@OldActualUntil<@NewActualUntil, @NewActualUntil, @OldActualUntil),  \
                DurationSit=ADDTIME(@OldDurationSit, @NewDurationSit),  \
                DurationSleep=ADDTIME(@OldDurationSleep, @NewDurationSleep),  \
                DurationStand=ADDTIME(@OldDurationStand, @NewDurationStand),  \
                DurationWalk=ADDTIME(@OldDurationWalk, @NewDurationWalk),  \
                TotalDuration=ADDTIME(@OldTotalDuration, @NewTotalDuration), \
                CountSit=@OldCountSit + @NewCountSit,  \
                CountSleep=@OldCountSleep + @NewCountSleep,  \
                CountStand=@OldCountStand + @NewCountStand,  \
                CountWalk=@OldCountWalk + @NewCountWalk,  \
                CountInactive=@OldCountInactive + @NewCountInactive,  \
                CountActive=@OldCountActive + @NewCountActive,  \
                CountTotal=@OldCountTotal + @NewCountTotal,  \
                CountActiveToInactive=@OldCountActToInact + @NewCountActToInact,  \
                DurationPerAction=ADDTIME(@OldDurationPAction, @NewDurationPAction)  \
                WHERE UserID=%s AND Date=%s \
            AND TimeFrom=%s \
            AND TimeUntil=%s;"

    values = zip(df_summary_all['UserID'], df_summary_all['Date'], \
        df_summary_all['TimeFrom'], df_summary_all['TimeUntil'], df_summary_all['ActualFrom'], df_summary_all['ActualUntil'], \
        df_summary_all['DurationSit'], df_summary_all['DurationSleep'], df_summary_all['DurationStand'], df_summary_all['DurationWalk'], \
        df_summary_all['TotalDuration'], df_summary_all['CountSit'], df_summary_all['CountSleep'], df_summary_all['CountStand'], df_summary_all['CountWalk'], \
        df_summary_all['CountInactive'], df_summary_all['CountActive'], df_summary_all['CountTotal'], \
        df_summary_all['CountActiveToInactive'], df_summary_all['DurationPerAction'])

    # Try executing the sql insert query. If the key already exists, execute the sql update query instead
    for z in values:
        u_id = z[0]; dat = z[1]; t_from = z[2]; t_until = z[3]; a_from = z[4]; a_until = z[5]; du_sit = z[6]; du_sleep = z[7];
        du_stand = z[8]; du_walk = z[9]; du_total = z[10]; c_sit = z[11]; c_sleep = z[12]; c_stand = z[13]; c_walk = z[14]; 
        c_inact = z[15]; c_act = z[16]; c_total = z[17]; c_acttoinact = z[18]; 
        du_peract = z[19]

        existed_sql = "SELECT @Existed := EXISTS(select * from cu_amd.HourlyActivitySummary where UserID={} and Date='{}' and TimeFrom='{}');".format(u_id, dat, t_from)
        mycursor.execute(existed_sql)
        result = mycursor.fetchone()
        existed = result[0]

        if(not existed):
            print('insert')
            mycursor.execute(insert_sql, z)
        else:
            print('update hourly summary')
            z_tuple = [a_from, t_from, u_id, dat, a_until, t_from, u_id, dat, du_sit, t_from, u_id, dat, 
            du_sleep, t_from, u_id, dat, du_stand, t_from, u_id, dat, du_walk, t_from, u_id, dat, 
            du_total, t_from, u_id, dat, c_sit, t_from, u_id, dat, c_sleep, t_from, u_id, dat, 
            c_stand, t_from, u_id, dat, c_walk, t_from, u_id, dat, c_inact, t_from, u_id, dat, 
            c_act, t_from, u_id, dat, c_total, t_from, u_id, dat, c_acttoinact, t_from, u_id, dat,
            du_peract, t_from, u_id, dat, u_id, dat, t_from, t_until]

            # print(z_tuple)
            results = mycursor.execute(update_sql, z_tuple, multi=True)
            # print(results)
            for cur in results:
                # print('cursor:', cur)
                if cur.with_rows:
                    continue
                    # print('result:', cur.fetchall())
    
        mydb.commit()

## Insert activity period data of the patient into ActivityPeriod table in database

def insert_db_act_period(df_act_period):
    if(df_act_period.empty):
        print('df activity period is empty')
        return

    cnx = get_sql_connection()

    df_act_period.to_sql('ActivityPeriod', cnx, schema='cu_amd', if_exists='append', index=False, index_label=None, chunksize=100, dtype=None)

## Select the accelerometer and heart rate data of the patient from acc_log_2 and hr_log_2 table in database

def get_patients_acc_hr(user_id):
    mydb, mycursor = connect_to_database()

    df_acc = pd.DataFrame()
    df_hr = pd.DataFrame()

    # Select all accelerometer data of the patient which has not been loaded yet
    sql = "SELECT * FROM cu_amd.acc_log_2 WHERE loaded_flag=False and user_id='{}';".format(user_id)

    mycursor.execute(sql)
    records = mycursor.fetchall()
    print('acc length: ', mycursor.rowcount)

    # If the selected data is not empty or null, append them to df_acc
    if(mycursor.rowcount!=0):
        xyz = []
        timestamp = []
        user_ids = []

        for row in records:
            xyz.append([row[2], row[3], row[4]])
            timestamp.append(row[1])
            user_ids.append(row[5])
        xyz = np.array(xyz)

        df_i = pd.DataFrame({'UserID': user_ids,
                            'timestamp': timestamp,
                            'x': xyz.transpose()[0],
                            'y': xyz.transpose()[1],
                            'z': xyz.transpose()[2]})

        df_acc = df_acc.append(df_i)

        # Select all heart rate data of the patient which has not been loaded yet
        sql2 = "SELECT * FROM cu_amd.hr_log_2 WHERE loaded_flag=False and user_id='{}';".format(user_id)

        mycursor.execute(sql2)
        records = mycursor.fetchall()
        print('hr length: ', mycursor.rowcount)

        # If the selected data is not empty or null, append them to df_hr
        if(mycursor.rowcount!=0):
            hr = []
            timestamp = []
            user_ids = []

            for row in records:
                hr.append(row[2])
                timestamp.append(row[1])
                user_ids.append(row[3])

            df_hr_i = pd.DataFrame({'UserID': user_ids,
                                'timestamp': timestamp,
                                'hr': hr})

            df_hr = df_hr.append(df_hr_i)

        # update the values of loaded_flag to 1
        sql3 = "UPDATE cu_amd.acc_log_2 SET loaded_flag=TRUE WHERE loaded_flag=False and user_id={};".format(user_id)
        sql4 = "UPDATE cu_amd.hr_log_2 SET loaded_flag=TRUE WHERE loaded_flag=False and user_id={};".format(user_id)

        mycursor.execute(sql3)
        mycursor.execute(sql4)

        mydb.commit()

    return df_acc, df_hr

## Select unpredicted data of the patient from ActivityLog table in database

def get_unpredicted_data(user_id, current_idx, batch_size=5000, time=300000):
    mydb, mycursor = connect_to_database()
    print('current idx:', current_idx)

    set_exec_time_sql = [
        "SET GLOBAL connect_timeout=28800", 
        "SET GLOBAL wait_timeout=28800", 
        "SET GLOBAL interactive_timeout=28800"
    ]

    set_user_sql = "SELECT @UserID := {}".format(user_id)
    first_res_sql = "SELECT @FirstResultIdx := Idx FROM ActivityLog WHERE (LoadedFlag=False or LABEL IS NULL) and UserID=@UserID ORDER BY Idx ASC LIMIT 1;"
    last_res_sql = "SELECT @LastResultIdx := Idx FROM ActivityLog WHERE LoadedFlag=False and UserID=@UserID ORDER BY Idx DESC LIMIT 1;"
    
    init_idx_sql = "SELECT @IdxToLoad1 := Idx FROM (SELECT Idx, UserID, DateAndTime, X, Y, Z, HR, Label FROM ActivityLog WHERE (LoadedFlag=False or LABEL IS NULL) and UserID=@UserID ORDER BY Idx ASC) AS table0 limit 1;"
    table_idx_sql = "SELECT @IdxToLoad2 := IdxToLoad from (SELECT IdxToLoad FROM UserProfile WHERE UserID=@UserID) AS table1 limit 1;"
    set_idx_sql = "SELECT @IdxToLoad_0 := IF(@IdxToLoad2 < @IdxToLoad1, @IdxToLoad2, @IdxToLoad1);"
    set_idx_sql_2 = "SELECT @IdxToLoad_0 := IF(@IdxToLoad_0 IS NULL, 0, @IdxToLoad_0);"
    
    query_1_sql = "SELECT Idx, UserID, DateAndTime, X, Y, Z, HR, Label FROM ActivityLog WHERE (LoadedFlag=False or LABEL IS NULL) and UserID=@UserID and Idx<=@LastResultIdx and Idx>=@IdxToLoad_0 and Idx<@IdxToLoad_0+{} and Idx>={};".format(batch_size, current_idx)
    
    set_load_idx_sql = "SELECT @IdxLoaded := IF(@IdxToLoad_0+{}>@LastResultIdx, @LastResultIdx, @IdxToLoad_0+{});".format(batch_size, batch_size)
    update_table_sql = "UPDATE cu_amd.UserProfile set IdxToLoad := @IdxLoaded where UserID=@UserID;"

    update_loaded_flag_sql = "UPDATE cu_amd.ActivityLog SET LoadedFlag=TRUE WHERE LoadedFlag=False and UserID=@UserID and Idx<@IdxLoaded;"

    # try:
    for query in set_exec_time_sql:
        mycursor.execute(query)
    mydb.commit()
    print('set exec time')

    # mycursor.execute(insert_act_log_2_sql)
    mycursor.execute(set_user_sql)
    mycursor.execute(last_res_sql)
    mycursor.execute(init_idx_sql)
    mycursor.execute(table_idx_sql)
    mycursor.execute(set_idx_sql)
    mycursor.execute(set_idx_sql_2)

    mycursor.execute(query_1_sql)
    records = mycursor.fetchall()
    print('records length:', len(records))
    mydb.commit()
    
    df_to_predict = pd.DataFrame()

    if(mycursor.rowcount!=0):
        user_ids = []
        timestamp = []
        xyz = []
        hr = []
        lbl = []

        for row in records:
            user_ids.append(row[1])
            timestamp.append(row[2])
            xyz.append([row[3], row[4], row[5]])
            hr.append(row[6])
            lbl.append(row[7])

        xyz = np.array(xyz)

        df_i = pd.DataFrame({
            'UserID': user_ids,
            'timestamp': timestamp,
            'x': xyz.transpose()[0],
            'y': xyz.transpose()[1],
            'z': xyz.transpose()[2],
            'HR': hr,
            'y_pred': lbl    
        })

        df_to_predict = df_to_predict.append(df_i)

    mycursor.execute(first_res_sql)
    result = mycursor.fetchone()
    if(mycursor.rowcount==0):
        print('here1')
        return df_to_predict, current_idx
    new_current_idx = result[0]
    print('spot1')
    print('new current idx:', new_current_idx)

    mycursor.execute(set_load_idx_sql)
    result = mycursor.fetchone()
    print(result)
    if(mycursor.rowcount==0):
        print('here2')
        return df_to_predict, new_current_idx
    
    print('spot2')
    print('new current idx:', new_current_idx)

    mycursor.execute(update_table_sql)
    mycursor.execute(update_loaded_flag_sql)
    mydb.commit()

    return df_to_predict, new_current_idx


## Select unsummarized data of the patient from ActivityLog table in database

def get_unsummarized_data(user_id):
    cnx = get_sql_connection()

    sql = "SELECT UserID, DateAndTime, X, Y, Z, HR, Label FROM ActivityLog WHERE SummarizedFlag=False and UserID={};".format(user_id)

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

## Select unique patient's IDs from acc_log_2 table in database

def get_distinct_user_ids():
    mydb, mycursor = connect_to_database()

    sql = "SELECT DISTINCT user_id from acc_log_2;"

    mycursor.execute(sql)
    records = mycursor.fetchall()

    return list([r[0] for r in records])

## Update SummarizedFlag in ActivityLog table in database

def update_summarized_flag(user_id, current_idx):
    mydb, mycursor = connect_to_database()
    
    sql = "UPDATE cu_amd.ActivityLog SET SummarizedFlag=TRUE WHERE SummarizedFlag=False and UserID={} and Idx<={};".format(user_id, current_idx)

    mycursor.execute(sql)
    mydb.commit()

## Select min, max value of x,y,z axis in accelerometer data of the patient

def get_user_profile(user_id):
    mydb, mycursor = connect_to_database()

    get_user_profile_sql = "SELECT MinX, MinY, MinZ, MaxX, MaxY, MaxZ FROM cu_amd.UserProfile WHERE UserID={};".format(user_id)
    get_distinct_user_sql = "SELECT distinct UserID from cu_amd.UserProfile;"

    mycursor.execute(get_user_profile_sql)
    user_profile = mycursor.fetchall()
    
    mycursor.execute(get_distinct_user_sql)
    result = mycursor.fetchall()
    distinct_user_ids = [a[0] for a in result]
    print('distinct users:', distinct_user_ids)

    if(not user_id in distinct_user_ids):
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