import mysql.connector
import numpy as np
import pandas as pd
import csv

status_started = 0
status_stopped = 1
status_error = -1
status_sleep = -2

def reset_status():
    load_status = status_sleep
    predict_status = status_sleep
    summarize_status = status_sleep
    return [load_status, predict_status, summarize_status]

def set_safe_updates(enable, mydb, mycursor):
    if(enable):
        sql = "SET SQL_SAFE_UPDATES = 1;"
    else:
        sql = "SET SQL_SAFE_UPDATES = 0;"
    
    mycursor.execute(sql)
    mydb.commit()

def insert_db_status(process_name, start_time, stop_time, status, mydb, mycursor):
    set_safe_updates(False, mydb, mycursor)

    sql2 = "UPDATE Logging SET ProcessStatus = %s WHERE StartTime = %s AND ProcessName = %s"
    error = False

    print(process_name, status)

    try:
        if(status==status_stopped or status_error):
            sql = "UPDATE Logging SET StopTime = %s WHERE StartTime = %s AND ProcessName = %s"

            values = (stop_time, start_time, process_name)
            values2 = (status, start_time, process_name)
        
        elif(status==status_started):
            sql = "INSERT INTO Logging (StartTime, StopTime, ProcessName, ProcessStatus) VALUES (%s, %s, %s, %s)"
            values = (start_time, stop_time, process_name, status)

    except:
        sql = "UPDATE Logging SET StopTime = %s WHERE StartTime = %s AND ProcessName = %s"
        
        values = (stop_time, start_time, process_name)
        values2 = (status_error, start_time, process_name)
        error = True
    
    mycursor.execute(sql, values)
    
    if(status==status_error or status==status_stopped or error):
        mycursor.execute(sql2, values2)

    mydb.commit()

    set_safe_updates(True, mydb, mycursor)

def insert_db_patient(df_all_p_sorted, mydb, mycursor):
    sql = "INSERT INTO Patient (ID, DateAndTime, X, Y, Z, HR, ActivityIndex, Label) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"

    for row in zip(df_all_p_sorted['ID'],
                df_all_p_sorted['timestamp'],
                df_all_p_sorted['x'],
                df_all_p_sorted['y'],
                df_all_p_sorted['z'],
                df_all_p_sorted['HR'],
                df_all_p_sorted['AI'],
                df_all_p_sorted['y_pred']):

        mycursor.execute(sql, row)

    mydb.commit()


def insert_db_all_day_summary(df_summary_all, mydb, mycursor):
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

def insert_db_act_period(df_act_period, mydb, mycursor):
    sql = "INSERT INTO ActivityPeriod (ID, Date, TimeFrom, TimeUntil, Label)    VALUES (%s, %s, %s, %s, %s)"

    for row in zip(df_act_period['ID'],
                df_act_period['date'],
                df_act_period['from'],
                df_act_period['to'],
                df_act_period['y_pred']):

        mycursor.execute(sql, row)

    mydb.commit()

def get_patients_acc_hr(all_patients, date_to_retrieve, mydb, mycursor):
    df_acc = pd.DataFrame()
    df_hr = pd.DataFrame()

    for p in all_patients:
        sql = "SELECT * FROM cu_amd.accelerometer_log where user_id={} and (event_timestamp > DATE_FORMAT({}, '%y-%m-%d'));".format(p, date_to_retrieve)
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
            user_id.append(row[5])
        xyz = np.array(xyz)

        df_i = pd.DataFrame({'ID': user_id,
                            'timestamp': timestamp,
                            'x': xyz.transpose()[0],
                            'y': xyz.transpose()[1],
                            'z': xyz.transpose()[2]})

        df_acc = df_acc.append(df_i)

        mycursor.close()

        sql2 = "SELECT * FROM cu_amd.hr_log where user_id={} and (event_timestamp > DATE_FORMAT({}, '%y-%m-%d'));".format(p, date_to_retrieve)

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

        mycursor.close()

    return df_acc, df_hr

