import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mysql.connector
import os

from datetime import datetime, timedelta
from sqlalchemy import create_engine
from pandas.plotting import register_matplotlib_converters
from pandas.tseries.offsets import DateOffset, MonthBegin, MonthEnd

class Datetime64Converter(mysql.connector.conversion.MySQLConverter):
    '''  A mysql.connector Converter that handles datetime64 types '''

    def _timestamp_to_mysql(self, value):
        return value.strftime('%Y-%m-%d %H:%M:%S.%f').encode('ascii')

def create_db_connection():
    user = 'php'
    passwd = 'HOD8912+php'

    mydb = mysql.connector.connect(
        host='localhost',
        user=user,
        passwd=passwd,
        database='cu_amd'
        )

    print('connected to db')
    mydb.set_converter_class(Datetime64Converter)
    mycursor = mydb.cursor(buffered=True)

    return mydb, mycursor

def get_db_connection():
    user = 'php'
    passwd = 'HOD8912+php'

    host = 'localhost'
    port = '3306'
    schema = 'cu_amd'
    cnx = create_engine('mysql+mysqlconnector://{}:{}@{}:{}/{}'.format(user, passwd, host, port, schema), echo=False)

    return cnx

def get_act_log(date='2019-05-13', user_id=17):
    cnx = get_db_connection()

    act_log_sql = "SELECT UserID, DateAndTime, X, Y, Z FROM ActivityLog WHERE UserID={} AND DateAndTime>=DATE_FORMAT('{}', '%Y-%m-%d')".format(user_id, date)
    
    cols = ['user_id', 'timestamp', 'x', 'y', 'z']
    df_act_log = pd.read_sql(act_log_sql, cnx, columns=cols)
    print('df act log:', df_act_log.head(3))

    return df_act_log

def get_activity_period(date='2019-05-13', user_id=17):
    cnx = get_db_connection()
    act_prd_sql = "SELECT UserID, Date, ActualFrom, ActualUntil, Label FROM ActivityPeriod WHERE UserID={} AND Date>=DATE_FORMAT('{}', '%Y-%m-%d')".format(user_id, date)

    cols = ['user_id', 'date', 'actual_from', 'actual_until', 'label']
    df_act_prd = pd.read_sql(act_prd_sql, cnx, columns=cols)
    print('df act prd:', df_act_prd.head(3))

    return df_act_prd

def get_hourly_summary(date='2019-05-13', user_id=17):
    cnx = get_db_connection()
    hourly_summary_sql = "SELECT * FROM HourlyActivitySummary WHERE UserID={} AND Date=DATE_FORMAT('{}', '%Y-%m-%d')".format(user_id, date)

    cols = ['idx', 'user_id', 'date', 'time_from', 'time_until', 'actual_from', 'actual_until',
        'd_sit', 'd_sleep', 'd_stand', 'd_walk', 'd_total', 
        'c_sit', 'c_sleep', 'c_stand', 'c_walk', 'c_inact', 'c_act', 'c_total',
        'c_trans', 'd_per_act' 
        ]

    df_hourly_sum = pd.read_sql(hourly_summary_sql, cnx, columns=cols)
    print('df hourly summary:', df_hourly_sum.head(3))

    return df_hourly_sum

def combine_date(x):
    return datetime.combine(x, datetime.min.time())

def plot_act_log(df_act_log, df_act_prd, date, min_time='09:30:00.000', max_time='11:00:00.000'):
    df_act_log.index = df_act_log['DateAndTime']
    ax = df_act_log.plot(y=['X','Y','Z'], figsize=(10,4), color=['r','g','b'])

    # Set highlighter ranges
    periods = []
    for i in range(df_act_prd.shape[0]):
        period_i = [df_act_prd.loc[i, 'Date'], df_act_prd.loc[i, 'ActualFrom'], df_act_prd.loc[i, 'ActualUntil'], df_act_prd.loc[i, 'Label']]
        periods.append(period_i)

    periods_new = [[combine_date(period_i[0]), period_i[1], period_i[2], period_i[3]] for period_i in periods]
    periods_plot = [[period_i[0]+period_i[1], period_i[0]+period_i[2], period_i[3]] for period_i in periods_new]

    # Highlighting
    colors = ['coral','goldenrod','lightgreen','violet']
    label_list = ['sit','sleep','stand','walk']
    ticks = [0,0,0,0]
    for i, x in enumerate(periods_plot):
        if(ticks[x[2]]==0):
            ax.axvspan(x[0]-DateOffset(days=0), x[1]+DateOffset(days=0), label=label_list[x[2]],facecolor=colors[x[2]], alpha=0.6)
            ticks[x[2]] = 1
        else:
            ax.axvspan(x[0]-DateOffset(days=0), x[1]+DateOffset(days=0), facecolor=colors[x[2]], alpha=0.6)

    # Set details for the graph plot
    datetime_format = '%Y-%m-%d %H:%M:%S.%f'
    min_timestamp = '{} {}'.format(date, min_time)
    max_timestamp = '{} {}'.format(date, max_time)
    min_x = mdates.date2num(datetime.strptime(min_timestamp, datetime_format))
    max_x = mdates.date2num(datetime.strptime(max_timestamp, datetime_format))

    ax.xaxis_date()
    ax.legend(loc='upper right')
    ax.set_xlim((min_x,max_x))
    ax.set_title('3D-Acceleration with highlighted labels')
    
    # Save graph to an image file
    basepath = '/var/www/html/python/mysql_connect/'
    graph_path = '{}/demo/graphs/'.format(basepath)

    i = 1
    while(i):
        filename = '{}activity_log_plot_{}.png'.format(graph_path, i)
        exists = os.path.isfile(filename)
        if exists:
            i += 1
        else:
            break

    ax.legend()

    f = plt.gcf()
    f.autofmt_xdate()
    myFmt = mdates.DateFormatter('%d/%m %H:%M')
    ax.xaxis.set_major_formatter(myFmt)

    f.savefig(filename, dpi=200)
    plt.close(f)

def get_delta_minutes(n_minutes):
    return timedelta(seconds=60*n_minutes)

if(__name__=='__main__'):
    date = datetime.now().date()
    
    max_time = datetime.strptime('2019-05-12 15:04:00.123', '%Y-%m-%d %H:%M:%S.%f')
    # max_time = datetime.now() - get_delta_minutes(2)
    min_time = max_time - get_delta_minutes(5)

    df_act_log = get_act_log(date)
    df_act_prd = get_activity_period(date)
    df_hourly_sum = get_hourly_summary(date)

    plot_act_log(df_act_log, df_act_prd, date, min_time.time(), max_time.time())