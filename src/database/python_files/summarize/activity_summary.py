#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:

import pickle
import ruptures as rpt
import numpy as np
import pandas as pd
import sys
import os

from datetime import date, datetime, timedelta

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

from load_data.load_dataset import calc_sec, calc_ts
from predict.preprocessing import convert_time_to_string

# # Analyze Predicted Results

# ## All Day Summary

# In[2]:

# Important Parameters

label_list = ['sit', 'sleep', 'stand', 'walk']
label_dict = {'sit': 0, 'sleep': 1, 'stand': 2, 'walk': 3}
LABELS = label_list

date_idx = 0
s_idx = 1
f_idx = 2
lb_idx = 3

date_format = '%Y-%m-%d'
time_format = '%H:%M:%S.%f'

fivemin = 60*60

midnight_1 = calc_sec('00:00:00.000')
midnight_2 = calc_sec('24:00:00.000')

empty_active_count = np.array([0 for i in range(2)])
empty_act_count = np.array([0 for i in range(len(label_list))])

zero_trans = 0
empty_trans_count = np.array([zero_trans])

def get_empty_lists():
    return empty_active_count, empty_trans_count, empty_act_count

def get_period_for_each_label(subject_id, df_date):
    label_period = []

    filter_ = [str(df_date.loc[i, 'ID'])==subject_id for i in range(len(df_date))]
    df_date_id = df_date[filter_]

    df_grp_date = df_date_id.groupby('date')
    for date in df_grp_date:
#         print(date[0])
        df_get_date = df_grp_date.get_group(date[0])
        df_get_date = df_get_date.reset_index(drop=True)

        # df_get_date.to_csv('df_date_{}.csv'.format(date[0]))
        df_get_date['date'] = df_get_date['date'].apply(lambda x: datetime.strptime(x, date_format))
        df_get_date['time'] = df_get_date['time'].apply(lambda x: datetime.strptime(x, time_format))

        print(df_get_date.head())
        print(df_get_date.dtypes)

        keep = 0

        for i in range(len(df_get_date)):

            keep_lb = df_get_date.loc[keep]['y_pred']

            if(keep_lb!=df_get_date.loc[i]['y_pred']):
                label_period.append([df_get_date.loc[keep]['date'], df_get_date.loc[keep]['time'],
                                     df_get_date.loc[i-1]['time'], df_get_date.loc[i-1]['y_pred']])

                keep = i

            if(i==len(df_get_date)-1):
                label_period.append([df_get_date.loc[keep]['date'], df_get_date.loc[keep]['time'],
                                     df_get_date.loc[i]['time'], df_get_date.loc[i]['y_pred']])

    return label_period, df_grp_date

# In[1]:


def get_floor_start(start_time):
    start_sec = calc_sec(start_time)
    return start_sec - (start_sec%fivemin)


# In[2]:


def get_ceil_finish(finish_time):
    finish_sec = calc_sec(finish_time)
    if(finish_sec%fivemin==0):
        return finish_sec - (finish_sec%fivemin)

    return finish_sec - (finish_sec%fivemin) + fivemin

def get_active_count(new_label_period, date, floor_start, ceil_finish):
    t_i = int(floor_start)

    print('floor start, ceil finish, t_i, date')
    print(calc_ts(floor_start), calc_ts(ceil_finish))
    print(calc_ts(t_i))
    print(date)

    active_count = empty_active_count
    act_count = empty_act_count
    trans = zero_trans

    count = 0
    for p_i in range(len(new_label_period)):
        prd = new_label_period[p_i]

        if(prd[date_idx]==date and calc_sec(prd[s_idx])>=t_i and calc_sec(prd[f_idx])<=t_i+fivemin):
            count += 1
            print('t_i')
            print(calc_ts(t_i))
            print('p_i')
            print(p_i)
            print('new label period[p_i]')
            print(prd)
            print('active count')
            print(active_count)

            # count activities
            act_count[prd[lb_idx]] += 1

            # count activeness
            if(prd[lb_idx]==label_dict['sit'] or prd[lb_idx]==label_dict['sleep']):
                active_count[0] += 1
            else:
                active_count[1] += 1

            # count transitions
            if(p_i+1<len(new_label_period)):
                next_prd = new_label_period[p_i+1]

                if((prd[lb_idx]==label_dict['sit'] or prd[lb_idx]==label_dict['sleep']) and
                    (next_prd[lb_idx]==label_dict['stand'] or next_prd[lb_idx]==label_dict['walk'])):
                    trans += 1
                if((prd[lb_idx]==label_dict['stand'] or prd[lb_idx]==label_dict['walk']) and
                    (next_prd[lb_idx]==label_dict['sit'] or next_prd[lb_idx]==label_dict['sleep'])):
                    trans += 1

    # all_active_count.append(active_count)
    # all_trans.append(trans)
    # all_act_count.append(act_count)

    print('counts')
    print(count)

    # if(len(all_act_count)==0):
    #     print('yo!!')
    #     return empty_active_count, empty_trans_count, empty_act_count

    return np.array(active_count), np.array([trans]), np.array(act_count)

# In[1]:


def get_new_label_period(label_period, df_grp_date):
    # separate activity cross between hours
    fivemin = 60*60
    new_label_period = []
    new_period_list = [[] for i in range(len(label_list))]
    time_range = []

    all_act_count = []
    all_active_count = []
    all_trans_count = []

    print('//////////')
    print('label period')
    print(label_period)
    # print(label_period[0][date_idx])

    for date_ in df_grp_date:
        date = date_[0]
        print('date in df_grp_date', date)

        label_period_date = [lb_date for lb_date in label_period if lb_date[date_idx].strftime(date_format)==date]

        if(len(label_period_date)==0):
            break
        
        print('>>>>>>>>>')
        print('label period date')
        print(label_period_date)
        print(label_period_date[0][s_idx])
        print(label_period_date[-1][f_idx])

        floor_start = get_floor_start(label_period_date[0][s_idx].strftime(time_format))
        ceil_finish = get_ceil_finish(label_period_date[-1][f_idx].strftime(time_format))

        time_range.append([date, calc_ts(floor_start), calc_ts(ceil_finish)])
#         print(date, calc_ts(floor_start), calc_ts(ceil_finish))

        tm_s = floor_start
        tm_f = floor_start + fivemin

        # print('date, tm_s, tm_f')
        # print(date)

        for prd in label_period_date:
            prd[s_idx] = prd[s_idx].strftime(time_format)
            prd[f_idx] = prd[f_idx].strftime(time_format)

            start = calc_sec(prd[s_idx])
            finish = calc_sec(prd[f_idx])

            # print('tm_s, tm_f')
            # print(calc_ts(tm_s), calc_ts(tm_f))

            if(finish>=tm_f and
              start-(start%fivemin)==tm_s):
                new_label_period.append([date, prd[s_idx], calc_ts(tm_f), prd[lb_idx]])

                new_period_list[prd[lb_idx]].append([date, prd[s_idx], calc_ts(tm_f), prd[lb_idx]])

            elif(start-(start%fivemin)==tm_s and
                finish-(finish%fivemin)+fivemin==tm_f):
                new_label_period.append([date, prd[s_idx], prd[f_idx], prd[lb_idx]])

                new_period_list[prd[lb_idx]].append([date, prd[s_idx], prd[f_idx], prd[lb_idx]])

            while(finish>=tm_f):
                tm_s += fivemin
                tm_f += fivemin

            if(start<tm_s):
                new_label_period.append([date, calc_ts(tm_s), prd[f_idx], prd[lb_idx]])

                new_period_list[prd[lb_idx]].append([date, calc_ts(tm_s), prd[f_idx], prd[lb_idx]])

        print('###########')
        print('new label period')
        print(new_label_period)
        print('###########')

        # cnt = 0
        for t_i in range(int(midnight_1), int(midnight_2), fivemin):
            # cnt += 1
            active_count_i, trans_count_i, act_count_i = get_active_count(new_label_period, date, t_i, t_i+fivemin)

            if(not np.array_equal(active_count_i, empty_active_count)):
                print()
                print('date')
                print(date)
                print('activity count i, active count, trans count i')
                print(act_count_i)
                print(active_count_i) 
                print(trans_count_i)
                print()
                all_act_count.append(act_count_i)
                all_active_count.append(active_count_i)
                all_trans_count.append(trans_count_i)
            else:
                all_act_count.append(empty_act_count)
                all_active_count.append(empty_active_count)
                all_trans_count.append(empty_trans_count)

        # print('loop counts')
        # print(cnt)
        print()

    print('??????')
    print('all act count')
    print(all_act_count)
    all_active_count = np.array(all_active_count)
    # all_active_count = all_active_count.reshape((all_active_count.shape[0], 2))

    if(len(all_act_count)!=0):
        all_act_count = np.vstack([arr for arr in all_act_count])
    else:
        all_act_count = np.array(all_act_count)

    if(len(all_trans_count)!=0):
        all_trans_count = np.hstack(all_trans_count)
    else:
        all_trans_count = np.array(all_trans_count)

    return new_label_period, new_period_list, all_active_count, all_trans_count, all_act_count


# In[265]:


def get_all_periods_label(new_label_period, df_grp_date):

    all_periods_label = {}
    print('new label period:', new_label_period)
    print('df grp date:', df_grp_date)

    for date_ in df_grp_date:
        date = date_[0]
        all_periods_label[date] = []
        print('!!!!!!!!!!')
        print('date, new label period')
        print(date)
        print(new_label_period)
        print('!!!!!!!!!!')

        new_label_period_date = [lb_date for lb_date in new_label_period if lb_date[date_idx]==date]
        print(new_label_period_date)

        if(len(new_label_period_date)!=0):
            print('=======')
            print('new label period date start, new label period date finish')
            print(new_label_period_date[0])
            print(new_label_period_date[-1])
            
            # start_time = calc_sec(new_label_period_date[0][s_idx])
            # finish_time = calc_sec(new_label_period_date[-1][f_idx])

            # floor_start = start_time - (start_time%fivemin)
            # ceil_finish = finish_time - (finish_time%fivemin) + fivemin

            for t_i in range(int(midnight_1), int(midnight_2), fivemin):
                period_lb = [0 for i in range(len(LABELS))]
                for prd in new_label_period_date:
                    if(calc_sec(prd[s_idx])>=t_i and
                    calc_sec(prd[f_idx])<=t_i+fivemin and
                    prd[lb_idx]!=-1
                    ):
                        period_lb[prd[lb_idx]] += calc_sec(prd[f_idx])-calc_sec(prd[s_idx])
                        period_lb[prd[lb_idx]] = round(period_lb[prd[lb_idx]], 3)

                all_periods_label[date].append(np.hstack(([calc_ts(t_i), calc_ts(t_i+fivemin)],
                                                    [convert_time_to_string(i) for i in period_lb])))

            print('all periods label')
            print(all_periods_label)
            print('=======')

    return all_periods_label


# In[5]:


def get_df_summary(all_periods_label, new_period_list, subject_id, all_active_count, all_trans_count, all_act_count):
    date_summary = []
    from_summary = []
    to_summary = []
    act_summary = [[] for i in range(len(label_list))]

    for date in all_periods_label.keys():
        date_length = len(all_periods_label[date])

        date_summary.append([date for i in range(date_length)])
        from_summary.append([i[0] for i in all_periods_label[date]])
        to_summary.append([i[1] for i in all_periods_label[date]])

        act_idx = [2,3,4,5]
        for j in range(len(act_idx)):
            act_summary[j].append([i[act_idx[j]] for i in all_periods_label[date]])

    date_summary = np.hstack(date_summary)
    from_summary = np.hstack(from_summary)
    to_summary = np.hstack(to_summary)

    for i in range(len(act_summary)):
        act_summary[i] = np.hstack(act_summary[i])

    total_act = np.array([convert_time_to_string(int(np.sum([calc_sec(act_summary[i][j]) for i in range(len(label_list))])))
                                        for j in range(act_summary[0].shape[0])])

    all_active_count = np.transpose(all_active_count)
    all_act_count = np.transpose(all_act_count)

    if(all_active_count.shape[0]!=0):
        all_active_count = np.vstack(all_active_count)
        all_act_count = np.vstack(all_act_count)
    else:
        return pd.DataFrame()

    # print('transposed all active count')
    # print(all_active_count)
    # print(all_active_count.shape)

    total_count = np.array([ac[0]+ac[1] for ac in all_active_count.transpose()])

    act_summary = np.array(act_summary)

    print('shapes of date summary, activity summary, total activity count, sit activity count, active count')
    print(date_summary.shape)
    print(act_summary[0].shape)
    print(total_act.shape)
    print(all_act_count.shape)
    print(all_active_count.shape)
    print('shapes of total count, transition count')
    print(total_count.shape)
    print(all_trans_count.shape)

    # print('display date summary, from summary, to summary, activity summary, activity count, activeness count, and transition count')
    # print(date_summary)
    # print(from_summary)
    # print(to_summary)
    print(act_summary)
    # print(all_act_count)
    print(all_active_count)
    # print(all_trans_count)
    # print('---------')

    df_summary = pd.DataFrame({
        'ID': [subject_id for i in range(date_summary.shape[0])],
        'date': date_summary,
        'from': from_summary,
        'to': to_summary,
        'sit': act_summary[0],
        'sleep': act_summary[1],
        'stand': act_summary[2],
        'walk': act_summary[3],
        'total': total_act,
        'sit count': all_act_count[0],
        'sleep count': all_act_count[1],
        'stand count': all_act_count[2],
        'walk count': all_act_count[3],
        'inactive count': all_active_count[0],
        'active count': all_active_count[1],
        'total count': total_count,
        'transition count': pd.Series(all_trans_count)
    })

    return df_summary


# In[1]:


def get_df_summary_all(all_subjects, df_date):
    df_summary_all = pd.DataFrame()
    df_act_period_all = pd.DataFrame()

    for subject_id in all_subjects:

        label_period, df_grp_date = get_period_for_each_label(subject_id, df_date)

        new_label_period, new_period_list, all_active_count, all_trans_count, all_act_count = get_new_label_period(label_period, df_grp_date)

        all_periods_label = get_all_periods_label(new_label_period, df_grp_date)

        df_summary = get_df_summary(all_periods_label, new_period_list, subject_id, all_active_count, all_trans_count, all_act_count)
        if(df_summary.empty):
            return pd.DataFrame(), pd.DataFrame()
        
        df_summary_all = df_summary_all.append(df_summary, sort=False)

        cols = ['date','from','to','y_pred']

        all_period_list = np.vstack([new_period_list[i] for i in range(len(label_list)) if(len(new_period_list[i])!=0)])
        df_act_period = pd.DataFrame(all_period_list, columns=cols)

        df_act_period['ID'] = pd.Series([subject_id for i in range(df_act_period.shape[0])])
        df_act_period['label'] = df_act_period['y_pred'].apply(lambda x: label_list[int(x)])
        df_act_period['from_sec'] = df_act_period['from'].apply(lambda x: calc_sec(x))

        df_act_period = df_act_period.sort_values(by=['date','from_sec'])
        df_act_period = df_act_period.reset_index(drop=True)
        df_act_period_all = df_act_period_all.append(df_act_period, sort=False)

#     all_period_list = np.vstack(new_period_list[i] for i in range(len(label_list)))

    return df_summary_all, df_act_period_all
