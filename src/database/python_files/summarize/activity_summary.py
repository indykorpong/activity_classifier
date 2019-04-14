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

label_list = ['sit', 'sleep', 'stand', 'walk']
label_dict = {'sit': 0, 'sleep': 1, 'stand': 2, 'walk': 3}
LABELS = label_list

def get_period_for_each_label(subject_id, df_date):
    label_period = []
    period_list = [[] for i in range(len(label_list))]

    filter_ = [str(df_date.loc[i, 'ID'])==subject_id for i in range(len(df_date))]
    df_date_id = df_date[filter_]

    df_grp_date = df_date_id.groupby('date')
    for date in df_grp_date:
#         print(date[0])
        df_get_date = df_grp_date.get_group(date[0])
        df_get_date = df_get_date.reset_index(drop=True)

        print('df_get_date:', df_get_date)
        df_get_date.to_csv('df_date_{}.csv'.format(date[0]))

        keep = 0

        for i in range(len(df_get_date)):

            keep_lb = df_get_date.loc[keep]['y_pred']

            if(keep_lb!=df_get_date.loc[i]['y_pred']):
                label_period.append([df_get_date.loc[keep]['date'], df_get_date.loc[keep]['time'],
                                     df_get_date.loc[i-1]['time'], df_get_date.loc[i-1]['y_pred']])

                period_list[int(df_get_date.loc[i-1]['y_pred'])].append(
                    [date[0], df_get_date.loc[keep]['time'], df_get_date.loc[i-1]['time']])

                keep = i

            if(i==len(df_get_date)-1):
                label_period.append([df_get_date.loc[keep]['date'], df_get_date.loc[keep]['time'],
                                     df_get_date.loc[i]['time'], df_get_date.loc[i]['y_pred']])

                period_list[int(df_get_date.loc[i]['y_pred'])].append(
                    [date[0], df_get_date.loc[keep]['time'], df_get_date.loc[i]['time']])

    return label_period, df_grp_date


# In[222]:


date_idx = 0
s_idx = 1
f_idx = 2
lb_idx = 3

fivemin = 60*60


# In[ ]:


def get_active_count(new_label_period, floor_start, ceil_finish):
    all_act_count = []
    all_active_count = []
    all_trans = []

    for t_i in range(int(floor_start), int(ceil_finish), fivemin):
        active_count = [0 for i in range(2)]
        act_count = [0 for i in range(len(label_list))]

        trans = 0

        for p_i in range(len(new_label_period)):
            prd = new_label_period[p_i]

            if(calc_sec(prd[s_idx])>=t_i and calc_sec(prd[f_idx])<=t_i+fivemin):
                act_count[prd[lb_idx]] += 1

                # count activities
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

        all_act_count.append(act_count)
        all_active_count.append(active_count)
        all_trans.append(trans)

    return np.array(all_active_count), np.array(all_trans), np.array(all_act_count)


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

    for date_ in df_grp_date:
        date = date_[0]

        label_period_date = [lb_date for lb_date in label_period if lb_date[date_idx]==date]

        print('label period date')
        print(label_period_date[0])
        print(label_period_date[-1])

        floor_start = get_floor_start(label_period_date[0][s_idx])
        ceil_finish = get_ceil_finish(label_period_date[-1][f_idx])

        time_range.append([date, calc_ts(floor_start), calc_ts(ceil_finish)])
#         print(date, calc_ts(floor_start), calc_ts(ceil_finish))

        tm_s = floor_start
        tm_f = floor_start + fivemin
        date = label_period[0][date_idx]

        for prd in label_period_date:
            start = calc_sec(prd[s_idx])
            finish = calc_sec(prd[f_idx])

            if(finish>=tm_f and
              start-(start%fivemin)==tm_s):

                new_prd = [date, prd[s_idx], calc_ts(tm_f), prd[lb_idx]]
                new_label_period.append(new_prd)

                new_period_list[prd[lb_idx]].append([date, prd[s_idx], calc_ts(tm_f), prd[lb_idx]])

            elif(start-(start%fivemin)==tm_s and
                finish-(finish%fivemin)+fivemin==tm_f):
                new_label_period.append(prd)

                new_period_list[prd[lb_idx]].append([date, prd[s_idx], prd[f_idx], prd[lb_idx]])

            while(finish>=tm_f):
                tm_s += fivemin
                tm_f += fivemin

            if(start<tm_s):
                new_prd = [date, calc_ts(tm_s), prd[f_idx], prd[lb_idx]]
                new_label_period.append(new_prd)

                new_period_list[prd[lb_idx]].append([date, calc_ts(tm_s), prd[f_idx], prd[lb_idx]])

        for t_i in range(int(floor_start), int(ceil_finish), fivemin):
            active_count_i, trans_count_i, act_count_i = get_active_count(new_label_period, t_i, t_i+fivemin)

            all_act_count.append(act_count_i)
            all_active_count.append(active_count_i)
            all_trans_count.append(trans_count_i)

    all_active_count = np.array(all_active_count)
    all_active_count = all_active_count.reshape((all_active_count.shape[0], 2))

    all_act_count = np.vstack([arr for arr in all_act_count])
#     print(all_act_count)

    return new_label_period, new_period_list, all_active_count, np.hstack(all_trans_count), all_act_count


# In[265]:


def get_all_periods_label(new_label_period, df_grp_date):

    all_periods_label = {}
    print('new label period:', new_label_period)
    print('df grp date:', df_grp_date)

    for date_ in df_grp_date:
        date = date_[0]
        all_periods_label[date] = []
        print(date)

        new_label_period_date = [lb_date for lb_date in new_label_period if lb_date[date_idx]==date]
        print(new_label_period_date)

        if(len(new_label_period_date)!=0):

            start_time = calc_sec(new_label_period_date[0][s_idx])
            finish_time = calc_sec(new_label_period_date[-1][f_idx])

            floor_start = start_time - (start_time%fivemin)
            ceil_finish = finish_time - (finish_time%fivemin) + fivemin

            for t_i in range(int(floor_start), int(ceil_finish), fivemin):
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

    total_act = [convert_time_to_string(int(np.sum([calc_sec(act_summary[i][j]) for i in range(len(label_list))])))
                                        for j in range(act_summary[0].shape[0])]

    all_act_count = np.transpose(all_act_count)
    total_count = [ac[0]+ac[1] for ac in all_active_count]
    # duration_per_act = []
    # for total_i in total_count:
    #     if(total_i!=0):
    #         duration_per_act.append(convert_time_to_string(60*60/total_i))
    #     else:
    #         duration_per_act.append(convert_time_to_string(0))

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
        'inactive count': np.transpose(all_active_count)[0],
        'active count': np.transpose(all_active_count)[1],
        'total count': total_count,
        'transition count': all_trans_count
        # 'duration per action': [convert_time_to_string(60*60/total_i) for total_i in total_count]
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
