#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math

from os import listdir, walk
from os.path import isfile, join
from sklearn.preprocessing import MinMaxScaler

basepath = ''
# basepath = '/var/www/html/python/mysql_connect/'
datapath = basepath + 'DDC_Data/'
mypath = basepath + 'DDC_Data/raw/'

# # Define Timestamp Methods

# In[4]:


def calc_sec(time):
    hms = time.split(':')
    hms = [float(x) for x in hms]
    sec = hms[2] + hms[1]*60 + hms[0]*3600
    sec = round(sec,3)
    return sec


# In[5]:


def calc_ts(sec):
    ts = ''
    hr = int(sec/3600)
    mn = int((sec - (hr*3600))/60)
    sc = sec - (hr*3600) - (mn*60)
    sc = round(sc,3)
    ts += str(hr) + ':' + str(mn) + ':' + str(sc)
    # print(ts)
    return ts


# In[6]:


def calc_t_period(dates,secs):
    t_period = []

    start_sec = secs[0]
    prev_sec = secs[0]
    prev_date = dates[0]

    for i in range(len(secs)):
        curr_sec = secs[i]
        diff_sec = curr_sec - prev_sec
        curr_date = dates[i]

        if((diff_sec>3.0) and (curr_date==prev_date)):
            t_period.append([curr_date,start_sec,prev_sec])
            start_sec = curr_sec
        elif(curr_date!=prev_date):
            t_period.append([prev_date,start_sec,prev_sec])
            start_sec = curr_sec
            prev_date = curr_date
        elif(i==len(secs)-1):
            t_period.append([curr_date,start_sec,curr_sec])

        prev_sec = curr_sec

    return t_period


# # Load Dataset

# In[7]:


# Retrieve file directories from Google Drive
dir_ = [f for f in walk(mypath)]
# print(dir_)

dir = list(dir_[0])
dir[1] = sorted(dir[1])

outer_path = dir[0]
sub_path = dir[1]

folders = [join(outer_path,d) for d in sub_path]

files = []
for fd in folders:
    temp_f = [f for f in listdir(fd) if isfile(join(fd, f)) and f[-3:]=='csv' and f[5:9]!='data' and f[:4]==fd[-4:]]
    temp_f = sorted(temp_f)


# ## Retrieve All Timestamp Periods from a File

# In[47]:


all_subjects = []

for i in range(1001,1013):
    all_subjects.append(str(i))

for i in range(2001,2003):
    all_subjects.append(str(i))

# print(all_subjects)


# In[48]:


def fix_thai_language():
    # -- coding: utf-8 --

    filepath = '/Users/admin/Downloads/history_amdtimer.csv'

    df = pd.read_csv(filepath, header=None, names=['sid','raw_label', 'timestamp', 'duration','label'])

    temp_series = []

    for i in range(len(df)):

        if(df.iloc[i][1]=='ยืน'):
            temp_series.append('stand')

        elif(df.iloc[i][1]=='นั่ง'):
            temp_series.append('sit')

        elif(df.iloc[i][1]=='นอน'):
            temp_series.append('sleep')

        elif(df.iloc[i][1]=='เดิน'):
            temp_series.append('walk')

        elif(df.iloc[i][1]=='ขึ้นบันได'):
            temp_series.append('walk')

        elif(df.iloc[i][1]=='ลงบันได'):
            temp_series.append('walk')

        else:
            temp_series.append(df.loc[i]['raw_label'])

    df['label'] = pd.Series(temp_series)
    df['raw_label'] = df['label']
    df = df.drop(columns=['label'])

#     print(df)
    df.to_csv('iphone-history_amdtimer.csv', sep=',')


# In[49]:


def identify_subj_id(i, all_subjects):
    subject_id = all_subjects[i]

    directory = basepath + 'Graphs/' + subject_id

    if(not os.path.exists(directory)):
        os.makedirs(directory)

    return subject_id


# In[107]:


def load_timer(subject_id):
  # Configure starting and ending time values
    sid_dir = mypath + '/' + subject_id
    sid_files = [f for f in listdir(sid_dir) if 'history_amdtimer' in f]

    sid_filepath = sid_dir + '/' + sid_files[0]

    # Timestamp periods dataframe
    timer_df = pd.read_csv(sid_filepath, header=None, names=['sid','raw_label', 'timestamp', 'duration','label'])

    filtered_timer = [i for i in timer_df['sid'] if i==int(subject_id)]

    timer_filt = timer_df[timer_df['sid'].isin(filtered_timer)]
    timer_filt = timer_filt.reset_index(drop=True)

    timer_arr = []

    for i in range(len(timer_filt)):
        if(timer_filt.loc[i]['raw_label']=='upstairs' or
          timer_filt.loc[i]['raw_label']=='downstairs'):
            timer_arr.append('walk')
        else:
            timer_arr.append(timer_filt.loc[i]['raw_label'])

    timer_filt['label'] = pd.Series(timer_arr)

    start_ts = timer_filt.loc[0]['timestamp']
    end_ts = timer_filt.loc[len(timer_filt)-1]['timestamp']

    rec_date = start_ts.split(' ')[0]
    start_time = start_ts.split(' ')[1]
    end_time = calc_ts(calc_sec(end_ts.split(' ')[1]) +
                       calc_sec(timer_filt.loc[len(timer_filt)-1]['duration']))

#     print(timer_filt)

    return timer_filt, rec_date, start_time, end_time


# ## Create Dataframe of ACC and HR

# In[51]:


def load_acc(subject_id, rec_date, start_time, end_time):
    # Load accelerations
    acc_filepath = mypath + '/' + subject_id + '/' + subject_id + '-log_acc.csv'

    df = pd.read_csv(acc_filepath, header=None, names=['x','y','z','timestamp'])

    filtered = [i for i in df['timestamp'] if str(i)[:10]==rec_date and calc_sec(str(i)[11:])>=calc_sec(start_time)
              and calc_sec(str(i)[11:])<=calc_sec(end_time)]

    df_filt = df[df['timestamp'].isin(filtered)]
    df_filt = df_filt.reset_index(drop=True)

    df_filt['ID'] = pd.Series([subject_id for i in range(len(df_filt))])

    cols = ['ID','timestamp','x','y','z']
    df_filt = df_filt[cols]

    return df_filt


# In[52]:


def load_hr(subject_id, rec_date, start_time, end_time):
    # Load heart rate
    hr_filepath = mypath + '/' + subject_id + '/' + subject_id + '-log_hr.csv'

    df2 = pd.read_csv(hr_filepath, header=None, names=['hr','timestamp'])

    filtered = [i for i in df2['timestamp'] if i[:10]==rec_date and calc_sec(i[11:])>=calc_sec(start_time)
              and calc_sec(i[11:])<=calc_sec(end_time)]

    df_hr = df2[df2['timestamp'].isin(filtered)]
    df_hr = df_hr.reset_index(drop=True)

    cols = ['timestamp','hr']
    df_hr = df_hr[cols]

    return df_hr


# In[53]:


def merge_acc_and_hr(df_filt, df_hr):
    # Fill in missing HRs
    hr_cnt = 0

    for i in range(len(df_filt)):
        hr_time = df_hr.loc[hr_cnt,'timestamp'].split(' ')[1]
        filt_time = df_filt.loc[i,'timestamp'].split(' ')[1]

        if(calc_sec(hr_time)<=calc_sec(filt_time)):
            if(hr_cnt<len(df_hr)-1):
                hr_cnt += 1
        df_filt.loc[i,'HR'] = df_hr.loc[hr_cnt,'hr']

    # Normalize by dividing by g (standard gravity)
    g = 9.8
    df_filt.loc[:,'x'] = df_filt['x'].apply(lambda x: x/g)
    df_filt.loc[:,'y'] = df_filt['y'].apply(lambda x: x/g)
    df_filt.loc[:,'z'] = df_filt['z'].apply(lambda x: x/g)

    cols = ['x','y','z']
    xyz_ = df_filt[cols].to_dict('split')['data']
    xyz_new = MinMaxScaler().fit_transform(xyz_)
#     print(np.array(xyz_new).shape)

    for i in range(len(cols)):
        df_filt[cols[i]] = pd.Series(xyz_new.transpose()[i])

#     print(df_filt['x'])

    return df_filt


# # Calculate Activity Index

# In[54]:


std_i_bar = [0.00349329,0.00465817,0.00543154]
std_i_bar = np.array(std_i_bar)


# In[55]:


def equation_bai(X_i):
    all_std = []

    std_i = np.std(X_i,axis=0)
    diff_std = std_i**2 - std_i_bar**2
    diff_std = (diff_std + 1) / (std_i_bar**2 + 1)

    diff_std_ = std_i**2

    all_std.append(diff_std)

    all_std = np.array(all_std)

    ai = np.sum(all_std**2,axis=1)/3
    ai[ai<0] = 0
    ai = np.sqrt(ai)

    return ai


# In[56]:


def calc_ai(df1):
    H = 10
    ai1 = []

    for i in range(len(df1)):
        xyz_val = []
        if(i-H>=0):
            for j in range(H,0,-1):
                xyz_val.append([df1.loc[i-j,'x'],df1.loc[i-j,'y'],df1.loc[i-j,'z']])
            ai_val = float(equation_bai(xyz_val))
            ai1.append(ai_val)
        else:
            ai1.append(1)

    return ai1


# # Colors for Each Acitivity

# In[77]:


def prepare_time_periods(timer_filt):
    t_ = [calc_sec(t.split(' ')[1]) for t in timer_filt['timestamp']]
    duration = [d for d in timer_filt['duration']]
    lb_ = [lb for lb in timer_filt['label']]

    t_end = [t_[i]+calc_sec(duration[i]) for i in range(len(t_))]

    ts_ = []
    labels = []

    for i in range(len(t_)):
        ts_.append(calc_sec(duration[i]))
        labels.append(lb_[i])
        if(i+1<len(t_)):
            ts_.append(round(t_[i+1]-t_end[i],3))
            labels.append('NaN')

    return ts_, labels


# In[58]:


def prepare_color_labels(ts_, labels):

    accum = 0
    ts = []
    for x in ts_:
        accum += x
        ts.append(round(accum,3))

    lb_set = set()
    for x in labels:
        lb_set.add(x)

    lb_ = list(lb_set)

    set_cnt = []
    for i in range(len(lb_)):
        set_cnt.append(0)

    lb = []
    lb.append('NaN')

    for x in labels:
        for i in range(len(lb_)):
            if(lb_[i]==x and set_cnt[i]!=1 and lb_[i]!='NaN'):
                set_cnt[i] = 1
                lb.append(x)

    colors = ['#808080', '#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231',
            '#911EB4', '#46F0F0', '#F032E6', '#BCF60C', '#008080', '#E6BEFF',
            '#9A6324', '#800000', '#AAFFC3', '#808000', '#000075']

    color_dict = {}
    for i in range(len(lb)):
        color_dict[lb[i]] = colors[i]

    #   print(color_dict)

    lb_color = []
    for x in labels:
        lb_color.append(color_dict[x])

    return ts, lb_color


# ## Plot ACC, AI with Colors

# In[59]:


def plot_ai(df1, ts, lb_color):
    dict1 = df1.to_dict(orient='list')

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16,12))
    fig.tight_layout()

    ts_cnt = 0
    x_axis = []
    y_ai = []

    ax = axes[0]
    ax.plot(dict1['x'],color='r',label='X')
    ax.plot(dict1['y'],color='g',label='Y')
    ax.plot(dict1['z'],color='b',label='Z')
    ax.legend(loc='upper right')
    ax.set_title('X,Y,Z')

    ax = axes[1]
    for i in range(len(dict1['timestamp'])):
        if(dict1['AI'][i]>0):
            if(calc_sec(dict1['timestamp'][i].split(' ')[1])>calc_sec(start_time)+ts[ts_cnt]):
                ax.plot(x_axis,y_ai,color=lb_color[ts_cnt])
                ts_cnt += 1
                x_axis = []
                y_ai = []

            elif(ts_cnt==len(lb_color)-1):
                ax.plot(x_axis,y_ai,color=lb_color[ts_cnt])

            x_axis.append(i)
            y_ai.append(dict1['AI'][i])

    ax.set_title('Activity Index')

    fig.savefig(basepath + 'Graphs/' + subject_id + '/' + subject_id + '_ddc_run.png', dpi = 300)

    #   plt.show()
    plt.close(fig)


# # Create Dataframe with AI

# In[60]:


def ai(subject_id, rec_date, start_time, end_time, df_timer):
    df_filt = load_acc(subject_id, rec_date, start_time, end_time)
    df_hr = load_hr(subject_id, rec_date, start_time, end_time)

    df1 = merge_acc_and_hr(df_filt, df_hr)
    ai1 = calc_ai(df1)

    df1['AI'] = pd.Series(ai1)

    ts_, labels = prepare_time_periods(df_timer)
    ts, lb_color = prepare_color_labels(ts_, labels)

    #   print(ts_, lb_color)

    return df1, ai1, ts, lb_color


# # Separate Data by Labels of Activity

# In[61]:


class period:
    def __init__(self, s, f):
        self.s = s
        self.f = f


# In[113]:


def separate_label(df1, df_timer, df_list, labels):

    df_label = df_timer.groupby('label')
    td_col = ['timestamp','duration']
    cols = ['x','y','z']

    for x in df_label:
    # x[0] is a label
    # x[1] is a groupby object

        df_label_x = df_label.get_group(x[0])
        df_label_x = df_label_x.reset_index(drop=True)

        temp_ts = [[df_label_x.loc[a]['timestamp'].split(' ')[1],
                    calc_ts( calc_sec(df_label_x.loc[a]['timestamp'].split(' ')[1])+
                            calc_sec(df_label_x.loc[a]['duration']) )]
                    for a in range(len(df_label_x))]

        for a in temp_ts:
            filter_ = [i for i in df1['timestamp']
                    if calc_sec(i.split(' ')[1])>=calc_sec(a[0]) and calc_sec(i.split(' ')[1])<=calc_sec(a[1])]

            df1_new = df1[df1['timestamp'].isin(filter_)]
            df1_new = df1_new.reset_index(drop=True)

#             xyz_ = df1_new[cols].to_dict('split')['data']
#             xyz_new = MinMaxScaler().fit_transform(xyz_)

#             for i in range(len(cols)):
#                 df1_new[cols[i]] = pd.Series(xyz_new.transpose()[i])

            for i in range(len(labels)):
                if(labels[i]==x[0]):
                    df_list[i] = df_list[i].append(df1_new, sort=False)

    return df_list


# # Dataframe List Grouped by Label

# In[110]:


def group_dataframe(df1, df_timer):
    df_list = []
    cols = ['timestamp','x','y','z','HR','AI']

    lbl = set()
    for tm in range(len(df_timer)):
        lbl.add(df_timer.loc[tm]['label'])

    LABELS = sorted(list(lbl))

    # dictionary mapped from activity label to index
    label_dict = {
      'sit': 0,
      'sleep': 1,
      'stand': 2,
      'walk': 3
    }

    for i in range(len(LABELS)):
        df_null = pd.DataFrame(columns=cols)
        df_null = df_null.fillna(0)

        df_list.append(df_null)

    df_list = separate_label(df1, df_timer, df_list, LABELS)

    for i in range(len(df_list)):
        df_list[i] = df_list[i].reset_index(drop=True)

#     print(len(df_list[3]))

    return df_list, label_dict


# ## Show Plots of Grouped Dataframe

# In[64]:


def plot_grouped_df(df_list, label_dict):
    xyz = ['x','y','z']
    xyz_color = ['r','g','b']

    for x in label_dict:
    #     print(label_dict[x])

        figure = plt.figure(figsize=(20,6))
        figure.tight_layout()

        cnt = 1

        for i in range(len(xyz)):
            ax = plt.subplot(1, len(xyz), cnt)

            ax.set_ylim(top=1.5, bottom=-3.0)
            ax.plot(df_list[label_dict[x]][xyz[i]], label=xyz[i], color=xyz_color[i])
            ax.legend(loc='upper right')
            ax.set_title(xyz[i] + '-axis for activity ' + x + ' subject no. ' + subject_id)

            cnt += 1

        figure.savefig(basepath + 'Graphs/ddc_' + x + '/' + subject_id + '.png', dpi=300)

    #     plt.show()

    # close the figure
    plt.close(figure)


# # Get X and y from Dataset for Each Subject

# In[65]:


def get_data(df_list, label_dict):
    feature_cols = ['x','y','z']
    count = 0

    y_all = []
    ts_all = []
    hr_all = []

    for x in label_dict:
#         print(x)

        X_series = df_list[label_dict[x]][feature_cols]

        X_ = X_series.values.reshape((len(X_series),3))
        y_ = np.array([label_dict[x] for i in range(len(df_list[label_dict[x]]))])
        ts_ = np.array(df_list[label_dict[x]]['timestamp'])
        hr_ = np.array(df_list[label_dict[x]]['HR'])

          # 'downstairs': 0,
          # 'sit': 1,
          # 'sleep': 2,
          # 'stand': 3,

        y_all.append(y_)
        ts_all.append(ts_)
        hr_all.append(hr_)

        if(count==0):
            X_all = X_
            count += 1

        else:
            X_all = np.vstack((X_all, X_))

    y_all = np.hstack(y_all)
    ts_all = np.hstack(ts_all)
    hr_all = np.hstack(hr_all)

    return np.array(X_all), np.array(y_all), np.array(ts_all), np.array(hr_all)


# In[ ]:


def get_sorted_data(X_i, y_i, ts_i, hr_i, subj_i):
    df_ = pd.DataFrame({
        'ID': subj_i,
        'timestamp': ts_i,
        'x': [x[0] for x in X_i],
        'y': [x[1] for x in X_i],
        'z': [x[2] for x in X_i],
        'HR': hr_i,
        'label': y_i
    })

    df_sorted = df_.sort_values(by=['timestamp'])

    cols = ['x','y','z']
    X_i = df_sorted[cols].values.tolist()
    y_i = df_sorted['label'].values.tolist()
    ts_i = df_sorted['timestamp'].values.tolist()
    hr_i = df_sorted['HR'].values.tolist()
    subj_i = df_sorted['ID'].values.tolist()

    return X_i, y_i, ts_i, hr_i, subj_i


# # Function Call *

# In[115]:


def load_all_data(all_subjects):
    itr = len(all_subjects)

    TRIAXIAL = 3
    itr_count = 0

    y_all = []
    subj_all = []
    ts_all = []
    hr_all = []
    df_all = pd.DataFrame()

    for idx in range(itr):

        subject_id = identify_subj_id(idx, all_subjects)
        print("Loading {0}'s data".format(subject_id))

        df_timer, rec_date, start_time, end_time = load_timer(subject_id)
        df1, ai1, ts, lb_color = ai(subject_id, rec_date, start_time, end_time, df_timer)

#         print(start_time, end_time)

        # get a list of dataframe in which there are 4 types of activity
        df_list, label_dict = group_dataframe(df1, df_timer)
        label_list = sorted(list(label_dict.keys()))

    #     plot_grouped_df(df_list, label_dict)
    #     plot_ai(df1, ts, lb_color)

        X_i, y_i, ts_i, hr_i = get_data(df_list, label_dict)
        subj_i = [subject_id for i in range(len(X_i))]

        X_i, y_i, ts_i, hr_i, subj_i = get_sorted_data(X_i, y_i, ts_i, hr_i, subj_i)

        if(idx==0):
            X_all = X_i
        else:
            X_all = np.vstack((X_all, X_i))

        y_all.append(y_i)
        subj_all.append(subj_i)
        ts_all.append(ts_i)
        hr_all.append(hr_i)
        df_all.append(df1)

    y_all = np.hstack(y_all)
    subj_all = np.hstack(subj_all)
    ts_all = np.hstack(ts_all)
    hr_all = np.hstack(hr_all)

    print("Finished loading")

    return df_all, X_all, y_all, subj_all, ts_all, hr_all
