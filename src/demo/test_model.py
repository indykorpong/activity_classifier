import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

from prettytable import PrettyTable
from os import listdir
from sklearn.metrics import f1_score
from datetime import datetime


# In[6]:


get_ipython().run_line_magic('run', 'preprocessing.ipynb')
get_ipython().run_line_magic('run', 'eval_score.ipynb')


# In[7]:


mypath = '../../DDC_Data/raw/'
basepath = '../../'


# # Get the Actual Timestamp Labels

# In[8]:


def load_timer(subject_id):
    
    sid_dir = mypath + subject_id
    sid_files = [f for f in listdir(sid_dir) if 'history_amdtimer' in f]

    sid_filepath = sid_dir + '/' + sid_files[0]

    timer_df = pd.read_csv(sid_filepath, header=None, names=['sid','raw_label', 'timestamp', 'duration','label'])

    filtered_timer = [i for i in timer_df['sid'] if i==int(subject_id)]

    timer_filt = timer_df[timer_df['sid'].isin(filtered_timer)]
    timer_filt = timer_filt.reset_index(drop=True)
    
    timer_label = []
    
    for i in range(len(timer_filt)):
        if(timer_filt.loc[i]['raw_label']=='upstairs' or 
          timer_filt.loc[i]['raw_label']=='downstairs'):
            timer_label.append('walk')
        else:
            timer_label.append(timer_filt.loc[i]['raw_label'])

    timer_filt['label'] = pd.Series(timer_label)
    
    datetime_format = '%Y-%m-%d %H:%M:%S.%f'
    timer_filt['time_start'] = timer_filt['timestamp'].apply(lambda x: datetime.strptime(x, datetime_format))
    
    time_format = '%H:%M:%S'
    zero_date = datetime(1900, 1, 1)
    
    timer_filt['duration'] = timer_filt['duration'].apply(lambda x: datetime.strptime(x, time_format)-zero_date)
    
    for i in range(timer_filt.shape[0]):
        timer_filt.loc[i, 'time_end'] = timer_filt.loc[i, 'time_start'] + timer_filt.loc[i, 'duration']

#     print(timer_filt)
    
    return timer_filt


# # Load Data of the Subject

# In[9]:


def load_acc(subject_id, start_time, end_time):
    # Load accelerations
    acc_path = mypath + '/' + subject_id + '/' + subject_id + '-log_acc.csv'

    df = pd.read_csv(acc_path, header=None, names=['x','y','z','timestamp'])
    
    datetime_format = '%Y-%m-%d %H:%M:%S.%f'
    df['timestamp'] = df['timestamp'].apply(lambda x: datetime.strptime(x, datetime_format))

    filtered = [r for r in df['timestamp'] if r>=start_time and r<=end_time]

    df_filt = df[df['timestamp'].isin(filtered)]
    df_filt = df_filt.reset_index(drop=True)

    df_filt['ID'] = pd.Series([subject_id for i in range(len(df_filt))])
    
    cols = ['ID','timestamp','x','y','z']
    df_filt = df_filt[cols]

    return df_filt


# In[10]:


def load_data(test_subj, df_sid):
    filepath = mypath + test_subj + '/' + test_subj + '-log_acc.csv'

    df_test = pd.read_csv(filepath, header=None, names=['x','y','z','timestamp'])

    test_filt = [i for i in range(len(df_test)) 
                 if calc_sec(df_test.loc[i]['timestamp'].split(' ')[1])<=calc_sec(df_sid.loc[len(df_sid)-1]['finish']) 
                 and calc_sec(df_test.loc[i]['timestamp'].split(' ')[1])>=calc_sec(df_sid.loc[0]['start'])]

    df_test_filt = df_test[df_test.index.isin(test_filt)]
    df_test = df_test_filt.reset_index(drop=True)
    
    return df_test


# # Preprocess (PCA, impure)

# In[11]:


def preprocess_data(df_test):
    ts_list = []
    g = 9.8

    X_list = []
    
    for i in range(len(df_test)):
        X_i = [df_test.loc[i]['x']/g, df_test.loc[i]['y']/g, df_test.loc[i]['z']/g]
        X_list.append(X_i)
        
    X_stack = np.vstack(X_list)
    X_norm = MinMaxScaler().fit_transform(X_stack)
#     X_pca = pca.transform(X_norm)

    y_imp = [-1 for i in range(X_norm.shape[0])]
    X_imp, y_imp = prepare_impure_label(X_norm, y_imp)
    
    return X_imp, y_imp


# # Predict

# In[12]:


get_ipython().run_line_magic('run', 'classifier_algo.ipynb')


# In[13]:


def predict_combine(X_imp, model, window_length=60):
    
    y_pred = model.predict(X_imp)
    print("Finished prediction")
    
#     y_pred = combine_2(X_imp, y_pred)
    y_pred_fill = np.hstack(([y_pred[0] for i in range(window_length-1)], y_pred))
    
#     print(X_imp.shape, y_pred_fill.shape)
    
    return y_pred_fill


# # Group dataframe by label

# In[14]:


def group_dataframe_by_label(df1, df_timer, subject_id, label_list):
    df_list = {}
    period = {}
    df2 = df1.copy()
    
    for label in label_list:
        df_list[label] = pd.DataFrame()
        period[label] = []
    
    for label in label_list:
#         print(label)
        for i in range(df_timer.shape[0]):
            start = 0
            end = 0
            
            if(df_timer.loc[i, 'label']==label):
                t_a = df_timer.loc[i, 'time_start']
                t_b = df_timer.loc[i, 'time_end']

                for j in range(df1.shape[0]):    
                    if(df1.loc[j, 'ID']==subject_id):
                        if(j>0 and df1.loc[j, 'timestamp']<=t_b and df1.loc[j-1, 'timestamp']<t_b):
                            end = j

                for j in reversed(range(df1.shape[0])):
                    if(df1.loc[j, 'ID']==subject_id):
                        if(j<df1.shape[0]-1 and df1.loc[j, 'timestamp']>=t_a and df1.loc[j+1, 'timestamp']>t_a):
                            start = j
                            
                for j in range(start, end+1):
                    df2.loc[j, 'label'] = label

                period[label].append([start, end])
                
                if(df_list[label].empty):
                    df_list[label] = df1.loc[start:end+1]
                else:
                    df_list[label].append(df1.loc[start:end+1], ignore_index=True)
                    
    for label in label_list:
        df_list[label] = df_list[label].reset_index(drop=True)

    return df_list, df2, period


# # Prepare Predicted Labels

# In[15]:


def get_periods_from_list(y_pred, label_list):
    
    pred_periods = [[] for i in range(len(label_list))]

    keep = 0

    for i in range(len(y_pred)):
        keep_lb = y_pred[keep]

        if(keep_lb!=y_pred[i]):
            
            if(y_pred[i]!=None):
                pred_periods[y_pred[i-1]].append([keep, i-1])               

            keep = i

        elif(i==len(y_pred)-1):

            if(y_pred[i]!=None):
                pred_periods[y_pred[i-1]].append([keep, i]) 

    pred_periods = np.array(pred_periods)
    
    return pred_periods


# In[16]:


def postprocess_predicted(pred_periods, y_length):
    onesec = 1  # 1 sec.
    T = 0.16    # T = 1/f

    pp_periods = []
    
    for pp in pred_periods:
        pp_i = pred_periods[pp]
        
        temp = []
        for p in pp_i:
            if(p[1]-p[0]>int(onesec*2*(1/T))):
                temp.append([p[0],p[1]])
                
        pp_periods.append(temp)

    pp_periods = np.array(pp_periods)
    
    other_label = -1
    all_run = [other_label for i in range(y_length)]

    for i in range(len(pp_periods)):
        for p in pp_periods[i]:
            for j in range(p[0],p[1]+1):
                all_run[j] = i

    for i in range(len(all_run)-1,0,-1):
        if(all_run[i-1]==other_label):
            all_run[i-1] = all_run[i]

    return all_run


# # Get sequence from periods

# In[17]:


def sequence_from_periods(periods, label_list):
    
    max_length = 0
    
    for label in label_list:
        if(len(periods[label])>0):
            periods_i = np.hstack(periods[label])
        
            if(max_length<max(periods_i)):
                max_length = max(periods_i)
    
    seq = ['' for i in range(max_length+1)]
    
    for label in label_list:
        for element in periods[label]:
            for i in range(element[0], element[1]+1):
                seq[i] = label
    
    return seq, max_length


# # Evaluation with IoU

# In[18]:


def evaluate_period(p1, p2, max_length, label_list):
    iou_all = []
    
    for lb in label_list:
        p1_onehot = []
        for i in range(max_length):
            if(p1[i]==lb):
                p1_onehot.append(1)
            else:
                p1_onehot.append(0)
                
        p2_onehot = []
        for i in range(max_length):
            if(p2[i]==lb):
                p2_onehot.append(1)
            else:
                p2_onehot.append(0)
                
        intersection = 0
        union = 0
        
        for i in range(max_length):
            if(p1_onehot[i]==1 and p2_onehot[i]==1):
                intersection += 1
            if(p1_onehot[i]==1 or p2_onehot[i]==1):
                union += 1
                
        iou_lb = intersection/union
        
        iou_all.append(iou_lb)
        
    return iou_all


# # Evaluation with F1 score

# In[52]:


def get_y_true(df2):
    filt = df2['label'].isnull()
    df2_filt = df2[filt].copy()
    idx_null = list(df2_filt.index)

    for i in idx_null:
        df2.loc[i, 'label'] = 'others'
    
    label_dict = {'sit': 0, 'sleep': 1, 'stand': 2, 'walk': 3, 'others':4}
    y_true = list(df2['label'].apply(lambda x: label_dict[x]))
    y_true = np.hstack(y_true)
    
    return y_true[1:], df2


# In[20]:


def evaluate_f1(y_test, y_pred, LABELS):
    show_conf_matrix(y_test, y_pred, LABELS)
    show_clf_report(y_test, y_pred, LABELS)


# # Plot Graph

# In[67]:


def plot_highlighted(test_subj, df_test, pred_periods, actual_periods, label_list):
    ax1 = df_test.plot(y=['x','y','z'], figsize=(10,3), color=['r','g','b'])

    color_list = {
        'sit': 'indianred',
        'sleep': 'khaki',
        'stand': 'lightgreen',
        'walk': 'skyblue'
    }

    for label in label_list:
        cnt = 0
        for item in pred_periods[label]:
            if(cnt==0):
                ax1.axvspan(item[0], item[1], color=color_list[label], label=label)
                cnt = 1
            else:
                ax1.axvspan(item[0], item[1], color=color_list[label])
            
    ax2 = df_test.plot(y=['x','y','z'], figsize=(10,3), color=['r','g','b'])

    for label in label_list:
        cnt = 0
        for item in actual_periods[label]:
            if(cnt==0):
                ax2.axvspan(item[0], item[1], color=color_list[label], label=label)
                cnt = 1
            else:
                ax2.axvspan(item[0], item[1], color=color_list[label])
    
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    
    ax1.set_title('Prediction for ' + test_subj)
    ax2.set_title('Actual for ' + test_subj)
    
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    
    fig1 = ax1.get_figure()
    fig2 = ax2.get_figure()
    
    plt.show()
    
    graph_path = '{}Graphs/{}/'.format(basepath, test_subj)
    if(not os.path.isdir(graph_path)):
        os.mkdir(graph_path)
    
    fig1.savefig(graph_path + test_subj + '-prediction.png', dpi=200)
    fig2.savefig(graph_path + test_subj + '-actual.png', dpi=200)
    
    plt.close(fig1)
    plt.close(fig2)


# # Function Call

# In[22]:


def call_functions(subject_id, model, label_list):
    
    print("Loading {0}'s data".format(subject_id))

    df_timer = load_timer(subject_id)
    
    start_time = df_timer.loc[0, 'time_start']
    end_time = df_timer.loc[df_timer.shape[0]-1, 'time_end']

    df_acc = load_acc(subject_id, start_time, end_time)

    X_impure, y_impure = preprocess_data(df_acc)
    y_pred = predict_combine(X_impure, model)
    
    df_acc_label, df2, true_periods = group_dataframe_by_label(df_acc, df_timer, subject_id, label_list)
    
    p_periods = get_periods_from_list(y_pred, label_list)
    pred_periods = {}

    for i in range(len(label_list)):
        pred_periods[label_list[i]] = p_periods[i]
    
    pp_all = postprocess_predicted(pred_periods, len(y_pred))
    
    p_true, len_true = sequence_from_periods(true_periods, label_list)
    p_pred, len_pred = sequence_from_periods(pred_periods, label_list)
    
    iou = evaluate_period(p_true, p_pred, len_pred, label_list)
    
    print('label:', label_list)
    print('iou:', iou)
    
    y_true, df2 = get_y_true(df2)
    other_list = ['sit','sleep','stand','walk','others']
    evaluate_f1(y_true, y_pred, other_list)
    
    plot_highlighted(subject_id, df_acc, pred_periods, true_periods, label_list)
    
    return df_acc_label, true_periods, pred_periods, iou

