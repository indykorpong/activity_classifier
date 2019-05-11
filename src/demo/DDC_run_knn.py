import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import pickle

from os import listdir, walk
from os.path import isfile, join

get_ipython().run_line_magic('run', 'load_dataset.ipynb')

all_subjects = ['1001','1002','1003','1004','1005','1006','1007','1008','1009','1010','1011','1012','1013','1014',]


# In[57]:


X_all, y_all, subj_all, ts_all, hr_all = load_all_data(all_subjects)


# # Data Preprocessing

# In[58]:


get_ipython().run_line_magic('run', 'preprocessing.ipynb')


# In[59]:


print(X_all.shape, y_all.shape)


# In[60]:


X_all


# # Group Data by Label and Normalize Data

# In[61]:


print(label_list)
print(label_dict)


# In[62]:


new_label_list = [0,1,2,3]
new_label_dict = {
    0: 'sit',
    1: 'sleep',
    2: 'stand',
    3: 'walk'
}

colors = ['r','g','b','navy','turquoise','darkorange']


# ## Show Plot for each Activity and Subject

# In[63]:


# group X_all and y_all from load_dataset.ipynb by labels
#X_label, y_label = label_grouping(X_all, y_all, subj_all, new_label_list)

# normalize X_label
#X_norm = normalize_data(X_label)


# In[64]:


#plot_all_label(X_label, y_all, new_label_list, new_label_dict)


# # Reshape Data (Pure Label)

# In[67]:


# get label-separated X and y
X_svm, y_svm = prepare_pure_label(X_all, y_all, subj_all, all_subjects, new_label_list)
y_svm = y_svm.reshape((y_svm.shape[0],))

X_pure, y_pure = prepare_pure_label(X_all, y_all, subj_all, all_subjects, new_label_list)
y_pure = y_pure.reshape((y_svm.shape[0],))


# In[68]:


print(X_svm.shape, y_svm.shape)


# # Reshape Data (Impure Label)

# In[69]:


X_impure, y_impure = prepare_impure_label(X_all, y_all)


# In[70]:


print(X_impure.shape, y_impure.shape)


# # Split Train and Test Set

# In[71]:


# Split training data and testing data
X_train, X_test, y_train, y_test = train_test_split(X_svm, y_svm, test_size=0.2, random_state=42)

X_tr, X_te, y_tr, y_te = train_test_split(X_impure, y_impure, test_size=0.2, random_state=42)

print(X_tr.shape)
print(X_te.shape)


# In[72]:


print(X_train.shape)
print(X_test.shape)


# # K-nearest neighbors

# In[73]:


knn_model = nn_classifier(X_train, y_train)
print("Finished training")


# In[74]:


filename = '../../model/knn_model.pkl'
pickle.dump(knn_model, open(filename, 'wb'))


# In[75]:


y_pred = knn_model.predict(X_test)


# In[76]:


knn_model_2 = nn_classifier(X_tr, y_tr)
print("Finished training")


# In[77]:


y_pred_2 = knn_model_2.predict(X_te)


# ## KNN Evaluation

# In[78]:


get_ipython().run_line_magic('run', 'eval_score.ipynb')


# In[79]:


LABELS = ['sit','sleep','stand','walk']


# In[80]:


acc = accuracy_score(y_test, y_pred)
print(acc)

show_conf_matrix(y_test, y_pred, LABELS)
show_clf_report(y_test, y_pred, LABELS)


# # Plot data on training and test sets

# In[97]:


get_ipython().run_line_magic('run', 'test_model.ipynb')


# In[94]:


filename = '../../model/knn_model.pkl'
model = pickle.load(open(filename,'rb'))
label_list = ['sit', 'sleep', 'stand', 'walk']

for subject_id in all_subjects:
    call_functions(subject_id, model, label_list)


# In[98]:


call_functions('1015', model, label_list)


# # Do not use cells below

# # Walk Algorithm

# In[ ]:


walk_exact = []
walk_label = 3
window_length = 60

for i in range(window_length, len(y_all)):
    if(y_all[i]==walk_label):
        walk_exact.append(walk_label)
    else:
        walk_exact.append(0)


# ## Walk Algorithm Evaluation

# In[ ]:


get_ipython().run_line_magic('run', 'preprocessing.ipynb')


# In[ ]:


X_pure, y_pure = prepare_pure_label(X_all, y_all, subj_all, all_subjects, new_label_list)
X_impure, y_impure = prepare_impure_label(X_all, y_all)


# In[ ]:


print(np.array(X_impure).shape, np.array(y_impure).shape)
print(np.array(X_all).shape)


# In[ ]:


X_test_inv = inverse_segmentation(X_test)


# In[ ]:


def get_df_walk(X):
    df_walk = pd.DataFrame({
        'x': [a[0] for a in X],
        'y': [a[1] for a in X],
        'z': [a[2] for a in X]
    })
    
    return df_walk


# In[61]:


df_walk = get_df_walk(X_all)


# In[69]:


get_ipython().run_line_magic('run', 'classifier_algo.ipynb')


# In[ ]:


# walk_pred = 
classify_walk_2(df_walk, 4)
# print(walk_pred)


# In[40]:


print(X_test_inv.shape)


# In[42]:


walk_true = []

walk_label = 3
nonwalk_label = 0

for i, y_i in enumerate(y_all):
    if(y_i==walk_label):
        walk_true.append(y_i)
    else:
        walk_true.append(nonwalk_label)

acc = accuracy_score(walk_true, walk_pred)
print(acc)

walk_lbl = ['NaN', 'walk']

show_conf_matrix(walk_true, walk_pred, walk_lbl)
show_clf_report(walk_true, walk_pred, walk_lbl)


# # Combine KNN and Walk Algorithm
# #### Before Post Process

# In[33]:


get_ipython().run_line_magic('run', 'preprocessing.ipynb')


# In[34]:


X_test_i = inverse_segmentation(X_test, y_pred.shape[0])


# In[35]:


print(X_test_i.shape, y_pred.shape)


# In[36]:


y_pred_svm = knn_model.predict(X_impure)
print("Finished predicting")


# In[37]:


y_pred_new = combine_2(X_pca, y_all, exceed_threshold=2)


# In[38]:


y_pred_new = np.array(y_pred_new)


# In[39]:


y_pred_fill = np.hstack(([y_pred_svm[0] for i in range(window_length-1)], y_pred_svm))


# In[40]:


acc = accuracy_score(y_pred_fill, y_pred_new)
print(acc)

show_conf_matrix(y_pred_fill, y_pred_new, LABELS)
show_clf_report(y_pred_fill, y_pred_new, LABELS)


# # Test Model with Some Subjects

# In[32]:


filename = '../../model/knn_model.pkl'

model = pickle.load(open(filename,'rb'))


# In[33]:


label_list = ['sit', 'sleep', 'stand', 'walk']


# In[37]:


get_ipython().run_line_magic('run', 'test_model.ipynb')


# #### After Post-process

# In[39]:


unseen_patients = [str(i) for i in range(3001,3006)]

for subject_id in unseen_patients:
    df_acc_label, true_periods, pred_periods, iou = call_functions(subject_id, model, label_list)


# In[40]:


unseen_patients = [str(i) for i in range(3001,3006)]

for subject_id in unseen_patients:
    df_acc_label, true_periods, pred_periods, iou = call_functions(subject_id, pca, model, label_list)


# # End of today's progress

# In[ ]:


all_subjects = [str(i) for i in range(3001,3006)]

call_functions(all_subjects,pca)


# In[ ]:


s = '3004'
df_sid = load_actual_timer(s)
df_test = load_data(s, df_sid)

X_vis_imp, ts_list_imp = preprocess_data(df_test, pca)
df_y = predict(X_vis_imp, ts_list_imp)

df_test, df_y = prepare_actual_lb(df_test, df_y, df_sid)

actual_periods = get_actual_periods(df_test)
pred_periods = get_predicted_periods(df_y)
pp_all_run = postprocess_predicted(pred_periods, df_y)

df_y['y_pred'] = pd.Series(pp_all_run)
pp_periods = get_predicted_periods(df_y)
plot_highlighted(s, df_test, pp_periods, actual_periods)


# In[ ]:


print(df_y.shape, df_test.shape)


# In[ ]:





# # Display Table

# In[ ]:


from prettytable import PrettyTable


# In[ ]:


print(s)
print(df_y)


# In[ ]:


label_period = []
period_list = [[] for i in range(len(LABELS))]

first = 0
keep = 0

for i in range(len(df_y)):
    if(calc_sec(df_y.loc[i]['timestamp'].split(' ')[1])>=calc_sec(df_sid.loc[0]['timestamp'].split(' ')[1]) and
       calc_sec(df_y.loc[i]['timestamp'].split(' ')[1])<=calc_sec(df_sid.loc[len(df_sid)-1]['timestamp'].split(' ')[1])):
        
        keep_lb = df_y.loc[keep]['y_pred']

        if(keep_lb!=df_y.loc[i]['y_pred']):
            label_period.append([df_y.loc[keep]['timestamp'], df_y.loc[i-1]['timestamp'], 
                                 df_y.loc[i-1]['y_pred']])

            period_list[df_y.loc[i-1]['y_pred']].append([df_y.loc[keep]['timestamp'], df_y.loc[i-1]['timestamp']])

            keep = i


# In[ ]:


labels_list = ['sit', 'sleep', 'stand', 'walk']
headers = ['start', 'end', 'pred']

t = PrettyTable(headers)

for row in label_period:
#     if(calc_sec(row[1].split(' ')[1])-calc_sec(row[0].split(' ')[1])>1):
    t.add_row([row[0], row[1], labels_list[row[2]]])


# In[ ]:


print(t)


# In[ ]:


print(len(label_period))

label_cnt_list = [0 for i in range(len(labels_list))]
for lb_p in label_period:
    label_i = lb_p[2]
    
    label_cnt_list[label_i] += 1

activity_changes = []
for i in range(len(labels_list)):
    activity_changes.append([labels_list[i], label_cnt_list[i]])
    
print(activity_changes)


# In[ ]:


headers = ['Label', 'Activities Count']
tabl_act_chng = PrettyTable(headers)

for ac in activity_changes:
    tabl_act_chng.add_row([ac[0], ac[1]])
    
tabl_act_chng.add_row(['', ''])
tabl_act_chng.add_row(['total changes', len(label_period)])


# In[ ]:


print(tabl_act_chng)


# ## Active Inactive AC (ALL)

# In[ ]:


headers = ['Label', 'Activities Count']
tabl_act = PrettyTable(headers)
inactive_table = []
active_table = []

sum = 0
sum_2 = 0
for ac in activity_changes:
    if(ac[0] == 'sit' or ac[0] == 'sleep'):
        sum += ac[1]
    else :
        sum_2 += ac[1]

tabl_act.add_row(['Inactive', sum])
tabl_act.add_row(['Active', sum_2])


# In[ ]:


print(tabl_act)


# # Convert Time to String Method

# In[ ]:


def convert_time_to_string(sec):
    minute = math.floor(sec/60)
    sec = int(sec%60)

    time_string = str(minute) + ':' + str(sec)
    if(sec<10):
        time_string = str(minute) + ':0' + str(sec)
    
    return time_string


# # Predicted Duration

# In[ ]:


total_secs = []
for i in range(len(period_list)):    
    secs = 0
    for p_i in period_list[i]:
        sec = calc_sec(p_i[1].split(' ')[1]) - calc_sec(p_i[0].split(' ')[1])
        secs += sec
    
    secs = round(secs, 3)
    total_secs.append(secs)
    
percent_secs = [round(t/np.sum(total_secs)*100, 3) for t in total_secs]

tb = PrettyTable(['Label', 'Minutes', 'Percentage', 'Activity Count'])

for i in range(len(LABELS)):
    tb.add_row([labels_list[i], convert_time_to_string(total_secs[i]), percent_secs[i], label_cnt_list[i]])

tb.add_row(['', '', '',''])
tb.add_row(['total', convert_time_to_string(round(np.sum(total_secs), 3)), 
            round(np.sum(percent_secs), 3), len(label_period)])


# # Actual Duration

# In[ ]:


df_lb = df_sid.groupby('label')

dura_dict = {}
for lb in labels_list:
    dura_dict[lb] = 0

idx = list(df_sid.index)
for i in range(len(labels_list)):
    lb = labels_list[i]
    df_temp = df_lb.get_group(lb)
    df_temp = df_temp.reset_index(drop=True)
        
    if(lb=='downstairs' or lb=='upstairs'):
        lb = 'walk'
    
    for j in range(len(df_temp)):
        dura_dict[lb] += calc_sec(df_temp.loc[j]['duration'])
        
total_dura = np.sum([dura_dict[lb] for lb in labels_list])

percent_list = []
        
tabl = PrettyTable(['Label', 'Minutes', 'Percentage'])
for lb in labels_list:
    percent = round(dura_dict[lb]/total_dura*100, 3)
    tabl.add_row([lb, convert_time_to_string(dura_dict[lb]), round(dura_dict[lb]/total_dura*100, 3)])
    
    percent_list.append(percent)
    
tabl.add_row(['', '', ''])    
tabl.add_row(['total', convert_time_to_string(total_dura), round(np.sum(percent_list), 3)])


# # Activity Durations Table

# In[ ]:


print('Prediction')
print(tb)

print('Actual')
print(tabl)


# # Bar Chart for Every 5 Minutes

# In[ ]:


s_idx = 0
f_idx = 1
lb_idx = 2


# ## Separate Each 5 Minutes

# In[ ]:


fivemin = 60*5
new_label_period = []

start_time = calc_sec(label_period[0][s_idx].split(' ')[1])
finish_time = calc_sec(label_period[-1][f_idx].split(' ')[1])

floor_start = start_time - (start_time%fivemin)
ceil_finish = finish_time - (finish_time%fivemin) + fivemin

print(calc_ts(floor_start), calc_ts(ceil_finish))

tm_s = floor_start
tm_f = floor_start + fivemin
date = label_period[0][s_idx].split(' ')[0]

for prd in label_period:
    if(calc_sec(prd[f_idx].split(' ')[1])>=tm_f):
        new_prd = [prd[s_idx], date + ' ' + calc_ts(tm_f), prd[lb_idx]]
        new_label_period.append(new_prd)
        
        tm_s += fivemin
        tm_f += fivemin
    else:
        new_label_period.append(prd)
                
    if(calc_sec(prd[s_idx].split(' ')[1])<tm_s):
        new_prd = [date + ' ' + calc_ts(tm_s), prd[f_idx], prd[lb_idx]]
        new_label_period.append(new_prd)


# In[ ]:


all_periods_label = []

for t_i in range(int(floor_start), int(ceil_finish), fivemin):
    period_lb = [0 for i in range(len(LABELS))]
    for prd in new_label_period:
        if(calc_sec(prd[s_idx].split(' ')[1])>=t_i and calc_sec(prd[f_idx].split(' ')[1])<=t_i+fivemin):
            period_lb[prd[lb_idx]] += calc_sec(prd[f_idx].split(' ')[1])-calc_sec(prd[s_idx].split(' ')[1])
            period_lb[prd[lb_idx]] = round(period_lb[prd[lb_idx]], 3)
            
    all_periods_label.append(period_lb)


# In[ ]:


df_all = pd.DataFrame(all_periods_label, columns=labels_list)


# ## Plot Bar Graph

# In[ ]:


pos = list(range(len(df_all['sit'])))
width = 0.2
colors = ['crimson','gold','lime','dodgerblue']

fig, ax = plt.subplots(figsize=(10,5))

for i in range(len(LABELS)):
    plt.bar([p + i*width for p in pos],
            df_all[labels_list[i]],
            width,
            alpha=0.5,
            color=colors[i],
            label=labels_list[i])
    
ax.set_xticks([p + 1.5 * width for p in pos])

xtick_labels = [calc_ts(floor_start + i*fivemin) + '-' + calc_ts(floor_start + (i+1)*fivemin)
                for i in range(len(df_all))]
ax.set_xticklabels(xtick_labels)

ax.set_ylabel('Time (sec)')

plt.xlim(min(pos)-width, max(pos)+width*4)
plt.legend(loc='upper left')
plt.title('Activity Summary for Subject ID: ' + s)

plt.show()


# In[ ]:




