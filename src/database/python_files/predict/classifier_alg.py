#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# In[1]:

from tqdm import tqdm
from detect_peaks import detect_peaks


# # Classification Algorithms

# ## Classify walk

# In[11]:


def fluc(X,axis,rs,rf,diff=0.15):
    cnt = 0
    mult = 0
    j = 1

    i = rs
    while(i+j<rf):
        if(abs(X[i+j][axis]-X[i][axis])>diff):
            if(cnt==0):
                if(X[i+j][axis]>X[i][axis]):
                    mult = 1
                else:
                    mult = -1
                cnt += 1
            else:
                if((X[i+j][axis]>X[i][axis] and mult==-1) or 
                   (X[i+j][axis]<X[i][axis] and mult==1)):
                    mult *= -1
                    cnt += 1
            i += j-1  
        else:
            if(cnt==0):
                if(X[i+j][axis]>X[i][axis]):
                    mult = 1
                else:
                    mult = -1
        i += 1

#     print(rf, cnt)
    return cnt


# In[1]:


def calc_walk(X, step=10):
    s = 0
    f = len(X)

    thres = 2

    walk = [set(),set(),set()]
    arr = [[],[],[]]
    diff_arr = [0.28, 0.18, 0.18]
#     diff_arr = [0.19,0.13,0.1]
    axes = 3
    streak = 6
        
    for i in range(s,f,step):
        for j in range(axes):
            if(i+step<f):
                arr[j].append(fluc(X,j,i,i+step,diff=diff_arr[j]))
            else:
                arr[j].append(fluc(X,j,i,f,diff=diff_arr[j]))
            if(len(arr[j])>=streak):
                walk_true = True
                for k in range(-streak,0):
                    if(arr[j][k]<thres):
                        walk_true = False
                        break
                
                if(walk_true==True):
                    for k in range(streak):
                        walk[j].add(i-k*step)

    for i in range(axes):
        walk[i] = list(walk[i])
        walk[i] = sorted(walk[i])

    return walk


# In[6]:


def intersection(lst1, lst2): 
    return [value for value in lst1 if value in lst2] 


# In[7]:


def intersection_walk(walk):
  
    walk_its = intersection(walk[0],walk[1])
    walk_its = intersection(walk_its,walk[2])

    return walk_its


# In[9]:

class period:
    def __init__(self,s,f):
        self.s = s
        self.f = f

def calc_walk_periods(walk_its, step=10):
    walk_its_p = []
    if(len(walk_its)!=0):
        kp = walk_its[0]

        for i in range(len(walk_its)):
            if(i+1<=len(walk_its)-1 and walk_its[i+1]-walk_its[i]>step):
                walk_its_p.append(period(kp,walk_its[i]))
                kp = walk_its[i+1]

            if(i==len(walk_its)-1 and walk_its[i]-walk_its[i-1]==step):
                walk_its_p.append(period(kp,walk_its[i]))

        #   print(len(walk_its_p))

        #   for x in walk_its_p:
        #       print(x.s,x.f)

    return walk_its_p


# ## Walk

# In[1]:


def get_exact_walk(y):
    walk_lbl = 3
    
#     walk_chk = np.array((y-walk_lbl)==0)
    walk_chk = np.array(y==walk_lbl)
    
    walk_ts = np.array([i for i in range(len(walk_chk)) if walk_chk[i]==True])
    
    walk_keep = walk_ts[0]
    walk_prd = []     # walk period
    
    for i in range(len(walk_ts)):
        if(walk_ts[i]-walk_ts[i-1]>1):
            walk_prd.append(period(walk_keep,walk_ts[i-1]))
            walk_keep = walk_ts[i]
        
    return walk_prd


# In[38]:


def binarize_walk_prd(walk_prd, y):
    walk_bin = np.zeros(len(y))
    
    for w in walk_prd:
        for i in range(w.s, w.f+1):
            walk_bin[i] = 1
    
    return walk_bin


# ## Classify inactive

# In[9]:


def calc_inactive(ai1):
    # Slide windows by 1
    # Each window length is at least 50
    ki = -1
    kj = -1
    s = []

    SD_THRES = 0.002
    WINDOW_LEN = 50

    for i in tqdm(range(0,len(ai1)-1)):
        if(not(len(s)!=0 and s[-1].s==ki and s[-1].f==kj)):
            s.append(period(ki,kj))
        for j in range(1,len(ai1)):
            if(j-i>=WINDOW_LEN):
                if(np.std(ai1[i:j])<SD_THRES):
                    kj = j
                    ki = i

    return s                  


# In[10]:


def get_sequence(s):
  
    seq = []
    for i in range(len(ai1)):
        seq.append(i)

    sequence = []
    for x in s:
        if(x.s in seq and x.f in seq):
            for i in range(x.s,x.f+1):
                seq.remove(i)
            sequence.append(x)

    #   for x in sequence:
    #       print(x.s,x.f)

    return sequence


# # Show Plots of Classification

# ## Walking

# In[12]:


def plot_walk(df1, ai1, walk_its_p):
    x_axis = []
    y_axis = []

    p_idx = 0

    fig = plt.figure(figsize=(16,6))

    # print(len(walk_its_p))
    for i in range(len(df1)):
        if(p_idx<len(walk_its_p)):
            if(i>=walk_its_p[p_idx].s and i<=walk_its_p[p_idx].f):
            #         print('r',i)
                x_axis.append(i)
                y_axis.append(ai1[i])

            if(i==walk_its_p[p_idx].f):
                plt.plot(x_axis,y_axis,color='r')
                x_axis = []
                y_axis = []
                p_idx += 1

        if(p_idx<len(walk_its_p)):
            if(i<=walk_its_p[p_idx].s):
            #         print('b++',i)
                x_axis.append(i)
                y_axis.append(ai1[i])

            if(i==walk_its_p[p_idx].s):
                plt.plot(x_axis,y_axis,color='b')
                x_axis = []
                y_axis = []

        else:
            x_axis.append(i)
            y_axis.append(ai1[i])
            if(i==len(df1)-1):
                plt.plot(x_axis,y_axis,color='b')


    fig.savefig(basepath + 'Graphs/' + subject_id + '/' + subject_id + '_ddc_walking.png', dpi=300) 
    #   plt.show()
    plt.close(fig)


# ## Inactive

# In[11]:


def plot_inactive(df1, ai1, sequence):
    x_axis = []
    y_axis = []
    active = set()
    seq_idx = 0
    idx = 1 + np.std(ai1)

    fig = plt.figure(figsize=(16,6))

    for i in range(len(ai1)):
        if(seq_idx<len(sequence) and i>=sequence[seq_idx].s and i<=sequence[seq_idx].f):
        #         print('r',i)
            x_axis.append(i)
            y_axis.append(ai1[i])
            active.add(ai1[i])

        if(seq_idx<len(sequence) and i==sequence[seq_idx].f):
            plt.plot(x_axis,y_axis,color='r')
            x_axis = []
            y_axis = []
            seq_idx += 1

        if(seq_idx<len(sequence) and i<=sequence[seq_idx].s):
        #         print('b++',i)
            x_axis.append(i)
            y_axis.append(ai1[i])

        if(seq_idx<len(sequence) and i==sequence[seq_idx].s):
            plt.plot(x_axis,y_axis,color='b')
            x_axis = []
            y_axis = []


    x_axis = []
    y_axis = []

    p_idx = 0
    for i in range(len(df1)):
        if(p_idx<len(walk_its_p)):
            if(i>=walk_its_p[p_idx].s and i<=walk_its_p[p_idx].f):
            #         print('r',i)
                x_axis.append(i)
                y_axis.append(ai1[i])

            if(i==walk_its_p[p_idx].f):
                plt.plot(x_axis,y_axis,color='b')
                x_axis = []
                y_axis = []
                p_idx += 1

        if(p_idx<len(walk_its_p)):
            if(i<=walk_its_p[p_idx].s):
            #         print('b++',i)
                x_axis.append(i)
                y_axis.append(ai1[i])

            if(i==walk_its_p[p_idx].s):
                # color some red?
                x_axis = []
                y_axis = []

        else:
            x_axis.append(i)
            y_axis.append(ai1[i])
            if(i==len(df1)-1):
                plt.plot(x_axis,y_axis,color='r')

    fig.savefig(basepath + 'Graphs/' + subject_id + '/' + subject_id + '_inactive.png', dpi=300)          
    #   plt.show()
    plt.close(fig)


# # Combine SVM and Walk Classifier

# In[3]:


def get_inverse_X(X):
    win_length = 180
    new_X = []
    
    for X_i in X:
        acc_x = np.array(X_i[0:int(win_length/3)])
        acc_y = np.array(X_i[int(win_length/3):int(win_length*2/3)])
        acc_z = np.array(X_i[int(win_length*2/3):win_length])
        
        new_X_i = np.vstack((acc_x, acc_y, acc_z))
        new_X_i = np.transpose(new_X_i)
        
        new_X.append(new_X_i)
            
    return np.array(new_X)


# In[15]:


def classify_walk(new_X, WALK_LABEL = 3):
    y_pred_walk = []
    
    WIN_LEN_S = 0
    WIN_LEN_F = 50

    for X_i in new_X:
        walk3 = calc_walk(X_i)
        walk3_its = intersection_walk(walk3)
        walk_p = calc_walk_periods(walk3_its)
        
        if(len(walk_p)!=0 and walk_p[0].s==WIN_LEN_S and walk_p[0].f==WIN_LEN_F):
            y_pred = WALK_LABEL
        else:
            y_pred = 0
    
        y_pred_walk.append(y_pred)
    
    return np.array(y_pred_walk)


# In[1]:


def classify_walk_2(xyz_pca):

    window_length = 60   # 3 sec/0.16 sec = 18.75 time point
    one_sec = 6     # 1 sec/0.16 sec = 6.25 time point

    cols = [0,1,2]
    threshold = [0.05, 0.035, 0.04]
    exceed_threshold = 4
    
    walk_label = 3
    nonwalk_label = 0

    walk_pred = [[],[],[]]

    for k in range(len(xyz_pca)):
        
        for cl in range(len(cols)):
            c = cols[cl]

#             for i in range(0, len(xyz_pca[k])):
            xyz_pca_k_c = np.transpose(xyz_pca[k])[c]

            peak_idx = detect_peaks(xyz_pca_k_c)    
            valley_idx = detect_peaks(xyz_pca_k_c, valley=True)

            peak_point = [xyz_pca_k_c[j] for j in peak_idx]    
            valley_point = [xyz_pca_k_c[j] for j in valley_idx]

            min_length = min(len(peak_idx), len(valley_idx))

            diff_peak_valley = [np.abs(peak_point[i] - valley_point[i]) for i in range(min_length)]
            diff_peak_valley = np.array(diff_peak_valley)

            exceed = len(diff_peak_valley[diff_peak_valley>=threshold[cl]])
#             print(exceed)
            if(exceed>=exceed_threshold):
                walk_pred[cl].append(walk_label)
            else:
                walk_pred[cl].append(nonwalk_label)
                
    
    walk_pred_ax = np.array(walk_pred)
    walk_pred_t = walk_pred_ax.transpose()
    
#     print(walk_pred_ax)

    walk_pred = []

    for w in walk_pred_t:
        walk_bool = walk_label
        
        for i in range(3):
            if(w[i]!=walk_label):
                walk_bool = nonwalk_label
                break

        walk_pred.append(walk_bool)
        
    return walk_pred


# In[16]:


def combine(X_test, y_pred_svm, WALK_LABEL = 3):
    X_test_new = get_inverse_X(X_test)
    y_pred_walk = classify_walk(X_test_new, WALK_LABEL)
    
    y_pred_new = []

    for i in range(len(y_pred_walk)):
        if(y_pred_svm[i]!=WALK_LABEL and y_pred_walk[i]==WALK_LABEL):
            y_pred_new.append(y_pred_walk[i])

        else:
            y_pred_new.append(y_pred_svm[i])
            
    return y_pred_new


# In[2]:


def combine_2(xyz_, y_pred_svm, WALK_LABEL = 3):
    y_pred_walk = classify_walk_2(xyz_)
    
    y_pred_new = []

    for i in range(len(y_pred_walk)):
        if(y_pred_svm[i]!=WALK_LABEL and y_pred_walk[i]==WALK_LABEL):
            y_pred_new.append(y_pred_walk[i])

        else:
            y_pred_new.append(y_pred_svm[i])
            
#     print(y_pred_walk)
            
    return y_pred_new

