#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[4]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


from sklearn.preprocessing import MinMaxScaler, label_binarize, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn import neighbors

from mpl_toolkits.mplot3d import Axes3D
from statsmodels.robust import mad
from scipy import stats


# # Calculate Roll, Pitch, Yaw

# In[1]:


def calc_rpy(y, colors):
    y_t = y.transpose()

    ax = np.array(y_t[0], dtype=np.float32)
    ay = np.array(y_t[1], dtype=np.float32)
    az = np.array(y_t[2], dtype=np.float32)

    rpy = []
    rpy_labels = ['pitch','roll','yaw']

    pitch = 180 * np.arctan(ax/np.sqrt(ay*ay + az*az))/math.pi
    rpy.append(pitch)

    roll = 180 * np.arctan(ay/np.sqrt(ax*ax + az*az))/math.pi
    rpy.append(roll)

    yaw = 180 * np.arctan(az/np.sqrt(ax*ax + ay*ay))/math.pi
    rpy.append(yaw)

    return roll, pitch, yaw


# # Normalize Data (Z-score)

# In[10]:

def normalize_data(X):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_norm = []

    for i in range(len(X_label)):
        X_lb = []
        for X_subj in X_label[i]:
            X_tp = np.array(X_subj).transpose()
            X_a = []
            for X_axis in X_tp:
                X_n = stats.zscore(X_axis)
                X_a.append(X_n)
            X_a = np.array(X_a).transpose()
            X_lb.append(X_a)

        X_norm.append(X_lb)

    return np.array(X_norm)


# # PCA, LDA, and SVD

def apply_pca(X, y, target_names):
    n_comp = 3

    pca = PCA(n_components=n_comp)
    X_r = pca.fit(X).transform(X)

    lw = 1
    fig = plt.figure(figsize=(15,6))
    ax = fig.add_subplot(111)

    for color, i, target_name in zip(colors, [0, 1, 2, 3, 4, 5, 6, 7], target_names):

        ax.scatter(X_r[np.where(y==i), 0], X_r[np.where(y==i), 1],
                   color=color, alpha=.8, linewidths=lw, label=target_name)
        ax.legend(loc='best', shadow=False, scatterpoints=1)

    plt.show()
    plt.close(fig)

    return np.array(X_r), pca


# In[14]:


def apply_lda(X, y, target_names):
    n_comp = 3

    lda = LinearDiscriminantAnalysis(n_components=n_comp)
    X_r = lda.fit(X, y).transform(X)
    #   print(X_r.shape)

    lw = 1
    fig, ax = plt.subplots(figsize=(10,6))

    for color, i, target_name in zip(colors, [0, 1, 2, 3, 4, 5], target_names):
        ax.scatter(X_r[np.where(y==i), 0], X_r[np.where(y==i), 1], color=color, alpha=.8, linewidths=lw,
                      label=target_name)
        ax.legend(loc='best', shadow=False, scatterpoints=1)

    plt.show()
    plt.close(fig)

    return np.array(X_r)


# In[15]:


def apply_svd(X, y, target_names):
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    return u, s, vh


# # Support Vector Classifier

def svm_classifier(X, y):
    clf = SVC(kernel='rbf', decision_function_shape='ovr', gamma='auto')
    clf.fit(X, y)

    return clf

# # K-Nearest Neighbors Classifier

def nn_classifier(X, y, k=6):
#   print(y.shape)
    neighbor = k
    clf = neighbors.KNeighborsClassifier(n_neighbors=neighbor, algorithm='ball_tree')
    clf.fit(X, y)

    return clf


# # Group X and y by Label and Subject ID

def label_grouping(X, y, subj, all_subjects, lb_list):
    X_label = []
    y_label = []

    for i in range(len(lb_list)):
        X_act = []
        y_act = []
        for s in range(len(all_subjects)):
            X_subj = []
            y_subj = []
            for j in range(len(y)):
                if(y[j]==lb_list[i]):
                    if(subj[j]==all_subjects[s]):
                        X_subj.append(np.array(X[j]))
                        y_subj.append(np.array(y[j]))

            X_act.append(X_subj)
            y_act.append(y_subj)

        X_label.append(X_act)
        y_label.append(y_act)

    return np.array(X_label), np.array(y_label)


# In[1]:


def plot_all_label(X, y, lb_list, lb_dict):
    TRIAXIAL = 3

    color_subj = ['darkred','crimson','orange','gold','yellow','peachpuff','lime','green','olive','aqua',
                  'paleturquoise','teal','dodgerblue','blue','navy','purple','lavender','magenta','mediumslateblue','pink']

    for i in range(len(lb_list)):
        f, ax = plt.subplots(nrows=1, ncols=TRIAXIAL, figsize=(15,5))

        print("plotting ", lb_dict[lb_list[i]])

        for j in range(len(all_subjects)):
            X_i_tp = np.array(X[i][j]).transpose()
            print(X_i_tp.shape)

            ax[0].plot(X_i_tp[0], color=color_subj[j], label='x')
            ax[0].set_title('X-axis for ' + lb_dict[lb_list[i]])

            ax[1].plot(X_i_tp[1], color=color_subj[j], label='y')
            ax[1].set_title('Y-axis for ' + lb_dict[lb_list[i]])

            ax[2].plot(X_i_tp[2], color=color_subj[j], label='z')
            ax[2].set_title('Z-axis for ' + lb_dict[lb_list[i]])

        plt.show()
        plt.close(f)


# # Make Overlapping Time Window

def make_overlapping(X, y, window_length = 60):
    length = X.shape[0]
    X_new = []
    y_new = []

    for i in range(length):
        X_temp = []
        for j in range(window_length):
            if(i+j<length):
                X_temp.append(X[i+j])

        if(i+window_length-1<length):
            X_new.append(X_temp)
            y_new.append(y[i+window_length-1])

    return np.array(X_new), np.array(y_new)


# # Concatenate X,Y,Z Axis into 1 Numpy Array

def concat_xyz(X):
    X_concat = []
    for X_i in X:
        X_tp = X_i.transpose()
        X_stack = np.hstack((X_tp[0],X_tp[1],X_tp[2]))
        X_concat.append(X_stack)

    return np.array(X_concat)

def concat_xyz_2(X):
    X_concat = []
    for X_i in X:
        X_tp = X_i.transpose()
        X_stack = []
        for i in range(X_tp.shape[1]):
            for j in range(3):
                X_stack.append(X_tp[j][i])

        X_stack = np.array(X_stack)
        X_concat.append(X_stack)

    return np.array(X_concat)


# # Concatenate X Axis-wise

def concat_label(X):
    X_concat = []
    for X_lb in X:
        X_temp = []
        for i in range(len(X_lb)):
            if(i==0):
                X_temp = X_lb[i]
            else:
                X_temp = np.vstack((X_temp, X_lb[i]))

        if(len(X_concat)==0):
            X_concat = X_temp
        else:
            X_concat = np.vstack((X_concat, X_temp))

    return np.array(X_concat)


# # Reshape Data (Pure Label)

def prepare_pure_label(X, y, subj_all, all_subjects, new_label_list):
    X_label, y_label = label_grouping(X, y, subj_all, all_subjects, new_label_list)

    X_concat = []
    y_concat = []
    for i in range(len(X_label)):
        for j in range(len(X_label[i])):
            X_ol, y_ol = make_overlapping(np.array(X_label[i][j]), y_label[i][j])

            if(len(X_concat)==0):
                X_concat = X_ol
            else:
                X_concat = np.vstack((X_concat, X_ol))

            if(len(y_concat)==0):
                y_concat = y_ol
            else:
                y_concat = np.hstack((y_concat, y_ol))

    X_concat_xyz = concat_xyz(X_concat)

    return X_concat_xyz, y_concat

def prepare_pure_label_2(X, y, subj_all, all_subjects, new_label_list):
    X_label, y_label = label_grouping(X, y, subj_all, all_subjects, new_label_list)

    X_concat = []
    y_concat = []
    for i in range(len(X_label)):
        for j in range(len(X_label[i])):
            X_ol, y_ol = make_overlapping(np.array(X_label[i][j]), y_label[i][j])

            if(len(X_concat)==0):
                X_concat = X_ol
            else:
                X_concat = np.vstack((X_concat, X_ol))

            if(len(y_concat)==0):
                y_concat = y_ol
            else:
                y_concat = np.hstack((y_concat, y_ol))

    X_concat_xyz = concat_xyz_2(X_concat)

    return X_concat_xyz, y_concat


# # Reshape Data (Impure Label)

def prepare_impure_label(X, y, window_length=60):
    if(X.shape[0]<window_length):
        return X, y

    X_ol, y_ol = make_overlapping(X, y)
    X_concat_ol = concat_xyz(X_ol)

    return X_concat_ol, y_ol


label_list = ['sit', 'sleep', 'stand', 'walk']
label_dict = {'sit': 0, 'sleep': 1, 'stand': 2, 'walk': 3}
LABELS = label_list