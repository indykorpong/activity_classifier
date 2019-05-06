#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import pickle
import sys

# path_to_module = '/var/www/html/python/mysql_connect/python_files'

# sys.path.append(path_to_module)
# os.chdir(path_to_module)

# # Set data path

basepath = '/var/www/html/python/mysql_connect/'
    
datapath = basepath + 'DDC_Data/'
mypath = basepath + 'DDC_Data/raw/'

model_path = basepath + 'model/knn_model.pkl'

from os import listdir, walk
from os.path import isfile, join

from load_data.load_methods import calc_ai
from preprocess.preprocessing import prepare_impure_label
from predict.classifier_alg import combine

## Predict label using a trained k-nearest neighbor model

def predict_label(df_all_p_sorted):

    model = pickle.load(open(model_path,'rb'))

    # # Predict Labels

    window_length = 60
    cols = ['x','y','z']
    y_pred_all = []

    df_all_p_sorted['date'] = df_all_p_sorted['timestamp'].apply(lambda x: x.date())
    grouped = df_all_p_sorted.groupby('date')

    df_all_p_sorted['AI'] = pd.Series(calc_ai(df_all_p_sorted))
    df_all_p_sorted = df_all_p_sorted.drop(columns='date')

    for x in grouped:
        label_grp = x[0]

        df_grp = grouped.get_group(label_grp)

        X_all_p = np.array(df_grp[cols].to_dict('split')['data'])
        y_all_p = np.zeros(X_all_p.shape[0])

        X_impure, y_impure = prepare_impure_label(X_all_p, y_all_p)
        
        print('shape', np.array(X_impure).shape)
        # print('isnan', np.isnan(X_impure))

        if(len(X_impure)>0):
            if(X_all_p.shape[0]>=window_length):
                y_pred = model.predict(X_impure)

                y_pred_fill = np.hstack(([y_pred[0] for i in range(window_length-1)], y_pred))

            else:
                y_pred_fill = np.array([0 for i in range(X_all_p.shape[0])])

            y_pred_walk = np.array(combine(X_all_p, y_pred_fill))

            y_pred_all.append(y_pred_walk)
    
    if(len(y_pred_all)>0):
        y_pred_all = np.hstack(y_pred_all)
    else:
        y_pred_all = None
        return df_all_p_sorted
    
    print('y pred all shape, df all p sorted shape')
    print(y_pred_all.shape)
    print(df_all_p_sorted.shape)

    cols = list(df_all_p_sorted.columns.values)

    for i in range(df_all_p_sorted.shape[0]):
        df_all_p_sorted.loc[i, 'y_pred'] = y_pred_all[i]

    df_all_p_sorted['y_pred'] = df_all_p_sorted['y_pred'].astype(int)

    return df_all_p_sorted