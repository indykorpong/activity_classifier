#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import sys
import csv

from os import listdir, walk
from os.path import isfile, join

from decimal import Decimal, getcontext

# from sklearn.preprocessing import MinMaxScaler

# path_to_module = '/var/www/html/python/mysql_connect/python_files'

# sys.path.append(path_to_module)
# os.chdir(path_to_module)

# # # Set data path

# basepath = '/var/www/html/python/mysql_connect/'
    
# datapath = basepath + 'DDC_Data/'
# mypath = basepath + 'DDC_Data/raw/'

from load_data.load_methods import merge_acc_and_hr
from insert_db.insert_db import get_patients_acc_hr, get_user_profile

def min_max_scale(user_id, df_xyz):
    df_new = df_xyz.copy()

    [min_x, min_y, min_z, max_x, max_y, max_z] = get_user_profile(user_id)[0]
    if(min_x==0 and min_y==0 and min_z==0 and 
        max_x==0 and max_y==0 and max_z==0):
        min_x = df_new['x'].min()
        min_y = df_new['y'].min()
        min_z = df_new['z'].min()

        max_x = df_new['x'].max()
        max_y = df_new['y'].max()
        max_z = df_new['z'].max()

    getcontext().prec = 8
    df_new['x'] = df_new['x'].apply(lambda x: (Decimal(x)-min_x)/(max_x-min_x))
    df_new['y'] = df_new['y'].apply(lambda x: (Decimal(x)-min_y)/(max_y-min_y))
    df_new['z'] = df_new['z'].apply(lambda x: (Decimal(x)-min_z)/(max_z-min_z))

    return df_new

def load_raw_data(user_id):

    df_acc, df_hr = get_patients_acc_hr(user_id)

    if(not df_acc.empty):
        cols = ['x','y','z']
        df_acc[cols] = min_max_scale(user_id, df_acc[cols])

        df1 = merge_acc_and_hr(df_acc, df_hr)
        print('Finished loading data')

        df1 = df1.reset_index(drop=True)
        print(df1.head())

        return df1
    
    return pd.DataFrame()