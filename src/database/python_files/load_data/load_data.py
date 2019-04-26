#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import sys
import csv

from os import listdir, walk
from os.path import isfile, join

from sklearn.preprocessing import MinMaxScaler

path_to_module = '/var/www/html/python/mysql_connect/python_files'

sys.path.append(path_to_module)
os.chdir(path_to_module)

# # Set data path

basepath = '/var/www/html/python/mysql_connect/'
    
datapath = basepath + 'DDC_Data/'
mypath = basepath + 'DDC_Data/raw/'

from load_data.load_methods import merge_acc_and_hr
from insert_db.insert_db import get_patients_acc_hr


def load_raw_data(user_id):

    df_acc, df_hr = get_patients_acc_hr(user_id)
    print(df_acc.head())
    print(df_hr.head())

    df1 = merge_acc_and_hr(df_acc, df_hr)

    if(not df1.empty):
        cols = ['x','y','z']
        xyz_ = df1[cols].to_dict('split')['data']
        xyz_new = MinMaxScaler().fit_transform(xyz_)

        for i in range(len(cols)):
            df1[cols[i]] = pd.Series(xyz_new.transpose()[i])

        print('Finished Loading')

        df1 = df1.reset_index(drop=True)

    return df1