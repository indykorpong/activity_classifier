#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from decimal import Decimal, getcontext
from sklearn.preprocessing import MinMaxScaler

def merge_acc_and_hr(df_filt, df_hr):
    # Fill in missing HRs
    hr_cnt = 0

    # If HR data is missing, then fill in the HR field with None
    if(not df_hr.empty):
        for i in range(len(df_filt)):
            hr_time = df_hr.loc[hr_cnt,'timestamp']
            filt_time = df_filt.loc[i,'timestamp']

            if(hr_time<=filt_time):
                if(hr_cnt<len(df_hr)-1):
                    hr_cnt += 1
            df_filt.loc[i,'HR'] = df_hr.loc[hr_cnt,'hr']
    else:
        df_filt['HR'] = pd.Series([0 for i in range(df_filt.shape[0])])

    if(not df_filt.empty):
        # Divide by g (standard gravity) and normalize by min-max scaling
        getcontext().prec = 4
        g = Decimal(9.8)
        df_filt.loc[:,'x'] = df_filt['x'].apply(lambda x: x/g)
        df_filt.loc[:,'y'] = df_filt['y'].apply(lambda x: x/g)
        df_filt.loc[:,'z'] = df_filt['z'].apply(lambda x: x/g)

        cols = ['x','y','z']
        xyz = df_filt[cols].to_dict('split')['data']
        xyz_new = MinMaxScaler().fit_transform(xyz)

        for i in range(len(cols)):
            df_filt[cols[i]] = pd.Series(xyz_new.transpose()[i])

    return df_filt

std_i_bar = [Decimal(0.00349329),Decimal(0.00465817),Decimal(0.00543154)]
std_i_bar = np.array(std_i_bar)

def equation_bai(X_i):
    all_std = []

    std_i = np.std(X_i,axis=0)
    diff_std = std_i**2 - std_i_bar**2
    diff_std = (diff_std + 1) / (std_i_bar**2 + 1)

    all_std.append(diff_std)

    all_std = np.array(all_std)

    ai = np.sum(all_std**2,axis=1)/3
    ai[ai<0] = 0
    ai = np.sqrt(ai)

    return ai

def calc_ai(df1):
    H = 10
    ai1 = []

    for i in range(len(df1)):
        xyz_val = []
        if(i-H>=0):
            for j in range(H,0,-1):
                xyz_val.append([df1.loc[i-j,'x'],df1.loc[i-j,'y'],df1.loc[i-j,'z']])

            # print('xyz val:', xyz_val)
            ai_val = float(equation_bai(xyz_val))
            ai1.append(ai_val)
        else:
            ai1.append(1)

    return ai1