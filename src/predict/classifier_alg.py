#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import os

from predict.detect_peaks import detect_peaks

# ## Classify walk

def classify_walk(xyz, walk_label=3):

    cols = [0,1,2]
    threshold = [0.05, 0.035, 0.04]
    exceed_threshold = 4
    
    walk_label = 3
    nonwalk_label = 0

    walk_pred = [[],[],[]]

    for k in range(len(xyz)):
        
        for cl in range(len(cols)):
            c = cols[cl]

            xyz_k_c = np.transpose(xyz[k])[c]

            peak_idx = detect_peaks(xyz_k_c)    
            valley_idx = detect_peaks(xyz_k_c, valley=True)

            peak_point = [xyz_k_c[j] for j in peak_idx]    
            valley_point = [xyz_k_c[j] for j in valley_idx]

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

    walk_pred = []

    for w in walk_pred_t:
        walk_bool = walk_label
        
        for i in range(3):
            if(w[i]!=walk_label):
                walk_bool = nonwalk_label
                break

        walk_pred.append(walk_bool)
        
    return walk_pred

def combine(xyz, y_pred_svm, walk_label = 3):
    y_pred_walk = classify_walk(xyz)
    y_pred_new = []

    for i in range(len(y_pred_walk)):
        if(y_pred_svm[i]!=walk_label and y_pred_walk[i]==walk_label):
            y_pred_new.append(y_pred_walk[i])
        else:
            y_pred_new.append(y_pred_svm[i])
         
    return y_pred_new

