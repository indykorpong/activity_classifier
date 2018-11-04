import csv
import numpy as np
import math
from os import listdir
from os.path import isfile, join

mypath = '/Users/admin/Desktop/coding/Dementia_proj/SmartwatchData/raw_data'
all_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# print(all_files)

acc_files = [f for f in all_files if(f[-3:]=='csv' and f[0:3]=='acc')]
hr_files = [f for f in all_files if(f[-3:]=='csv' and f[0:2]=='hr')]
print(acc_files)
print(hr_files)

all_data_list = []
all_labels = set()
for a in hr_files:
    a = "raw_data/" + a
    f = open(a,"r")

    start = "raw_data/hr_"
    stop = "_1.csv"

    label = f.name[:-len(stop)]
    label = label[len(start):]
    print(label)
    all_labels.add(label)

    for line in f:
        e = line.strip('\n').split(',')
        date, time = e[1].split(' ')
        new_e = [date,time,e[0],label]
        all_data_list.append(new_e)
    # break

# with open('prep_data/labels_hr.txt','w') as label_file:
#     for item in all_labels:
#         label_file.write(item + " ")

def sortfunc(elem):
    return elem[0]     # Sort by timestamp

all_data_list.sort(key=sortfunc)

all_data = np.array(all_data_list)
print(all_data)

with open('prep_data/data_heartrate.csv','w') as data_file:
    writer = csv.writer(data_file,delimiter=',')

    headers = ['date','time','heart rate','label']
    writer.writerow(headers)

    for e in all_data_list:
        writer.writerow(e)

def calc_sec(time):
    hms = time.split(':')
    hms = [float(x) for x in hms]
    sec = hms[2] + hms[1]*60 + hms[0]*3600
    sec = round(sec,3)
    return sec

hr_idx = 0
all_elem = []
no_hr = 0
with open('prep_data/data_activities_eq_all.csv','r') as act_file:
    for line in act_file:
        elem = line.strip('\n').split(',')
        d = elem[0]
        t = elem[1]
        if(t!='time'):
            if(d==all_data[hr_idx,0]):
                if(math.floor(calc_sec(all_data[hr_idx,1]))>calc_sec(t) and hr_idx<len(all_data)-1):
                    hr_idx += 1
                    print(d,"yay")
            if(d!=all_data[hr_idx,0]):
                hr_idx -= 1
                print(d,"sad")


            new_elem = [d,t,elem[2],elem[3],elem[4],all_data[hr_idx,2],elem[5]]
            if(hr_idx==len(all_data)-1):
                new_elem = [d,t,elem[2],elem[3],elem[4],no_hr,elem[5]]
            all_elem.append(new_elem)

# np_all_elem = np.array(all_elem)
# print(np_all_elem)

with open('prep_data/data_act_and_hr.csv','w') as data_file:
    writer = csv.writer(data_file,delimiter=',')

    headers = ['date','time','x','y','z','heart rate','label']
    writer.writerow(headers)

    for e in all_elem:
        writer.writerow(e)





