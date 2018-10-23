import numpy as np
import csv

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
for a in acc_files:
	a = "raw_data/" + a
	f = open(a,"r")

	start = "raw_data/acc_"
	stop = "_1.csv"


	label = f.name[:-len(stop)]
	label = label[len(start):]
	print(label)
	all_labels.add(label)

	for line in f:
		e = line.strip('\n').split(',')
		date,time = e[3].split(' ')
		new_e = [date,time,e[0],e[1],e[2]]   # Timestamp, x, y, z
		new_e.append(label)				# Label
		all_data_list.append(new_e)
	# break

with open('prep_data/labels.txt','w') as label_file:
	for item in all_labels:
		label_file.write(item + " ")

def sortfunc(elem):
	return elem[0]		# Sort by timestamp

all_data_list.sort(key=sortfunc)

# print(all_data_list)

with open('prep_data/data_activities.csv','w') as data_file:
	writer = csv.writer(data_file)

	headers = ['date','time','x','y','z','label']
	writer.writerow(headers)

	for e in all_data_list:
		writer.writerow(e)