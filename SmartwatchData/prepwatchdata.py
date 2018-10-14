import numpy as np
import csv

from os import listdir
from os.path import isfile, join

mypath = '/Users/admin/Desktop/coding/senior_proj_resources/SmartwatchData'
all_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# print(all_files)

acc_files = [f for f in all_files if(f[-3:]=='csv' and f[0:3]=='acc')]
print(acc_files)

all_data_list = []
for a in acc_files:
	f = open(a,"r")

	label = f.name[:-6]
	label = label[4:]
	print(label)

	for line in f:
		e = line.strip('\n').split(',')
		new_e = [e[3],e[0],e[1],e[2]]
		new_e.append(label)
		all_data_list.append(new_e)
	# break

def sortfunc(elem):
	return elem[0]

all_data_list.sort(key=sortfunc)

with open('data_activities.csv','w') as data_file:
	writer = csv.writer(data_file)

	headers = ['timestamp','x','y','z','label']
	writer.writerow(headers)

	for e in all_data_list:
		writer.writerow(e)






