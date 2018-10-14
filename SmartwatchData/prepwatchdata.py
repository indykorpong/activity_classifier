import numpy as np
import csv

from os import listdir
from os.path import isfile, join

mypath = 'C:/Users/Indy/Desktop/Coding/Dementia_proj/SmartwatchData'
all_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
# print(all_files)

acc_files = [f for f in all_files if(f[-3:]=='csv' and f[0:3]=='acc')]
hr_files = [f for f in all_files if(f[-3:]=='csv' and f[0:2]=='hr')]
print(acc_files)
print(hr_files)

all_data_list = []
for a in acc_files:
	f = open(a,"r")

	label = f.name[:-6]
	label = label[4:]
	print(label)

	for line in f:
		e = line.strip('\n').split(',')
		new_e = [e[3],e[0],e[1],e[2]]   # Timestamp, x, y, z
		new_e.append(label)				# Label
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

all_lines = []
with open('data_activities.csv','r') as data_file:
	for line in data_file:
		line = line.strip('\n')
		line = line.split(',')
		if(line!=['']):
			all_lines.append(line)
	
all_lines = np.array(all_lines)
print(all_lines)

def calc_sec(time):
	hms = time.split(':')
	hms = [float(x) for x in hms]
	sec = hms[2] + hms[1]*60 + hms[0]*3600
	sec = round(sec,3)
	return sec

def calc_ts(sec):
	ts = ''
	hr = int(sec/3600)
	mn = int((sec - (hr*3600))/60)
	sc = sec - (hr*3600) - (mn*60)
	sc = round(sc,3)
	ts += str(hr) + ':' + str(mn) + ':' + str(sc)
	# print(ts)
	return ts

prev_ts = all_lines[1][0].split(' ')[1]
prev_sec = calc_sec(prev_ts)
kept_sec = prev_sec
print(prev_ts, prev_sec)
exact_dur = 0.16

# still not complete
for elem in all_lines:
	timestamp = elem[0].split(' ')
	x = elem[1]
	y = elem[2]
	z = elem[3]
	if(len(timestamp)!=1):
		time = timestamp[1]
		# print(time)
		sec = calc_sec(time)

		duration = sec - prev_sec
		duration = round(duration,3)
		print(duration)

		if(duration!=exact_dur and duration!=0.0 and duration<2*exact_dur):
			sec = prev_sec + exact_dur
		elif(duration>=2*exact_dur and duration<3*exact_dur):

		else:
			sec = prev_sec + duration
		new_ts = calc_ts(sec)
		print(new_ts)

		prev_sec = sec
# calc_ts(52948.999)






