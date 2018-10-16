import numpy as np
import csv

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
# print(prev_ts, prev_sec)
exact_dur = 0.16

new_lines = []
new_elem = []
for elem in all_lines:
	timestamp = elem[0].split(' ')
	x = elem[1]
	y = elem[2]
	z = elem[3]
	if(len(timestamp)!=1):		# exclude header
		date = timestamp[0]
		time = timestamp[1]
		sec = calc_sec(time)

		duration = sec - prev_sec
		duration = round(duration,3)
		# print(duration)

		if(duration!=exact_dur and duration!=0.0 and duration<2*exact_dur):
			sec = prev_sec + exact_dur
		else:
			sec = prev_sec + duration
		new_ts = calc_ts(sec)
		# print(new_ts)
		prev_sec = sec

		elem[0] = date + " " + new_ts
		new_lines.append(elem)

new_lines = np.array(new_lines)
# print(new_lines)

with open('data_activities_eq_time.csv','w') as csv_file:
	writer = csv.writer(csv_file,delimiter=',')
	headers = ['timestamp','x','y','z','label']
	writer.writerow(headers)

	for elem in new_lines:
		writer.writerow(elem)