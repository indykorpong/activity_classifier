import numpy as np
import csv
import math

all_lines = []
with open('prep_data/data_activities.csv','r') as data_file:
	for line in data_file:
		line = line.strip('\n')
		line = line.split(',')
		if(line!=['']):
			all_lines.append(line)
	
all_lines = np.array(all_lines)
# print(all_lines)

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
trimmed_mean_list = []
trimmed_mean_elem = []

for elem in all_lines:
	date = elem[0]
	time = elem[1]
	if(date!='date'):		# Exclude header
		x = elem[1]
		y = elem[2]
		z = elem[3]
		label = elem[4]

		sec = calc_sec(time)

		duration = sec - prev_sec
		duration = round(duration,3)
		# print(duration)

		if(duration!=exact_dur and duration!=0.0 and duration<2*exact_dur):
			new_sec = prev_sec + exact_dur
		else:
			new_sec = prev_sec + duration

		diff_sec = math.floor(new_sec)-math.floor(prev_sec)
		if(diff_sec>=1):									# If it's the first millisecond in timestamp
			trimmed_mean_np = np.array(trimmed_mean_list)
			# print(trimmed_mean_np)
			x_item = sorted([float(item[1]) for item in trimmed_mean_np])
			y_item = sorted([float(item[2]) for item in trimmed_mean_np])
			z_item = sorted([float(item[3]) for item in trimmed_mean_np])

			x_item = np.array(x_item[1:-1])			# Drop the min,max data in the list
			y_item = np.array(y_item[1:-1])
			z_item = np.array(z_item[1:-1])

			# print(math.floor(prev_sec))

			if(len(x_item)>0):
				elem[1] = round(np.mean(x_item),8)		# Calculate mean value on each axis
				elem[2] = round(np.mean(y_item),8)
				elem[3] = round(np.mean(z_item),8)
			else:
				elem[1] = trimmed_mean_np[0,1]
				elem[2] = trimmed_mean_np[0,2]
				elem[3] = trimmed_mean_np[0,3]

			prev_sec = math.floor(prev_sec)
			new_ts = calc_ts(prev_sec)
			# print(new_ts)

			elem[0] = date + " " + new_ts
			new_lines.append(elem)

			trimmed_mean_list.clear()

			prev_sec = new_sec
			# print(prev_sec)

		trimmed_mean_elem = [new_sec,x,y,z,label]
		trimmed_mean_list.append(trimmed_mean_elem)

new_lines = np.array(new_lines)
print(new_lines)

with open('prep_data/data_activities_eq_time.csv','w') as csv_file:
	writer = csv.writer(csv_file,delimiter=',')
	headers = ['timestamp','x','y','z','label']
	writer.writerow(headers)

	for elem in new_lines:
		writer.writerow(elem)