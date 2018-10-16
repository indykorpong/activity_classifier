import matplotlib.pyplot as plt
import matplotlib.dates as dat
import datetime
import time
import eq_timeintervals

ts = {}
x = {}
y = {}
z = {}
labels = [item.strip(' ').split(' ') for item in open('labels.txt','r')]
print(labels[0])

for lb in labels[0]:
	ts[lb] = []
	x[lb] = []
	y[lb] = []
	z[lb] = []

with open('data_activities_eq_time.csv','r') as csv_file:
	# for lb in lbls[0]:
	count = 0
	for line in csv_file:
		# print(line)
		count+=1
		if(line!='\n' and count!=1):
			elem = line.strip('\n').split(',')
			# print(elem)
			ets = elem[0]
			ex = elem[1]
			ey = elem[2]
			ez = elem[3]
			lbl = elem[4]
			ts[lbl].append(ets)
			x[lbl].append(ex)
			y[lbl].append(ey)
			z[lbl].append(ez)

	for lbl in labels[0]:
		first_hms = ts[lbl][0].split(' ')[1]
		first_sec = eq_timeintervals.calc_sec(first_hms)
		for t in ts[lbl]:
			d , hms = t.split(' ')
			sec = eq_timeintervals.calc_sec(hms)
			diff_sec = sec - first_sec
			print(round(diff_sec,3))
	