import matplotlib.dates as mdate
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from datetime import datetime
from datetime import timedelta

ts = []
x = []
y = []
z = []


with open('data_activities_eq_time.csv','r') as csv_file:
	linecount = 0

	for line in csv_file:
		linecount += 1
		if(linecount>1):
			first_elem_0 = line.strip('\n').split(',')[0]
			break

	prev_dt = datetime.strptime(first_elem_0,'%Y-%m-%d %H:%M:%S')
	
	dt_periods = []
	dt_periods.append(prev_dt)

	linecount = 0
	for line in csv_file:
		linecount += 1
		if(linecount>1):
			elem = line.strip('\n').split(',')
			# dt, time = elem[0].split(' ')
			dt = datetime.strptime(elem[0],'%Y-%m-%d %H:%M:%S')
			diff_dt = dt - prev_dt
			time_thres = 3600   # in seconds
			if(diff_dt.seconds>time_thres):
				dt_periods.append(prev_dt)
				dt_periods.append(dt)

			prev_dt = dt

			ts.append(dt)
			x.append(float(elem[1]))
			y.append(float(elem[2]))
			z.append(float(elem[3]))
		# if(linecount==100):
		# 	break
	dt_periods.append(dt)


# print("hehe", np.array(dt_periods), "haha")

npts = np.array(ts)
npx = np.array(x)
npy = np.array(y)
npz = np.array(z)
# print(npts)
npts_1 = npts[2:19]			# 1 for walking downstairs
# print(npts_walking_down)
npts_2 = npts[47:64]			# 2 for walking on the ground
npx_1 = npx[2:19]
npx_2 = npx[47:64]
npy_1 = npy[2:19]
npy_2 = npy[47:64]
npz_1 = npz[2:19]
npz_2 = npz[47:64]

mts = mdate.date2num(npts)
mts_1 = mdate.date2num(npts_1)
mts_2 = mdate.date2num(npts_2)

plt.subplot(211)
plt.plot_date(mts_1,npx_1,'r',label='X')
plt.plot_date(mts_1,npy_1,'g',label='Y')
plt.plot_date(mts_1,npz_1,'b',label='Z')
plt.legend()
plt.title('Walking Downstairs')

plt.subplot(212)
plt.plot_date(mts_2,npx_2,'r',label='X')
plt.plot_date(mts_2,npy_2,'g',label='Y')
plt.plot_date(mts_2,npz_2,'b',label='Z')
plt.legend()
plt.title('Walking on the ground')

plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M:%S"))


plt.show()
