import numpy
import math

from os import listdir
from os.path import isfile, join

mypath = '/Users/admin/Desktop/coding/senior_proj_resources/UCI HAR Dataset/train/Inertial Signals'
all_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for f in all_files:
	if(f[0:9]=='total_acc'):
		print(f)